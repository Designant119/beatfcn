"""Complete event E-D with separate local, whole-song and visible-reference encoders.

Frozen frontend tensors are inputs. Hidden chart attributes are never required
by encode(). This module does not select source data or claim benchmark quality.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import torch
from torch import nn
from torch.nn import functional as F

from beatfcn.events.canonical import TYPES
from .full_event_codec import FIELDS, EventVocabulary

# Onset head channels: [any onset] + one per canonical type (tap/hold/flick/drag).
ONSET_CHANNELS = 1 + len(TYPES)
# Charter-timing (beat grid) features per 10 ms mel frame (beatfcn.train.charter_timing).
BEAT_FEATURE_DIM = 12


@dataclass(frozen=True)
class FullEventConfig:
    d: int = 768
    heads: int = 12
    kv_heads: int = 4
    ffn: int = 2816
    encoder_layers: int = 3
    decoder_layers: int = 18
    tap_dim: int = 1024
    taps: int = 3
    bands: int = 8
    band_slots: int = 2
    reference_dim: int = 12
    control_dim: int = 6
    controls: int = 8
    dropout: float = .1
    grad_checkpointing: bool = False
    # Optional candidates; the historical Transformer configuration stays loadable.
    history_mixer: str = 'attention'
    decoder_ffn: int | None = None
    hybrid_attention_every: int = 4
    ssm_state: int = 128
    ssm_headdim: int = 64
    mel_bins: int = 0
    intensity_dim: int = 0
    qk_norm: bool = False
    learnable_audio_scale: bool = False
    # Opt-in review fixes (all default off so historical checkpoints load and
    # reproduce bit-for-bit):
    # timed_intensity: intensity/density curve rows enter memory at their own
    #   request-relative times (rotary-timed where intensity_valid), instead of
    #   all at t=0 untimed, so cross-attention can localise requested density.
    timed_intensity: bool = False
    # timed_memory_values: cross-attention values carry the key's time offset
    #   relative to the query (VO-RoPE style), not only the q/k scores.
    timed_memory_values: bool = False
    # onset_head: per-frame onset (+ per-type) logits on the encoded local
    #   audio stream as an audio-grounding auxiliary (see training_objective).
    onset_head: bool = False
    # beat_features: optional charter timing (BPM/offset/meter) as per-mel-frame
    #   features (inputs['beat'] [B,T_mel,BEAT_FEATURE_DIM]) added to the mel
    #   tokens through a zero-initialised beat_projection. A missing 'beat' key
    #   is the absent state (all-zero features, presence 0).
    beat_features: bool = False

    def __post_init__(self):
        if (self.d % self.heads or self.heads % self.kv_heads
                or (self.d // self.heads) % 4 or not 0 <= self.dropout < 1):
            raise ValueError('Invalid attention dimensions or dropout')
        if self.history_mixer not in ('attention', 'mamba2', 'hybrid'):
            raise ValueError('Unknown event history mixer')
        if (self.hybrid_attention_every < 1 or self.ssm_headdim < 1
                or self.ssm_state < 1 or self.mel_bins < 0 or self.intensity_dim < 0
                or (self.decoder_ffn is not None and self.decoder_ffn < 1)):
            raise ValueError('Invalid candidate dimensions')
        if self.history_mixer != 'attention' and 2 * self.d % self.ssm_headdim:
            raise ValueError('Mamba-2 inner dimension must be divisible by head dimension')
        if self.beat_features and not self.mel_bins:
            raise ValueError('beat_features are per 10ms mel frame and need mel_bins > 0')


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6).to(x.dtype) * self.weight


@lru_cache(maxsize=32)
def _mps_periods(lo, hi, half):
    # Torch 2.6 MPS lacks logspace. This is a fixed configuration constant,
    # constructed once, not a CPU fallback or data-dependent transfer per step.
    with torch.inference_mode(False):
        return torch.logspace(math.log10(lo), math.log10(hi), half, device='cpu').to('mps')


@lru_cache(maxsize=64)
def _cached_periods(lo, hi, half, device):
    # Same fp32 logspace the old code rebuilt on every call (~90x per decode
    # step); built outside inference_mode so a first call under inference_mode
    # cannot poison later autograd use.
    with torch.inference_mode(False):
        return torch.logspace(math.log10(lo), math.log10(hi), half, device=device)


def rotary_periods(lo, hi, half, device):
    device = torch.device(device)
    if device.type == 'mps':
        return _mps_periods(lo, hi, half)
    if torch.compiler.is_compiling() or (device.type == 'cuda' and torch.cuda.is_current_stream_capturing()):
        # Never populate a process-wide cache from a traced/captured region.
        return torch.logspace(math.log10(lo), math.log10(hi), half, device=device)
    return _cached_periods(lo, hi, half, device)


def rotary_tables(positions, *, lo, hi, half, timed=None):
    """fp32 (cos, sin) [B,1,N,half]; identical arithmetic to the old rotate()."""
    periods = rotary_periods(lo, hi, half, positions.device)
    angle = positions.float()[:, None, :, None] * (2 * math.pi / periods)
    if timed is not None:
        angle = angle * timed[:, None, :, None]
    return angle.cos(), angle.sin()


def apply_rotary(x, tables, *, inverse=False):
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    cos, sin = tables[0].to(x.dtype), tables[1].to(x.dtype)
    if inverse:
        sin = -sin
    return torch.cat((a * cos - b * sin, b * cos + a * sin), -1)


def rotate(x, positions, *, lo, hi, timed=None):
    return apply_rotary(x, rotary_tables(positions, lo=lo, hi=hi, half=x.shape[-1] // 2, timed=timed))


def aligned_attention(q, k, v, mask, dropout_p):
    """Zero-pad heads for fused SDPA while retaining the original score scale."""
    # A wider value (timed_memory_values) is supported by zero-padding q/k to
    # the common width: zero dims leave q.k unchanged and the scale is explicit.
    # (Torch 2.6 MPS SDPA silently assumes value width == q width.)
    width, value_width = q.shape[-1], v.shape[-1]
    common = max(width, value_width)
    common += (-common) % 8
    if common != width:
        q, k = (F.pad(tensor, (0, common - width)) for tensor in (q, k))
    if common != value_width:
        v = F.pad(v, (0, common - value_width))
    result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
        dropout_p=dropout_p, scale=width ** -.5)
    return result[..., :value_width]


class Attention(nn.Module):
    def __init__(self, c, *, sequence=False, value_time=False):
        super().__init__()
        self.c, self.sequence = c, sequence
        # value_time (cross-attention only): rotary on q/k makes WHERE to look
        # time-aware, but the returned mixture had no notion of WHEN the
        # attended rows are. We rotate each timed value by its key time and
        # inverse-rotate the output by the query time, so the residual stream
        # receives sum_k a_k R(t_k - t_q) v_k: a translation-invariant relative
        # offset, consistent with the request-relative clock (a learned absolute
        # time projection would re-introduce absolute position). Untimed rows
        # (controls, empty reference, unspecified curve cells) must stay
        # unrotated, so values are carried as [timed-rotated | untimed] halves
        # and only the timed half is inverse-rotated before summing. Costs 2x
        # value width in cross-attention only.
        self.value_time = value_time
        hd = c.d // c.heads
        self.q = nn.Linear(c.d, c.d, bias=False)
        self.k = nn.Linear(c.d, c.kv_heads * hd, bias=False)
        self.v = nn.Linear(c.d, c.kv_heads * hd, bias=False)
        self.out = nn.Linear(c.d, c.d, bias=False)
        if c.qk_norm:
            self.q_norm, self.k_norm = RMSNorm(hd), RMSNorm(hd)

    def _position(self, x, times, positions, timed=None, rope=None):
        """``rope``: optional precomputed fp32 tables shared by all layers of
        one decode call (dict with 'time_half', 'pos_half', 'time_full');
        only valid for untimed-mask queries/keys, numerically identical."""
        if self.sequence:
            a, b = x.chunk(2, -1)
            if rope is not None and timed is None:
                return torch.cat((apply_rotary(a, rope['time_half']), apply_rotary(b, rope['pos_half'])), -1)
            return torch.cat((rotate(a, times, lo=.02, hi=2400, timed=timed),
                              rotate(b, positions, lo=2, hi=65536)), -1)
        if rope is not None and timed is None:
            return apply_rotary(x, rope['time_full'])
        return rotate(x, times, lo=.02, hi=2400, timed=timed)

    def project_kv(self, x, times, positions, timed=None, rope=None):
        b, n, _ = x.shape
        k, v = (layer(x).view(b, n, self.c.kv_heads, -1).transpose(1, 2)
                for layer in (self.k, self.v))
        if self.c.qk_norm:
            k = self.k_norm(k)
        if self.value_time:
            flag = (torch.ones_like(times, dtype=torch.bool) if timed is None else timed.bool())
            weight = flag[:, None, :, None].to(v.dtype)
            v = torch.cat((rotate(v, times, lo=.02, hi=2400, timed=timed) * weight, v * (1 - weight)), -1)
        return self._position(k, times, positions, timed, rope), v

    def forward(self, x, query_times, query_positions, kv, mask, rope=None):
        b, n, _ = x.shape
        q = self.q(x).view(b, n, self.c.heads, -1).transpose(1, 2)
        if self.c.qk_norm:
            q = self.q_norm(q)
        q = self._position(q, query_times, query_positions, rope=rope)
        k, v = kv
        # Explicit expansion is the CPU/CUDA reference. A fused GQA path can
        # only replace it after equivalent-mask/numerical checks on that backend.
        repeat = self.c.heads // self.c.kv_heads
        y = aligned_attention(q, k.repeat_interleave(repeat, 1),
                v.repeat_interleave(repeat, 1), mask,
                self.c.dropout if self.training else 0.)
        if self.value_time:
            timed_part, untimed_part = y.chunk(2, -1)
            tables = (rope['time_full'] if rope is not None else
                      rotary_tables(query_times, lo=.02, hi=2400, half=timed_part.shape[-1] // 2))
            y = apply_rotary(timed_part, tables, inverse=True) + untimed_part
        return self.out(y.transpose(1, 2).reshape(b, n, self.c.d))


    def forward_grouped(self, x, query_times, query_positions, kv, mask, rope=None):
        """Inference-only single-query forward without K/V head expansion.

        The ``heads // kv_heads`` query heads sharing one K/V head are folded
        into the query axis ([B,kv,group,hd]); attention per (row, head) is the
        same dot products and softmax as ``forward``'s repeat_interleave path,
        only the kernel's reduction blocking may differ (fp32 round-off). The
        training/reference ``forward`` is untouched.
        """
        b, n, _ = x.shape
        if n != 1 or self.training:
            raise ValueError('forward_grouped is an eval-mode single-query path')
        c = self.c
        q = self.q(x).view(b, n, c.heads, -1).transpose(1, 2)
        if c.qk_norm:
            q = self.q_norm(q)
        q = self._position(q, query_times, query_positions, rope=rope)
        k, v = kv
        group = c.heads // c.kv_heads
        y = aligned_attention(q.reshape(b, c.kv_heads, group, q.shape[-1]), k, v, mask, 0.)
        y = y.reshape(b, c.heads, n, y.shape[-1])
        if self.value_time:
            timed_part, untimed_part = y.chunk(2, -1)
            tables = (rope['time_full'] if rope is not None else
                      rotary_tables(query_times, lo=.02, hi=2400, half=timed_part.shape[-1] // 2))
            y = apply_rotary(timed_part, tables, inverse=True) + untimed_part
        return self.out(y.transpose(1, 2).reshape(b, n, c.d))

class FeedForward(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.up = nn.Linear(c.d, 2 * c.ffn, bias=False)
        self.down = nn.Linear(c.ffn, c.d, bias=False)

    def forward(self, x):
        a, b = self.up(x).chunk(2, -1)
        return self.down(F.silu(a) * b)


class EncoderBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(c.d), RMSNorm(c.d)
        self.attn, self.ffn, self.drop = Attention(c), FeedForward(c), nn.Dropout(c.dropout)

    def forward(self, x, times, valid):
        z = self.norm1(x)
        positions = torch.arange(x.shape[1], device=x.device)[None].expand(x.shape[0], -1)
        x = x + self.drop(self.attn(z, times, positions, self.attn.project_kv(z, times, positions),
                                    valid[:, None, None, :]))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x * valid[..., None]


class DecoderBlock(nn.Module):
    def __init__(self, c, index=0):
        super().__init__()
        from dataclasses import replace
        self.norm1, self.norm2, self.norm3 = (RMSNorm(c.d) for _ in range(3))
        self.is_ssm = c.history_mixer == 'mamba2' or (c.history_mixer == 'hybrid'
            and (index + 1) % c.hybrid_attention_every != 0)
        if self.is_ssm:
            from .event_ssm import EventMamba2
            self.history = EventMamba2(c, index)
        else:
            self.self_attn = Attention(c, sequence=True)
        self.cross_attn = Attention(c, value_time=c.timed_memory_values)
        self.ffn = FeedForward(replace(c, ffn=c.decoder_ffn or c.ffn))
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, times, positions, memory_kv, memory_valid, token_valid, cache=None, rope=None):
        z = self.norm1(x)
        if self.is_ssm:
            x = x + self.drop(self.history(z, token_valid, cache))
            x = x + self.drop(self.cross_attn(self.norm2(x), times, positions, memory_kv,
                                             memory_valid[:, None, None, :], rope))
            return x + self.drop(self.ffn(self.norm3(x)))
        k, v = self.self_attn.project_kv(z, times, positions, rope=rope)
        if cache is not None:
            if cache.get('kv') is not None:
                oldk, oldv = cache['kv']
                k, v = torch.cat((oldk, k), 2), torch.cat((oldv, v), 2)
            cache['kv'] = (k, v)
            # Queries carry absolute sequence positions. Non-square is_causal
            # would align upper-left and hide most of a preexisting KV cache.
            allowed = torch.arange(k.shape[2], device=x.device)[None, :] <= positions[:, :, None]
            mask = allowed[:, None]
        else:
            n = x.shape[1]
            causal = torch.arange(n, device=x.device)[None, :] <= torch.arange(n, device=x.device)[:, None]
            mask = causal[None, None] & token_valid[:, None, None, :]
        x = x + self.drop(self.self_attn(z, times, positions, (k, v), mask, rope))
        x = x + self.drop(self.cross_attn(self.norm2(x), times, positions, memory_kv,
                                         memory_valid[:, None, None, :], rope))
        return x + self.drop(self.ffn(self.norm3(x)))


class FullEventModel(nn.Module):
    def __init__(self, config=None, vocab=None):
        super().__init__()
        self.c, self.vocab = config or FullEventConfig(), vocab or EventVocabulary()
        c = self.c
        self.layer_mix = nn.Parameter(torch.zeros(c.taps))
        if c.learnable_audio_scale:
            self.local_input_scale = nn.Parameter(torch.ones(c.taps, c.tap_dim))
            self.song_input_scale = nn.Parameter(torch.ones(c.tap_dim))
        self.local_projection = nn.Linear(c.tap_dim, c.d)
        self.band_embedding = nn.Parameter(torch.randn(c.bands, c.d) * .02)
        self.band_queries = nn.Parameter(torch.randn(c.band_slots, c.d) * .02)
        self.song_projection = nn.Linear(c.tap_dim, c.d)
        self.reference_projection = nn.Linear(c.reference_dim, c.d)
        self.control_projection = nn.Linear(c.control_dim, c.d)
        self.control_ids = nn.Embedding(c.controls, c.d)
        self.empty_reference = nn.Parameter(torch.zeros(1, 1, c.d))
        self.source_ids = nn.Embedding(4, c.d)
        self.mode_ids = nn.Embedding(2, c.d)
        if c.mel_bins:
            self.mel_projection = nn.Linear(c.mel_bins, c.d)
            self.mel_source = nn.Parameter(torch.randn(c.d) * .02)
        if c.intensity_dim:
            self.intensity_projection = nn.Linear(c.intensity_dim, c.d)
            self.intensity_source = nn.Parameter(torch.randn(c.d) * .02)
            if c.timed_intensity:
                # Marks an attendable-but-unspecified curve cell, so it differs
                # from a specified zero-density cell.
                self.intensity_unspecified = nn.Parameter(torch.randn(c.d) * .02)
        if c.onset_head:
            # A deliberately small (linear-probe sized) head: its job is to force
            # and to measure onset evidence in the local encoder, not to be a
            # separate onset detector.
            self.onset_norm = RMSNorm(c.d)
            self.onset_projection = nn.Linear(c.d, ONSET_CHANNELS)
        self.encoders = nn.ModuleList(nn.ModuleList(EncoderBlock(c) for _ in range(c.encoder_layers))
                                      for _ in range(3))
        self.token_embedding = nn.Embedding(self.vocab.size, c.d, padding_idx=0)
        self.field_embedding = nn.Embedding(len(FIELDS), c.d)
        self.decoder = nn.ModuleList(DecoderBlock(c, i) for i in range(c.decoder_layers))
        self.final_norm = RMSNorm(c.d)
        self.output = nn.Linear(c.d, self.vocab.size, bias=False)
        if c.beat_features:
            # Created last and outside the global RNG stream, so every other
            # parameter (and the caller's RNG) is identical to beat_features=False.
            with torch.random.fork_rng(devices=[]):
                self.beat_projection = nn.Linear(BEAT_FEATURE_DIM, c.d)
            nn.init.zeros_(self.beat_projection.weight)
            nn.init.zeros_(self.beat_projection.bias)

    def encode(self, inputs):
        """Only audio, authorized reference, explicit requests and mode enter here.

        local [B,tap,T,band,D], song [B,S,D], reference [B,R,16],
        controls [B,8,6]. All times are seconds relative to the requested core.
        Empty/unknown reference regions are explicit rows, never hidden events.
        """
        c = self.c
        local = inputs['local']
        if local.ndim != 5 or local.shape[1] != c.taps or local.shape[3:] != (c.bands, c.tap_dim):
            raise ValueError('Local frontend tensor contract mismatch')
        b, _, n, _, _ = local.shape
        normalized = F.layer_norm(local.float(), (c.tap_dim,))
        if c.learnable_audio_scale:
            normalized = normalized*self.local_input_scale[None, :, None, None, :]
        mixed = (normalized * self.layer_mix.softmax(0)[None, :, None, None, None]).sum(1)
        band = self.local_projection(mixed) + self.band_embedding
        weights = torch.einsum('btfd,qd->btqf', band, self.band_queries).float().div(math.sqrt(c.d)).softmax(-1).to(band.dtype)
        local_tokens = torch.einsum('btqf,btfd->btqd', weights, band).reshape(b, n * c.band_slots, c.d)
        local_times = inputs['local_times'].repeat_interleave(c.band_slots, 1)
        local_valid = inputs['local_valid'].repeat_interleave(c.band_slots, 1)
        if c.mel_bins:
            mel = inputs['mel']
            if mel.ndim != 3 or mel.shape[-1] != c.mel_bins:
                raise ValueError('10ms mel tensor contract mismatch')
            # Keep the 10ms stream at its actual resolution and clock. It is
            # bidirectional audio evidence, never a resampled target sequence.
            detail = self.mel_projection(mel) + self.mel_source
            if c.beat_features:
                detail = detail + self.beat_projection(self._beat_inputs(inputs, mel))
            local_tokens = torch.cat((local_tokens, detail), 1)
            local_times = torch.cat((local_times, inputs['mel_times']), 1)
            local_valid = torch.cat((local_valid, inputs['mel_valid']), 1)
        curve_tokens = None
        if c.intensity_dim:
            intensity = inputs['intensity']
            if intensity.ndim != 3 or intensity.shape[-1] != c.intensity_dim:
                raise ValueError('Intensity curve tensor contract mismatch')
            curve_tokens = (self.intensity_projection(intensity) + self.intensity_source
                            + self.source_ids.weight[3])
            curve_specified = inputs['intensity_valid'].to(device=local.device, dtype=torch.bool)
            if c.timed_intensity:
                # Optional 'intensity_present' = real (non-padding) cells. Present
                # but unspecified cells stay attendable (untimed + marker); when it
                # is absent, unspecified and padding are indistinguishable, so they
                # stay masked exactly as before (padding-safe default).
                curve_attend = inputs.get('intensity_present')
                curve_attend = (curve_specified if curve_attend is None else
                                curve_attend.to(device=local.device, dtype=torch.bool) | curve_specified)
                curve_tokens = curve_tokens + (~curve_specified)[..., None].to(curve_tokens.dtype) * self.intensity_unspecified
        song = F.layer_norm(inputs['song'].float(), (c.tap_dim,))
        if c.learnable_audio_scale:
            song = song*self.song_input_scale
        song = self.song_projection(song)
        reference = self.reference_projection(inputs['reference'])
        reference = torch.cat((self.empty_reference.expand(b, -1, -1), reference), 1)
        rt = torch.cat((torch.zeros(b, 1, device=local.device), inputs['reference_times']), 1)
        rv = torch.cat((torch.ones(b, 1, device=local.device, dtype=torch.bool), inputs['reference_valid']), 1)
        streams = [(local_tokens, local_times, local_valid),
                   (song, inputs['song_times'], inputs['song_valid']), (reference, rt, rv)]
        memory, times, valid, timed = [], [], [], []
        for i, (x, t, mask) in enumerate(streams):
            x = x + self.source_ids.weight[i]
            for encoder in self.encoders[i]:
                if c.grad_checkpointing and self.training:
                    import torch.utils.checkpoint as cp
                    x = cp.checkpoint(encoder, x, t, mask, use_reentrant=False)
                else:
                    x = encoder(x, t, mask)
            memory.append(x); times.append(t); valid.append(mask)
            if i == 0 and c.onset_head:
                onset = self._onset_outputs(x, inputs, n)
            source_timed = mask.clone()
            if i == 2:
                source_timed[:, 0] = False
            timed.append(source_timed)
        controls = self.project_controls(inputs['controls'])
        if curve_tokens is not None and c.timed_intensity:
            memory.append(curve_tokens)
            times.append(inputs['intensity_times'].to(device=local.device, dtype=torch.float32)
                         * curve_specified)
            valid.append(curve_attend)
            timed.append(curve_specified & curve_attend)
            curve_tokens = None
        condition_valid = torch.ones(b, c.controls, device=local.device, dtype=torch.bool)
        if curve_tokens is not None:
            controls = torch.cat((curve_tokens, controls), 1)
            condition_valid = torch.cat((curve_specified, condition_valid), 1)
        condition_tokens = controls.shape[1]
        # Controls must remain the LAST c.controls memory rows (replace_controls).
        memory.append(controls)
        times.append(torch.zeros(b, condition_tokens, device=local.device))
        valid.append(condition_valid)
        timed.append(torch.zeros(b, condition_tokens, device=local.device, dtype=torch.bool))
        result = dict(values=torch.cat(memory, 1), times=torch.cat(times, 1),
                      valid=torch.cat(valid, 1), timed=torch.cat(timed, 1), mode=inputs['mode'])
        if c.onset_head:
            result.update(onset)
        return result

    def _beat_inputs(self, inputs, mel):
        """[B,T_mel,BEAT_FEATURE_DIM]; absent (no 'beat' key) = zeros. Optional 'beat_valid' [B,T_mel] masks frames."""
        beat = inputs.get('beat')
        if beat is None:
            return mel.new_zeros(*mel.shape[:2], BEAT_FEATURE_DIM)
        if beat.ndim != 3 or beat.shape[:2] != mel.shape[:2] or beat.shape[-1] != BEAT_FEATURE_DIM:
            raise ValueError('Beat feature tensor contract mismatch')
        beat = beat.to(dtype=mel.dtype)
        if inputs.get('beat_valid') is not None:
            beat = beat * inputs['beat_valid'].to(dtype=beat.dtype)[..., None]
        return beat

    def _onset_outputs(self, encoded_local, inputs, frames):
        """Per-frame onset logits [B,T,ONSET_CHANNELS] on the encoded local stream.

        Frames are the 10ms mel rows when mel_bins, else the local MERT frames
        (band slots averaged). Channel 0 = any onset, 1+k = onset of TYPES[k].
        """
        c = self.c
        if c.mel_bins:
            x = encoded_local[:, frames * c.band_slots:]
            times, valid = inputs['mel_times'], inputs['mel_valid']
        else:
            x = encoded_local[:, :frames * c.band_slots]
            x = x.view(x.shape[0], frames, c.band_slots, c.d).mean(2)
            times, valid = inputs['local_times'], inputs['local_valid']
        return dict(onset_logits=self.onset_projection(self.onset_norm(x)),
                    onset_times=times, onset_valid=valid.bool())

    def project_controls(self, controls):
        if controls.ndim != 3 or controls.shape[1:] != (self.c.controls, self.c.control_dim):
            raise ValueError('Control tensor contract mismatch')
        return self.control_projection(controls) + self.control_ids.weight[None] + self.source_ids.weight[3]

    def replace_controls(self, memory, controls):
        """Reuse condition-independent encoder activations for a matched contrast.

        Never modify an existing cache in place; callers decode the returned
        memory with fresh cross-attention keys and independent history state.
        """
        if len(controls) != len(memory['mode']):
            raise ValueError('Replacement controls must match the memory batch')
        projected = self.project_controls(controls)
        return {**memory, 'values': torch.cat((memory['values'][:, :-self.c.controls], projected), 1)}

    def memory_kv(self, memory):
        x, t = memory['values'], memory['times']
        positions = torch.arange(x.shape[1], device=x.device)[None].expand(x.shape[0], -1)
        return [layer.cross_attn.project_kv(x, t, positions, memory['timed']) for layer in self.decoder]

    def _decode_rope(self, times, positions):
        """Rotary tables shared by every decoder layer of one decode call.

        All layers rotate the same query times/positions; computing cos/sin
        once replaces ~5 rebuilds per layer with identical fp32 arithmetic.
        """
        hd = self.c.d // self.c.heads
        return dict(time_half=rotary_tables(times, lo=.02, hi=2400, half=hd // 4),
                    pos_half=rotary_tables(positions, lo=2, hi=65536, half=hd // 4),
                    time_full=rotary_tables(times, lo=.02, hi=2400, half=hd // 2))

    def decode(self, tokens, fields, times, valid, memory, *, cache=None):
        b, n = tokens.shape
        if cache is not None and (self.training or n != 1 or b != cache['batch_size']):
            raise ValueError('Incremental cache requires eval mode and one query per fixed batch row')
        if self.c.history_mixer != 'attention':
            if valid.shape != tokens.shape or bool((valid[:, 1:] & ~valid[:, :-1]).any()):
                raise ValueError('Mamba-2 requires right-padded independent requests')
            if cache is not None and not bool(valid.all()):
                raise ValueError('Cached queries must be valid')
        offset = cache['position'] if cache is not None else 0
        positions = torch.arange(offset, offset + n, device=tokens.device)[None].expand(b, -1)
        x = self.token_embedding(tokens) + self.field_embedding(fields) + self.mode_ids(memory['mode'])[:, None]
        memory_kv = cache['memory_kv'] if cache is not None else self.memory_kv(memory)
        rope = self._decode_rope(times, positions)
        for i, layer in enumerate(self.decoder):
            cache_i = cache['layers'][i] if cache is not None else None
            if self.c.grad_checkpointing and self.training:
                import torch.utils.checkpoint as cp
                x = cp.checkpoint(layer, x, times, positions, memory_kv[i],
                    memory['valid'], valid, cache_i, rope, use_reentrant=False)
            else:
                x = layer(x, times, positions, memory_kv[i], memory['valid'], valid, cache_i, rope)
        if cache is not None:
            cache['position'] += n
        return self.output(self.final_norm(x))

    def start_cache(self, memory):
        return dict(position=0, batch_size=len(memory['mode']), memory_kv=self.memory_kv(memory),
                    layers=[{} for _ in self.decoder])

    # ---- inference-only fixed-shape decoding (generation fast path) ----
    def start_static_cache(self, memory, capacity):
        """Preallocated decode state for ``decode_static`` (eval, attention mixer).

        Self-attention K/V live in fixed [B,kv,capacity,hd] buffers written in
        place (no torch.cat growth); cross-attention K/V are projected once.
        Buffers are allocated on the first step (dtype follows autocast) and
        zero-initialised so masked slots are finite.
        """
        if self.training or self.c.history_mixer != 'attention' or capacity < 1:
            raise ValueError('Static decoding requires eval mode and the attention history mixer')
        b = len(memory['mode'])
        return dict(batch_size=b, capacity=int(capacity), memory_kv=self.memory_kv(memory),
                    memory_mask=memory['valid'][:, None, None, :],
                    mode_embedding=self.mode_ids(memory['mode'])[:, None],
                    keys=None, values=None,
                    slots=torch.arange(int(capacity), device=memory['values'].device))

    def decode_static(self, tokens, fields, times, cache, position, length):
        """One cached decode step with static shapes; logits [B,1,V].

        ``position`` is a long tensor [1] holding the absolute query position
        (shared by every row, as in ``decode(cache=...)``) or [B] with one
        position per row (continuous batching: each row writes and attends at
        its own position); ``length`` (python int, ``> position``) is the
        attended K/V prefix, so a fixed ``length`` bucket keeps every shape
        static for CUDA-graph capture. Same math as ``decode(..., cache=...)``
        (masked slots get exactly zero weight); only fp32 reduction order may
        differ.
        """
        b = tokens.shape[0]
        if self.training or tokens.shape != (cache['batch_size'], 1) or not 0 < length <= cache['capacity']:
            raise ValueError('Static decode needs eval mode, one query per row and length <= capacity')
        positions = position.view(-1, 1).expand(b, 1)
        x = self.token_embedding(tokens) + self.field_embedding(fields) + cache['mode_embedding']
        rope = self._decode_rope(times, positions)
        # Batch-shaped like decode()'s [B,1,1,K] mask (Torch 2.6 MPS SDPA mis-caches
        # a [1,1,1,K] mask against a same-shaped [B,1,1,M] cross-attention call).
        mask = (cache['slots'][:length] <= position.view(-1, 1)).view(-1, 1, 1, length).expand(b, 1, 1, length)
        for i, layer in enumerate(self.decoder):
            z = layer.norm1(x)
            k, v = layer.self_attn.project_kv(z, times, positions, rope=rope)
            if cache['keys'] is None:
                shape = (len(self.decoder), b, k.shape[1], cache['capacity'])
                cache['keys'] = torch.zeros(*shape, k.shape[-1], dtype=k.dtype, device=k.device)
                cache['values'] = torch.zeros(*shape, v.shape[-1], dtype=v.dtype, device=v.device)
            keys, values = cache['keys'][i], cache['values'][i]
            if position.numel() > 1:
                keys.scatter_(2, position.view(b, 1, 1, 1).expand_as(k), k)
                values.scatter_(2, position.view(b, 1, 1, 1).expand_as(v), v)
            elif keys.device.type == 'mps':  # Torch 2.6 MPS lacks index_copy_.
                keys[:, :, position], values[:, :, position] = k, v
            else:
                keys.index_copy_(2, position, k)
                values.index_copy_(2, position, v)
            x = x + layer.self_attn.forward_grouped(z, times, positions,
                (keys[:, :, :length], values[:, :, :length]), mask, rope)
            x = x + layer.cross_attn.forward_grouped(layer.norm2(x), times, positions,
                cache['memory_kv'][i], cache['memory_mask'], rope)
            x = x + layer.ffn(layer.norm3(x))
        return self.output(self.final_norm(x))

    def forward(self, inputs, sequence, *, return_aux=False):
        """Logits [B,N,V]. ``return_aux=True`` returns ``(logits, aux)`` where aux
        holds onset_logits/onset_times/onset_valid when ``onset_head`` (else {})."""
        memory = self.encode(inputs)
        logits = self.decode(sequence['tokens'], sequence['fields'], sequence['times'], sequence['valid'], memory)
        if not return_aux:
            return logits
        return logits, {k: memory[k] for k in ('onset_logits', 'onset_times', 'onset_valid') if k in memory}


def event_loss(logits, sequence):
    allowed, valid, targets = sequence['allowed'], sequence['valid'], sequence['targets']
    # Masked padding has a dummy legal class provided by the collator. A real
    # illegal target is a caller error, never silently clamped or repaired.
    selected = logits.float().masked_fill(~allowed, -1e9)
    losses = F.cross_entropy(selected.flatten(0, 1), targets.flatten(), reduction='none').view_as(targets)
    # Equal request exposure must not silently become token-count weighting.
    count = valid.sum(-1)
    per_request = (losses * valid).sum(-1) / count.clamp_min(1)
    return per_request.sum() / (count > 0).sum().clamp_min(1)
