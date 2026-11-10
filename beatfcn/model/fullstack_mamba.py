"""Experimental attention-free trainable chart stack with frozen audio inputs.

Every context encoder is bidirectional Mamba-2; context is prefixed to a causal
Mamba-2 event decoder. Pupu remains an external frozen frontend. No attention
encoder/cross-attention layer is retained in this candidate.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from beatfcn.model.full_event_codec import EventVocabulary, FIELDS
from beatfcn.model.full_event_model import RMSNorm, FeedForward, FullEventConfig
from beatfcn.model.mamba2_reference import Mamba2Reference


@dataclass(frozen=True)
class FullStackMambaConfig:
    d: int = 128
    ffn: int = 256
    encoder_layers: int = 1
    decoder_layers: int = 4
    taps: int = 3
    bands: int = 8
    band_slots: int = 2
    tap_dim: int = 1024
    reference_dim: int = 12
    controls: int = 8
    control_dim: int = 6
    mel_bins: int = 128
    d_state: int = 128
    headdim: int = 64
    chunk_size: int = 64
    dropout: float = 0.


class MambaBlock(nn.Module):
    def __init__(self, config, *, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.norm1, self.norm2 = RMSNorm(config.d), RMSNorm(config.d)
        self.mixer = Mamba2Reference(config.d, d_state=config.d_state,
                                    headdim=config.headdim, chunk_size=config.chunk_size)
        self.ffn = FeedForward(FullEventConfig(d=config.d, heads=4, kv_heads=2, ffn=config.ffn))

    def forward(self, x, *, return_cache=False):
        z = self.norm1(x)
        if return_cache:
            if self.bidirectional:
                raise ValueError('Bidirectional context is encoded once, never event-cached')
            y, cache = self.mixer(z, return_cache=True)
        else:
            y = self.mixer(z)
            if self.bidirectional:
                y = .5 * (y + self.mixer(z.flip(1)).flip(1))
        x = x + y
        x = x + self.ffn(self.norm2(x))
        return (x, cache) if return_cache else x

    def step(self, x, cache):
        y, fresh = self.mixer.step(self.norm1(x), cache)
        x = x + y
        return x + self.ffn(self.norm2(x)), fresh


class FullStackMamba(nn.Module):
    def __init__(self, config=None, vocab=None):
        super().__init__()
        self.c = config or FullStackMambaConfig()
        self.vocab = vocab or EventVocabulary()
        c = self.c
        if c.dropout != 0:
            raise ValueError('This frozen experiment uses dropout=0')
        self.layer_mix = nn.Parameter(torch.zeros(c.taps))
        self.local_projection = nn.Linear(c.tap_dim, c.d)
        self.band_embedding = nn.Parameter(torch.randn(c.bands, c.d) * .02)
        self.band_linear = nn.Linear(c.bands, c.band_slots, bias=False)
        with torch.no_grad():
            self.band_linear.weight.zero_()
            for i in range(c.band_slots):
                lo, hi = i * c.bands // c.band_slots, (i + 1) * c.bands // c.band_slots
                self.band_linear.weight[i, lo:hi] = 1 / (hi - lo)
        self.song_projection = nn.Linear(c.tap_dim, c.d)
        self.mel_projection = nn.Linear(c.mel_bins, c.d)
        self.mel_source = nn.Parameter(torch.randn(c.d) * .02)
        self.reference_projection = nn.Linear(c.reference_dim, c.d)
        self.control_projection = nn.Linear(c.control_dim, c.d)
        self.control_ids = nn.Embedding(c.controls, c.d)
        self.empty_reference = nn.Parameter(torch.zeros(1, 1, c.d))
        self.source_ids = nn.Embedding(4, c.d)
        self.mode_ids = nn.Embedding(2, c.d)
        self.time_projection = nn.Linear(33, c.d, bias=False)
        self.register_buffer('time_periods', torch.logspace(math.log10(.02), math.log10(2400), 16))
        self.encoders = nn.ModuleList(nn.ModuleList(MambaBlock(c, bidirectional=True)
                                    for _ in range(c.encoder_layers)) for _ in range(3))
        self.token_embedding = nn.Embedding(self.vocab.size, c.d, padding_idx=0)
        self.field_embedding = nn.Embedding(len(FIELDS), c.d)
        self.decoder = nn.ModuleList(MambaBlock(c) for _ in range(c.decoder_layers))
        self.final_norm = RMSNorm(c.d)
        self.output = nn.Linear(c.d, self.vocab.size, bias=False)

    def clock(self, t):
        angles = t[..., None] * (2 * math.pi / self.time_periods)
        return self.time_projection(torch.cat((t[..., None] / 1200, angles.sin(), angles.cos()), -1))

    def encode(self, inputs):
        if len(inputs['mode']) != 1:
            raise ValueError('This isolated reference backend uses microbatch one; it never flattens requests')
        c = self.c
        local = inputs['local']
        if local.shape[1] != c.taps or local.shape[-2:] != (c.bands, c.tap_dim):
            raise ValueError('Pupu input contract changed')
        keep = inputs['local_valid'][0]
        mixed = (F.layer_norm(local.float(), (c.tap_dim,))
                 * self.layer_mix.softmax(0)[None, :, None, None, None]).sum(1)[:, keep]
        bands = self.local_projection(mixed) + self.band_embedding
        fine = self.band_linear(bands.transpose(-2, -1)).transpose(-2, -1).flatten(1, 2)
        fine_times = inputs['local_times'][:, keep].repeat_interleave(c.band_slots, 1)
        mk = inputs['mel_valid'][0]
        fine = torch.cat((fine, self.mel_projection(inputs['mel'][:, mk]) + self.mel_source), 1)
        fine_times = torch.cat((fine_times, inputs['mel_times'][:, mk]), 1)
        order = fine_times[0].argsort(stable=True)
        fine, fine_times = fine[:, order], fine_times[:, order]
        sk, rk = inputs['song_valid'][0], inputs['reference_valid'][0]
        song = self.song_projection(F.layer_norm(inputs['song'][:, sk].float(), (c.tap_dim,)))
        reference = torch.cat((self.empty_reference, self.reference_projection(inputs['reference'][:, rk])), 1)
        reference_times = torch.cat((torch.zeros_like(inputs['reference_times'][:, :1]), inputs['reference_times'][:, rk]), 1)
        streams = ((fine, fine_times), (song, inputs['song_times'][:, sk]), (reference, reference_times))
        encoded = []
        for i, (x, times) in enumerate(streams):
            x = x + self.clock(times) + self.source_ids.weight[i] + self.mode_ids(inputs['mode'])[:, None]
            for layer in self.encoders[i]:
                x = layer(x)
            encoded.append(x)
        controls = (self.control_projection(inputs['controls']) + self.control_ids.weight[None]
                    + self.source_ids.weight[3] + self.mode_ids(inputs['mode'])[:, None])
        # Global context precedes fine music, then visible human evidence and requests.
        pieces = []
        for i in (1, 0, 2):
            pieces.extend((self.source_ids.weight[i][None, None], encoded[i]))
        pieces.extend((self.source_ids.weight[3][None, None], controls))
        return dict(values=torch.cat(pieces, 1), mode=inputs['mode'])

    def decode(self, tokens, fields, times, valid, memory, *, cache=None):
        if tokens.shape[0] != 1 or bool((valid[:, 1:] & ~valid[:, :-1]).any()):
            raise ValueError('Independent right-padded request required')
        x = (self.token_embedding(tokens) + self.field_embedding(fields)
             + self.clock(times) + self.mode_ids(memory['mode'])[:, None])
        x = x * valid[..., None]
        if cache is None:
            length = x.shape[1]
            x = torch.cat((memory['values'], x), 1)
            for layer in self.decoder:
                x = layer(x)
            x = x[:, -length:]
        else:
            if self.training or tokens.shape[1] != 1 or not bool(valid.all()):
                raise ValueError('Single valid eval token required')
            for i, layer in enumerate(self.decoder):
                x, cache['layers'][i] = layer.step(x, cache['layers'][i])
            cache['position'] += 1
        return self.output(self.final_norm(x))

    def start_cache(self, memory):
        if self.training:
            raise ValueError('Inference caches require eval mode')
        x, layers = memory['values'], []
        for layer in self.decoder:
            x, state = layer(x, return_cache=True)
            layers.append(state)
        return dict(layers=layers, position=0, batch_size=1)

    def forward(self, inputs, sequence):
        return self.decode(*(sequence[k] for k in ('tokens', 'fields', 'times', 'valid')), self.encode(inputs))
