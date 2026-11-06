"""HN2: public-clock audio encoding and two-pass native-tuple states.

Targets never enter this module. Sampling and exact endpoint support live in
``beatfcn.train.hn2_native``. Narrow configurations exercise the same modules.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HN2Config:
    d: int = 768
    heads: int = 12
    ffn: int = 3072
    slots: int = 512
    fine: int = 256
    local_layers: int = 4
    global_layers: int = 4
    relation_layers: int = 2
    decoder_layers: int = 6
    max_duration_us: int = 512_000_000

    def __post_init__(self):
        if self.d % self.heads or self.slots != 512 or min(self.d, self.ffn, self.fine) < 1:
            raise ValueError('HN2 requires divisible attention and exactly 512 public slots')


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * x.float().square().mean(-1, keepdim=True).add(1e-6).rsqrt().to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, memory, valid=None, bias=None):
        b, n, d = x.shape
        def split(value):
            return value.reshape(b, -1, self.heads, d // self.heads).transpose(1, 2)
        mask = None if valid is None else valid[:, None, None, :]
        if bias is not None:
            mask = bias if mask is None else bias.masked_fill(~mask, -torch.inf)
        y = F.scaled_dot_product_attention(split(self.q(x)), split(self.k(memory)),
                                           split(self.v(memory)), attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    def __init__(self, c, cross=False):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(c.d), RMSNorm(c.d)
        self.attention = Attention(c.d, c.heads)
        self.up = nn.Linear(c.d, c.ffn, bias=False)
        self.down = nn.Linear(c.ffn, c.d, bias=False)
        if cross:
            self.norm3 = RMSNorm(c.d)
            self.cross = Attention(c.d, c.heads)

    def forward(self, x, valid=None, bias=None, memory=None, memory_valid=None):
        z = self.norm1(x)
        x = x + self.attention(z, z, valid, bias)
        if memory is not None:
            x = x + self.cross(self.norm3(x), memory, memory_valid)
        x = x + self.down(F.gelu(self.up(self.norm2(x))))
        return x if valid is None else x * valid[..., None]


def read_features(values, times, valid, query, half=0., bounds=None):
    """Linear read on supplied physical centers; returns zero outside coverage."""
    b, t, d = values.shape
    shape = query.shape
    q = query.reshape(b, -1).contiguous()
    clock = times.masked_fill(~valid, 1e10).contiguous()
    last = (valid.sum(1, keepdim=True)-1).clamp_min(0)
    right = torch.minimum(torch.searchsorted(clock, q), last)
    left = (right - 1).clamp(min=0)
    tl, tr = times.gather(1, left), times.gather(1, right)
    vl = values.gather(1, left[..., None].expand(-1, -1, d))
    vr = values.gather(1, right[..., None].expand(-1, -1, d))
    alpha = ((q - tl) / (tr - tl).clamp_min(1e-8)).clamp(0, 1).to(values.dtype)
    result = vl + alpha[..., None] * (vr - vl)
    low = times.masked_fill(~valid, 1e10).amin(1, keepdim=True)
    high = times.masked_fill(~valid, -1e10).amax(1, keepdim=True)
    if bounds is not None:
        low, high = bounds[:, :1], bounds[:, 1:]
        half = 0.
    covered = (q >= low-half) & (q <= high+half) & valid.any(1, keepdim=True)
    return (result * covered[..., None]).reshape(*shape, d), covered.reshape(shape)


def read_fields(values, times, valid, query, half=0., bounds=None):
    """Read B,S,T scalar fields without materializing B,S,J,D endpoint keys."""
    b, s, t = values.shape
    q = query.reshape(b, -1).contiguous()
    clock = times.masked_fill(~valid, 1e10).contiguous()
    last = (valid.sum(1, keepdim=True)-1).clamp_min(0)
    right = torch.minimum(torch.searchsorted(clock, q), last).reshape_as(query)
    left = (right - 1).clamp(min=0)
    times_expanded = times[:, None, :].expand(b, s, t)
    tl, tr = times_expanded.gather(2, left), times_expanded.gather(2, right)
    alpha = ((query - tl) / (tr - tl).clamp_min(1e-8)).clamp(0, 1).to(values.dtype)
    result = values.gather(2, left) + alpha * (values.gather(2, right) - values.gather(2, left))
    low = times.masked_fill(~valid, 1e10).amin(1)[:, None, None]
    high = times.masked_fill(~valid, -1e10).amax(1)[:, None, None]
    if bounds is not None:
        low, high = bounds[:, :1, None], bounds[:, 1:, None]
        half = 0.
    coverage = (query >= low-half) & (query <= high+half) & valid.any(1)[:, None, None]
    return result * coverage, coverage


class HN2Native(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.c = config or HN2Config()
        c, d, fine = self.c, self.c.d, self.c.fine
        self.register_buffer('periods', torch.tensor([.02 * (25600 ** (i / 15)) for i in range(16)]))
        self.local_projection = nn.Linear(2048, d, bias=False)
        self.mel1 = nn.Conv1d(128, fine, 5, padding=2, bias=False)
        self.mel2 = nn.Conv1d(fine, fine, 5, padding=2, bias=False)
        self.mel_stride = nn.Conv1d(fine, d, 4, stride=4, bias=False)
        self.whole_projection = nn.Linear(512, fine, bias=False)
        self.whole_depthwise = nn.ModuleList(nn.Conv1d(fine, fine, 9, padding=4, groups=fine, bias=False) for _ in range(2))
        self.whole_pointwise = nn.ModuleList(nn.Linear(fine, fine, bias=False) for _ in range(2))
        self.song_projection = nn.Linear(1024, d, bias=False)
        self.phrase_projection = nn.Linear(d, d, bias=False)
        self.reference_projection = nn.Linear(16, d, bias=False)
        self.control_projection = nn.Linear(6, d, bias=False)
        self.control_ids = nn.Embedding(8, d)
        self.status = nn.Embedding(3, d)
        self.position = nn.Embedding(512, d)
        self.pass_ids = nn.Embedding(2, d)
        self.modality = nn.Embedding(5, d)
        self.kind_embedding = nn.Embedding(4, 32)
        self.event_projection = nn.Linear(106, d, bias=False)
        self.relation_bias = nn.Sequential(nn.Linear(8, 64, bias=False), nn.GELU(), nn.Linear(64, c.heads, bias=False))
        self.clock_projection = nn.Linear(35, d, bias=False)
        self.local_layers = nn.ModuleList(Block(c) for _ in range(c.local_layers))
        self.global_layers = nn.ModuleList(Block(c) for _ in range(c.global_layers))
        self.relation_layers = nn.ModuleList(Block(c) for _ in range(c.relation_layers))
        self.decoder = nn.ModuleList(Block(c, cross=True) for _ in range(c.decoder_layers))
        self.address_projection = nn.Linear(2*d + 2*fine + 37, d, bias=False)
        self.address_norm = RMSNorm(d)
        self.head_query = nn.Linear(d, d, bias=False)
        self.head_key = nn.Linear(d, d, bias=False)
        self.type_read = nn.Sequential(nn.Linear(2*d, d, bias=False), nn.GELU(), nn.Linear(d, 4, bias=False))
        self.tail_query = nn.Linear(2*d, d, bias=False)
        self.tail_key = nn.Linear(d + 2*fine + 37, d, bias=False)
        self.tail_level = nn.Embedding(3, d)
        self.tail_residual = nn.Linear(d, 1, bias=False)
        nn.init.zeros_(self.tail_residual.weight)
        self.active_read = nn.Linear(d, 1, bias=False)
        self.final_norms = nn.ModuleList(RMSNorm(d) for _ in range(4))

    def time_features(self, times):
        a = times.float()[..., None] * (2*math.pi / self.periods)
        return torch.cat((times.float()[..., None]/1200, a.sin(), a.cos()), -1)

    def clock(self, times, spans=None):
        spans = torch.stack((times, times), -1) if spans is None else spans
        return self.clock_projection(torch.cat((self.time_features(times), spans/1200), -1))

    def atom(self, us, origin):
        return torch.stack((((us-origin) % 40000).float()/40000,
                            (us % 1000).float()/1000, (us % 1000 != 0).float()), -1)

    def _group(self, values, times, valid, size):
        pad = (-values.shape[1]) % size
        values = F.pad(values, (0, 0, 0, pad))
        times, valid = F.pad(times, (0, pad)), F.pad(valid, (0, pad))
        b, t, d = values.shape
        vg = valid.reshape(b, t//size, size)
        count = vg.sum(-1).clamp_min(1)
        pooled = (values.reshape(b, t//size, size, d)*vg[..., None]).sum(2)/count[..., None]
        tg = times.reshape(b, t//size, size)
        centers = (tg*vg).sum(2)/count
        spans = torch.stack((tg.masked_fill(~vg, 1e10).amin(2), tg.masked_fill(~vg, -1e10).amax(2)), -1)
        return pooled, centers, vg.any(2), spans

    def encode(self, inputs):
        """Only public inputs, public head/type support, and physical clocks."""
        x = inputs
        local = x['local'].permute(0, 2, 1, 3, 4).flatten(-3)
        if local.shape[-1] != 2048:
            raise ValueError('HN2 requires MERT L12/L23, one band, D1024')
        local = self.local_projection(local)
        mel = x['mel'].masked_fill(~x['mel_valid'][..., None], 0)
        mel = F.gelu(self.mel1(mel.transpose(1, 2)))
        fine = F.gelu(self.mel2(mel)).transpose(1, 2) * x['mel_valid'][..., None]
        padded = F.pad(fine.transpose(1, 2), (0, (-fine.shape[1]) % 4))
        down = self.mel_stride(padded).transpose(1, 2)
        _, down_times, down_valid, _ = self._group(fine, x['mel_times'], x['mel_valid'], 4)
        aligned, _ = read_features(down, down_times, down_valid, x['local_times'], .02)
        local = (local + aligned + self.clock(x['local_times'], x.get('local_intervals')) + self.modality.weight[0]) * x['local_valid'][..., None]
        for layer in self.local_layers:
            local = layer(local, x['local_valid'])
        local = self.final_norms[0](local) * x['local_valid'][..., None]
        phrase, pt, pv, spans = self._group(local, x['local_times'], x['local_valid'], 8)
        phrase = self.phrase_projection(phrase) + self.clock(pt, spans) + self.modality.weight[1]
        song = self.song_projection(x['song']) + self.clock(x['song_times']) + self.modality.weight[2]
        global_values = torch.cat((phrase, song), 1)
        gt, gv = torch.cat((pt, x['song_times']), 1), torch.cat((pv, x['song_valid']), 1)
        for layer in self.global_layers:
            global_values = layer(global_values, gv)
        global_values = self.final_norms[1](global_values) * gv[..., None]
        # Direct song reads use the contextualized original 0.5s positions.
        song_values = global_values[:, phrase.shape[1]:]
        whole = x['whole_mel'].masked_fill(~x['whole_mel_valid'][..., None], 0)
        pad = (-whole.shape[1]) % 4
        whole = F.pad(whole, (0, 0, 0, pad)).reshape(whole.shape[0], -1, 512)
        _, wt, wv, _ = self._group(x['whole_mel'], x['whole_mel_times'], x['whole_mel_valid'], 4)
        whole = self.whole_projection(whole) * wv[..., None]
        for depth, point in zip(self.whole_depthwise, self.whole_pointwise):
            whole = (whole + point(F.gelu(depth(whole.transpose(1, 2)).transpose(1, 2)))) * wv[..., None]
        reference = self.reference_projection(x['reference']) + self.clock(x['reference_times']) + self.modality.weight[3]
        controls = self.control_projection(x['controls']) + self.control_ids.weight[None] + self.modality.weight[4]
        context = torch.cat((local, global_values, reference, controls), 1)
        context_valid = torch.cat((x['local_valid'], gv, x['reference_valid'],
                                   torch.ones(controls.shape[:2], dtype=torch.bool, device=controls.device)), 1)
        head_us, head_valid, type_allowed = x['head_us'], x['head_valid'], x['type_allowed']
        if head_us.shape[1] == 0:
            head_us = F.pad(head_us, (0, 1))
            head_valid = F.pad(head_valid, (0, 1))
            type_allowed = F.pad(type_allowed, (0, 0, 0, 1))
        hu = head_us.masked_fill(~head_valid, torch.iinfo(torch.int64).max)
        hu, order = hu.sort(1)
        hv = head_valid.gather(1, order)
        allowed = type_allowed.gather(1, order[..., None].expand(-1, -1, 4)) & hv[..., None]
        memory = dict(inputs=x, local=local, fine=fine, whole=whole, whole_times=wt, whole_valid=wv,
                      song=song_values, context=context, context_valid=context_valid,
                      head_us=hu, head_valid=hv, type_allowed=allowed)
        if 'whole_mel_intervals' in x:
            intervals = x['whole_mel_intervals']
            low = intervals[..., 0].masked_fill(~x['whole_mel_valid'], 1e10).amin(1)
            high = intervals[..., 1].masked_fill(~x['whole_mel_valid'], -1e10).amax(1)
            memory['whole_bounds'] = torch.stack((low, high), -1)
        if 'local_intervals' in x:
            intervals = x['local_intervals']
            low = intervals[..., 0].masked_fill(~x['local_valid'], 1e10).amin(1)
            high = intervals[..., 1].masked_fill(~x['local_valid'], -1e10).amax(1)
            memory['local_bounds'] = torch.stack((low, high), -1)
        safe = torch.where(hv, hu, x['request_start_us'][:, None])
        memory['head_features'] = self.address_features(memory, safe)
        memory['head_keys'] = self.head_key(memory['head_features'])
        return memory

    def address_features(self, memory, us):
        x = memory['inputs']
        times = (us-x['request_start_us'][:, None]).float()/1e6
        local, _ = read_features(memory['local'], x['local_times'], x['local_valid'], times, .02, memory.get('local_bounds'))
        song, _ = read_features(memory['song'], x['song_times'], x['song_valid'], times, .25)
        fine, fc = read_features(memory['fine'], x['mel_times'], x['mel_valid'], times, .005)
        whole, wc = read_features(memory['whole'], memory['whole_times'], memory['whole_valid'], times, .02, memory.get('whole_bounds'))
        exact = torch.stack(((us % 1000).float()/1000, (us % 1000 != 0).float()), -1)
        features = torch.cat((local, song, fine, whole, self.time_features(times), exact,
                              fc[..., None], wc[..., None]), -1)
        return self.address_norm(F.gelu(self.address_projection(features)))

    def forward(self, inputs, state=None, *, memory=None, pass_index=0):
        memory = self.encode(inputs) if memory is None else memory
        b = inputs['mode'].shape[0]
        x = self.position.weight[None].expand(b, -1, -1) + self.pass_ids.weight[pass_index]
        if state is None:
            x = x + self.status.weight[0]
            context, valid = memory['context'], memory['context_valid']
        else:
            events, active = state['events'], state['active']
            h, e, k = events[..., 0], events[..., 1], events[..., 2]
            origin = inputs['request_start_us'][:, None]
            fields = torch.cat((self.time_features((h-origin).float()/1e6),
                                self.time_features((e-origin).float()/1e6), self.atom(h, origin), self.atom(e, origin),
                                self.kind_embedding(k.clamp(0, 3)), torch.log1p((e-h).clamp_min(0).float()/1000)[..., None],
                                active[..., None]), -1)
            relation = x + self.status(torch.where(active, 2, 1)) + self.event_projection(fields)*active[..., None]
            dh = (h[:, :, None]-h[:, None, :]).float()/1e6
            de = (h[:, :, None]-e[:, None, :]).float()/1e6
            overlap = ((torch.minimum(e[:, :, None], e[:, None, :])-torch.maximum(h[:, :, None], h[:, None, :])).clamp_min(0).float()/1e6)
            a = active[:, :, None] & active[:, None, :]
            rf = torch.stack((dh.sign()*dh.abs().log1p(), de.sign()*de.abs().log1p(), overlap.log1p(),
                              (dh == 0).float(), (k[:, :, None] == k[:, None, :]).float(),
                              active[:, :, None].expand_as(a).float(), active[:, None, :].expand_as(a).float(), a.float()), -1)
            bias = self.relation_bias(rf*a[..., None]).permute(0, 3, 1, 2)
            for layer in self.relation_layers:
                relation = layer(relation, bias=bias)
            relation = self.final_norms[2](relation)
            context = torch.cat((memory['context'], relation), 1)
            valid = torch.cat((memory['context_valid'], torch.ones_like(active)), 1)
            x = x + self.status.weight[0]
        for layer in self.decoder:
            x = layer(x, memory=context, memory_valid=valid)
        x = self.final_norms[3](x)
        return dict(states=x, active_logits=self.active_read(x).squeeze(-1), memory=memory)

    def head_logits(self, output):
        memory = output['memory']
        logits = self.head_query(output['states']) @ memory['head_keys'].transpose(1, 2) / math.sqrt(self.c.d)
        valid = memory['head_valid'].clone()
        valid[:, 0] |= ~valid.any(1)
        return logits.masked_fill(~valid[:, None], -torch.inf)

    def type_logits(self, output, addresses):
        memory = output['memory']
        features = memory['head_features'].gather(1, addresses[..., None].expand(-1, -1, self.c.d))
        logits = self.type_read(torch.cat((output['states'], features), -1))
        allowed = memory['type_allowed'].gather(1, addresses[..., None].expand(-1, -1, 4)).clone()
        allowed[..., 0] |= ~allowed.any(-1)
        return logits.masked_fill(~allowed, -torch.inf)

    def tail_scores(self, output, heads, us, level):
        """Separable bilinear read avoids B,S,candidates,D endpoint activations."""
        memory, x = output['memory'], output['memory']['inputs']
        head_features = self.address_features(memory, heads)
        query = (self.tail_query(torch.cat((output['states'], head_features), -1)) + self.tail_level.weight[level]) * self.tail_residual.weight
        weights = query @ self.tail_key.weight
        times = (us-x['request_start_us'][:, None, None]).float()/1e6
        d, f = self.c.d, self.c.fine
        song_field = weights[..., :d] @ memory['song'].transpose(1, 2)
        whole_field = weights[..., d:d+f] @ memory['whole'].transpose(1, 2)
        fine_field = weights[..., d+f:d+2*f] @ memory['fine'].transpose(1, 2)
        song, _ = read_fields(song_field, x['song_times'], x['song_valid'], times, .25)
        whole, wc = read_fields(whole_field, memory['whole_times'], memory['whole_valid'], times, .02, memory.get('whole_bounds'))
        fine, fc = read_fields(fine_field, x['mel_times'], x['mel_valid'], times, .005)
        exact = torch.stack(((us % 1000).float()/1000, (us % 1000 != 0).float()), -1)
        clocks = torch.cat((self.time_features(times), exact, fc[..., None], wc[..., None]), -1)
        clock_score = (weights[..., d+2*f:].unsqueeze(-2)*clocks).sum(-1)
        return (song + whole + fine + clock_score) / math.sqrt(d)
