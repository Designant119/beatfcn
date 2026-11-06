"""Isolated non-autoregressive, physically addressed Phira experiment.

No target count, event history, onset or tail enters forward(). Fixed slots are
an explicit resource contract; overflowing inputs/outputs fail, never truncate.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .full_event_model import FullEventModel, EncoderBlock, Attention, FeedForward, RMSNorm
from .local_bundle import read_grid


KINDS = ('tap', 'hold', 'flick', 'drag')
CELL_US, FRAMES, SLOTS, COPIES = 40_000, 200, 16, 16
TAIL_BLOCKS = 601


class CanvasBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = RMSNorm(c.d), RMSNorm(c.d)
        self.attn, self.ffn = Attention(c), FeedForward(c)

    def forward(self, x, times, memory):
        p = torch.arange(x.shape[1], device=x.device)[None].expand(x.shape[0], -1)
        mt = torch.arange(memory['values'].shape[1], device=x.device)[None].expand(x.shape[0], -1)
        kv = self.attn.project_kv(memory['values'], memory['times'], mt, memory['timed'])
        x = x + self.attn(self.n1(x), times, p, kv, memory['valid'][:, None, None])
        return x + self.ffn(self.n2(x))


class TemporalBlock(nn.Module):
    def __init__(self, d, dilation):
        super().__init__()
        self.norm = RMSNorm(d)
        self.depth = nn.Conv1d(d, d, 5, padding=2 * dilation, dilation=dilation, groups=d)
        self.mix = nn.Sequential(nn.Linear(d, 2*d), nn.SiLU(), nn.Linear(2*d, d))

    def forward(self, x, valid):
        z = self.depth((self.norm(x) * valid[..., None]).transpose(1, 2)).transpose(1, 2)
        return (x + self.mix(F.silu(z))) * valid[..., None]


class LocalTypedCanvas(nn.Module):
    # Reuse exactly the input encoder equations, without constructing a token
    # embedding, vocabulary output matrix, or autoregressive decoder.
    encode = FullEventModel.encode
    project_controls = FullEventModel.project_controls

    def __init__(self, config, *, density_reference):
        super().__init__()
        self.c = c = config
        d = c.d
        self.layer_mix = nn.Parameter(torch.zeros(c.taps))
        self.local_projection = nn.Linear(c.tap_dim, d)
        self.band_embedding = nn.Parameter(torch.randn(c.bands, d) * .02)
        self.band_queries = nn.Parameter(torch.randn(c.band_slots, d) * .02)
        self.song_projection = nn.Linear(c.tap_dim, d)
        self.reference_projection = nn.Linear(c.reference_dim, d)
        self.control_projection = nn.Linear(c.control_dim, d)
        self.control_ids = nn.Embedding(c.controls, d)
        self.empty_reference = nn.Parameter(torch.zeros(1, 1, d))
        self.source_ids = nn.Embedding(4, d)
        self.mode_ids = nn.Embedding(2, d)
        self.mel_projection = nn.Linear(c.mel_bins, d)
        self.mel_source = nn.Parameter(torch.randn(d) * .02)
        self.encoders = nn.ModuleList(nn.ModuleList(EncoderBlock(c) for _ in range(c.encoder_layers))
                                      for _ in range(3))
        self.frame_read = nn.Linear(4*d, d)
        self.clock_read = nn.Linear(4, d)
        self.direct = nn.Sequential(nn.Linear(c.controls*c.control_dim, d), nn.SiLU(), nn.Linear(d, 2*d))
        self.style = nn.Linear(c.reference_dim, d)
        self.address = nn.ModuleList(CanvasBlock(c) for _ in range(2))
        self.temporal = nn.ModuleList(TemporalBlock(d, r) for r in (1, 2, 4, 8))
        self.norm = RMSNorm(d)
        self.quantity = nn.Linear(d, 4*(SLOTS+1))
        self.existence = nn.Linear(d, 4)
        self.kind_slot = nn.Parameter(torch.randn(4, SLOTS, d) * .02)
        self.slot_read = nn.Sequential(RMSNorm(d), nn.Linear(d, 2*d), nn.SiLU(), nn.Linear(2*d, d))
        self.offset = nn.Linear(d, 40+COPIES)
        self.tail_block = nn.Linear(d, TAIL_BLOCKS)
        self.tail_cell = nn.Linear(d, 50)
        self.tail_ms = nn.Linear(d, 40)
        self.density_alpha = nn.Parameter(torch.tensor(.54132485))  # softplus = 1
        self.register_buffer('density_reference', torch.tensor(float(density_reference)))

    def forward(self, inputs):
        controls = inputs['controls'].clone()
        density = controls[:, 0].clone()
        controls[:, 0] = 0  # density has exactly one route into count logits
        clean = {**inputs, 'controls': controls}
        memory = self.encode(clean)
        b = len(controls)
        n = inputs['local'].shape[2] * self.c.band_slots
        m = inputs['mel'].shape[1]
        query = torch.arange(FRAMES, device=controls.device).float()[None] * .04
        query = query.expand(b, -1)
        four = query[..., None] + torch.tensor([.005, .015, .025, .035], device=query.device)
        mel = read_grid(memory['values'][:, n:n+m], inputs['mel_times'], inputs['mel_valid'], four, .01)
        x = self.frame_read(mel.flatten(-2))
        clock = torch.stack((query / 8, (query / 8).square(), torch.sin(query*math.pi), torch.cos(query*math.pi)), -1)
        x = x + self.clock_read(clock) + self.mode_ids(inputs['mode'])[:, None]
        visible = inputs['reference_valid'] & (inputs['reference'][..., 10] > 0)
        style = (inputs['reference'] * visible[..., None]).sum(1) / visible.sum(1, keepdim=True).clamp_min(1)
        scale, bias = self.direct(controls.flatten(1)).chunk(2, -1)
        x = x * (1 + .1*scale[:, None].tanh()) + bias[:, None] + self.style(style)[:, None]
        for layer in self.address:
            x = layer(x, query + .02, memory)
        for layer in self.temporal:
            x = layer(x, inputs['canvas_valid'])
        x = self.norm(x)
        base_count = self.quantity(x).view(b, FRAMES, 4, SLOTS+1)
        rho = density[:, :2].mean(-1) * 64
        tilt = torch.where(density[:, 2] > .5,
            F.softplus(self.density_alpha) * (rho.clamp_min(1e-4) / self.density_reference).log(), 0.)
        count = base_count + tilt[:, None, None, None] * torch.arange(SLOTS+1, device=x.device)
        slot = x[:, :, None, None] + self.kind_slot[None, None]
        slot = slot + self.slot_read(slot)
        offset = self.offset(slot).masked_fill(~inputs['canvas_head_allowed'][:, :, None, None], -1e9)
        hold = slot[:, :, 1]
        return dict(count=count, base_count=base_count, existence=self.existence(x), offset=offset,
                    tail_block=self.tail_block(hold), tail_cell=self.tail_cell(hold), tail_ms=self.tail_ms(hold))


def canvas_loss(output, target, inputs):
    valid = inputs['canvas_valid']
    count = target['count']
    loss_count = F.cross_entropy(output['count'].flatten(0, 2), count.flatten(), reduction='none').view_as(count)
    loss_count = (loss_count * valid[..., None]).sum() / (valid.sum()*4).clamp_min(1)
    present = count > 0
    raw = F.binary_cross_entropy_with_logits(output['existence'], present.float(), reduction='none')
    positive, negative = present & valid[..., None], ~present & valid[..., None]
    existence = .5*((raw*positive).sum()/positive.sum().clamp_min(1) +
                    (raw*negative).sum()/negative.sum().clamp_min(1))
    mask = target['slot_valid'] & valid[..., None, None]
    offsets = F.cross_entropy(output['offset'].flatten(0, 3), target['offset'].flatten(), reduction='none').view_as(mask)
    per_type = (offsets*mask).sum((0, 1, 3)) / mask.sum((0, 1, 3)).clamp_min(1)
    offset = per_type.sum() / (mask.sum((0, 1, 3)) > 0).sum().clamp_min(1)
    holds = mask[:, :, 1]
    tail = 0.
    for name in ('tail_block', 'tail_cell', 'tail_ms'):
        ce = F.cross_entropy(output[name].flatten(0, 2), target[name].flatten(), reduction='none').view_as(holds)
        tail = tail + (ce*holds).sum()/holds.sum().clamp_min(1)/3
    values = torch.arange(SLOTS+1, device=count.device)
    expected = (output['count'].softmax(-1)*values).sum(-1)
    expected_total = (expected*valid[..., None]).sum((1, 2))
    actual = (count*valid[..., None]).sum((1, 2))
    total = F.smooth_l1_loss(expected_total.log1p(), actual.float().log1p())
    density = inputs['controls'][:, 0]
    width = inputs['canvas_width_s']
    rate = expected_total / width
    lo, hi = density[:, 0]*64, density[:, 1]*64
    distance = (lo.log1p()-rate.log1p()).relu() + (rate.log1p()-hi.log1p()).relu()
    interval = (distance*density[:, 2]).sum()/density[:, 2].sum().clamp_min(1)
    loss = loss_count + .5*existence + offset + tail + .2*total + .2*interval
    return loss, dict(count=loss_count, existence=existence, offset=offset, tail=tail, total=total, interval=interval)


def best_tail(block, cell, ms, onset_us, duration_us):
    """Exact joint argmax of three additive fields over all legal ms endpoints."""
    lo, hi = onset_us//1000 + 1, duration_us//1000
    if lo > hi:
        raise ValueError('No legal native-ms hold endpoint')
    block, cell, ms = map(np.asarray, (block, cell, ms))
    low_b, high_b = lo//2000, hi//2000
    pairs = np.add.outer(cell, ms).reshape(-1)
    candidates = []
    for b in {low_b, high_b}:
        first, last = max(0, lo-b*2000), min(1999, hi-b*2000)
        j = first + int(pairs[first:last+1].argmax())
        candidates.append((float(block[b]+pairs[j]), b*2000+j))
    if high_b-low_b > 1:
        b = low_b+1+int(block[low_b+1:high_b].argmax())
        j = int(pairs.argmax()); candidates.append((float(block[b]+pairs[j]), b*2000+j))
    return max(candidates, key=lambda x: (x[0], -x[1]))[1]*1000
