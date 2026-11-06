"""Experimental complete event-set generator; no count or fixed machine heads.

Matching is a permutation-invariant surrogate, not a marginal set likelihood.
Every query has the same native/COPY head support, including duplicate events.
The relative endpoint law is normalized on the predicted head's legal domain.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .local_bundle import read_grid
from .local_mamba_canvas import MambaCanvas


class SetBlock(nn.Module):
    def __init__(self, d, heads, ffn):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(3))
        self.self_attention = nn.MultiheadAttention(d, heads, dropout=0., batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d, heads, dropout=0., batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))

    def forward(self, x, memory):
        z = self.norms[0](x)
        x = x + self.self_attention(z, z, z, need_weights=False)[0]
        z = self.norms[1](x)
        x = x + self.cross_attention(z, memory, memory, need_weights=False)[0]
        return x + self.ffn(self.norms[2](x))


class JointNativeSet(MambaCanvas):
    def __init__(self, config, *, density_reference, queries=256, set_layers=2):
        super().__init__(config, density_reference=density_reference)
        for name in ('quantity', 'existence', 'kind_slot', 'slot_read', 'offset',
                     'tail_block', 'tail_cell', 'tail_ms', 'density_alpha'):
            delattr(self, name)
        d = config.d
        self.queries = queries
        self.event_queries = nn.Parameter(torch.randn(queries, d) * .02)
        self.set_blocks = nn.ModuleList(SetBlock(d, config.heads, config.ffn) for _ in range(set_layers))
        self.set_norm = nn.LayerNorm(d)
        self.kind = nn.Linear(d, 5)  # absent, tap, hold, flick, drag
        self.head_query = nn.Linear(d, 128)
        self.head_frame = nn.Linear(d, 128)
        self.head_mel = nn.Linear(config.mel_bins, 128)
        self.head_clock = nn.Linear(33, 128, bias=False)
        self.head_fine = nn.Linear(d, 56)
        self.tail_context = nn.Sequential(nn.Linear(d + config.mel_bins + 33, d), nn.SiLU(), nn.LayerNorm(d))
        self.tail_blocks = nn.Linear(d, 601)
        self.tail_cells = nn.Linear(d, 50)
        self.tail_milliseconds = nn.Linear(d, 40)

    def time_features(self, times):
        angles = times[..., None] * (2 * math.pi / self.time_periods)
        return torch.cat((times[..., None] / 1200, angles.sin(), angles.cos()), -1)

    def forward(self, inputs):
        # No label/event/history enters this interface. The only support is the
        # caller's request. Unlike old count tilt, all controls are visible here.
        memory = self.encode(inputs)
        frame_times = inputs['joint_frame_times']
        four = frame_times[..., None] + frame_times.new_tensor([.005, .015, .025, .035])
        mel = read_grid(memory['audio'], memory['audio_times'], memory['audio_valid'], four, .01)
        x = self.frame_read(mel.flatten(-2))
        clock = torch.stack((frame_times / 8, (frame_times / 8).square(),
                             torch.sin(frame_times * math.pi), torch.cos(frame_times * math.pi)), -1)
        x = x + self.clock_read(clock) + self.mode_ids(inputs['mode'])[:, None]
        visible = inputs['reference_valid'] & (inputs['reference'][..., 10] > 0)
        reference = inputs['reference'].masked_fill(~visible[..., None], 0)
        style = reference.sum(1) / visible.sum(1, keepdim=True).clamp_min(1)
        scale, bias = self.direct(inputs['controls'].flatten(1)).chunk(2, -1)
        x = x * (1 + .1 * scale[:, None].tanh()) + bias[:, None] + self.style(style)[:, None]
        context = memory['values']
        frames = x.shape[1]
        for layer in self.address:
            joint = layer(torch.cat((x, context), 1))
            x, context = joint[:, :frames], joint[:, frames:]
        for layer in self.temporal:
            x = layer(x)
        x = self.norm(x)

        # Full reference rows and ordered song states remain directly readable,
        # rather than assuming a long concatenated SSD scan used them.
        fine = int(inputs['local_valid'][0].sum()) * self.c.band_slots + int(inputs['mel_valid'][0].sum())
        song_count = int(inputs['song_valid'][0].sum())
        global_memory = torch.cat((x, memory['values'][:, fine:]), 1)
        q = self.event_queries[None]
        for layer in self.set_blocks:
            q = layer(q, global_memory)
        q = self.set_norm(q)

        times = inputs['joint_head_times']
        raw_mel = read_grid(inputs['mel'], inputs['mel_times'], inputs['mel_valid'], times, .01)
        frame = self.head_frame(x).repeat_interleave(56, 1)
        keys = frame + self.head_mel(raw_mel) + self.head_clock(self.time_features(times))
        head = torch.einsum('bqd,bad->bqa', self.head_query(q), keys) / math.sqrt(128)
        fine_bias = self.head_fine(q)
        head = head + fine_bias.repeat(1, 1, frames)
        head = head.masked_fill(~inputs['joint_head_valid'][:, None], -1e9)
        kind = self.kind(q).masked_fill(~inputs['joint_kind_allowed'][:, None], -1e9)
        return dict(kind=kind, head=head, query=q, song_rows=song_count)

    def tail(self, query, relative_head_s, inputs):
        """Attribute likelihood may condition on a head; free decode supplies its own.

        Absolute origin is excluded: d=end_ms-floor(head_us/1000). This is a
        head-conditioned additive joint law, not Sol's hierarchical tail proposal.
        """
        t = relative_head_s.reshape(1, -1)
        raw = read_grid(inputs['mel'], inputs['mel_times'], inputs['mel_valid'], t, .01)[0]
        clock = self.time_features(relative_head_s)
        x = self.tail_context(torch.cat((query, raw, clock), -1))
        return self.tail_blocks(x), self.tail_cells(x), self.tail_milliseconds(x)


def relative_tail_log_partition(block, cell, ms, high):
    """Vectorized exact log Z on integer deltas 1..high; no rounded clocks."""
    if bool((high < 1).any()) or bool((high >= block.shape[-1] * 2000).any()):
        raise ValueError('No legal relative endpoint support')
    pair = (cell[..., :, None] + ms[..., None, :]).flatten(-2)
    values = torch.arange(2000, device=block.device)
    last = high // 2000
    rest = high % 2000
    first_scores = pair.masked_fill((values < 1) | (values > high[..., None]), -1e9)
    first = block[..., 0] + first_scores.logsumexp(-1)
    final_scores = pair.masked_fill(values > rest[..., None], -1e9)
    final = block.gather(-1, last[..., None]).squeeze(-1) + final_scores.logsumexp(-1)
    final = final.masked_fill(last == 0, -1e9)
    bins = torch.arange(block.shape[-1], device=block.device)
    middle = block.masked_fill((bins < 1) | (bins >= last[..., None]), -1e9).logsumexp(-1)
    middle = middle + cell.logsumexp(-1) + ms.logsumexp(-1)
    return torch.stack((first, final, middle), -1).logsumexp(-1)


def relative_tail_nll(block, cell, ms, high, target):
    if bool(((target < 1) | (target > high)).any()):
        raise ValueError('Target relative endpoint outside legal support')
    score = (block.gather(-1, (target // 2000)[..., None]).squeeze(-1)
             + cell.gather(-1, ((target % 2000) // 40)[..., None]).squeeze(-1)
             + ms.gather(-1, (target % 40)[..., None]).squeeze(-1))
    return relative_tail_log_partition(block, cell, ms, high) - score
