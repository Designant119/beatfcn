"""Normalized local event packages on the lossless 40 ms typed canvas.

The audio/request encoder is shared with :class:`LocalTypedCanvas`.  A hurdle
count chooses the multiplicity of each cell/type group, then a GRU128 emits its
canonically ordered members.  This module contains objectives and decoding only;
it does not provide a training driver.  The current objective/decoder contract
explicitly requires microbatch=1.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .canvas_diffusion import legal_tail_nll
from .event_canvas import RhythmEvent
from .local_bundle import read_grid
from .local_typed_canvas import (
    CELL_US, COPIES, FRAMES, KINDS, SLOTS, TAIL_BLOCKS, LocalTypedCanvas, best_tail,
)

GRU_WIDTH = 128
ONSET_ADDRESSES = 40 + COPIES


def hurdle_count_probabilities(gate_logit, positive_logits):
    """Return p(C=0..16), with one gate and a normalized positive-count law."""
    if positive_logits.shape[-1] != SLOTS or gate_logit.shape != positive_logits.shape[:-1]:
        raise ValueError('Hurdle count expects one gate and sixteen positive counts')
    gate = gate_logit.sigmoid()
    positive = F.softmax(positive_logits, -1)
    return torch.cat((1 - gate[..., None], gate[..., None] * positive), -1)


def _hurdle_count_log_probabilities(gate_logit, positive_logits):
    positive = F.log_softmax(positive_logits, -1)
    return torch.cat((F.logsigmoid(-gate_logit)[..., None],
        F.logsigmoid(gate_logit)[..., None] + positive), -1)


class EventPackageCanvas(LocalTypedCanvas):
    """Shared typed-canvas encoder with hurdle counts and local GRU members."""

    def __init__(self, config, *, density_reference):
        super().__init__(config, density_reference=density_reference)
        # Do not leave the independent slot/quantity objective as dead weights.
        for name in ('quantity', 'existence', 'kind_slot', 'slot_read', 'offset',
                     'tail_block', 'tail_cell', 'tail_ms'):
            delattr(self, name)
        d = config.d
        self.count_gate = nn.Linear(d, len(KINDS))
        self.count_positive = nn.Linear(d, len(KINDS) * SLOTS)
        self.kind_ids = nn.Embedding(len(KINDS), d)
        self.count_ids = nn.Embedding(SLOTS + 1, d)
        self.onset_ids = nn.Embedding(ONSET_ADDRESSES, d)
        self.tail_block_ids = nn.Embedding(TAIL_BLOCKS, d)
        self.tail_cell_ids = nn.Embedding(50, d)
        self.tail_ms_ids = nn.Embedding(40, d)
        self.member_start = nn.Parameter(torch.randn(d) * .02)
        self.no_tail = nn.Parameter(torch.randn(d) * .02)
        self.member_read = nn.Linear(2 * d, d)
        self.member_gru = nn.GRUCell(2 * d, GRU_WIDTH)
        self.onset_read = nn.Linear(GRU_WIDTH, ONSET_ADDRESSES)
        self.tail_condition = nn.Sequential(nn.Linear(GRU_WIDTH + d, d), nn.SiLU())
        self.tail_block = nn.Linear(d, TAIL_BLOCKS)
        self.tail_cell = nn.Linear(d, 50)
        self.tail_ms = nn.Linear(d, 40)

    def forward(self, inputs):
        # Keep the density route and every audio/reference/control equation of
        # LocalTypedCanvas.forward up to its normalized cell representation x.
        controls = inputs['controls'].clone()
        density = controls[:, 0].clone()
        controls[:, 0] = 0
        memory = self.encode({**inputs, 'controls': controls})
        b = len(controls)
        n = inputs['local'].shape[2] * self.c.band_slots
        m = inputs['mel'].shape[1]
        query = torch.arange(FRAMES, device=controls.device).float()[None] * .04
        query = query.expand(b, -1)
        four = query[..., None] + torch.tensor([.005, .015, .025, .035], device=query.device)
        mel = read_grid(memory['values'][:, n:n + m], inputs['mel_times'],
            inputs['mel_valid'], four, .01)
        x = self.frame_read(mel.flatten(-2))
        clock = torch.stack((query / 8, (query / 8).square(),
            torch.sin(query * math.pi), torch.cos(query * math.pi)), -1)
        x = x + self.clock_read(clock) + self.mode_ids(inputs['mode'])[:, None]
        visible = inputs['reference_valid'] & (inputs['reference'][..., 10] > 0)
        style = ((inputs['reference'] * visible[..., None]).sum(1)
            / visible.sum(1, keepdim=True).clamp_min(1))
        scale, bias = self.direct(controls.flatten(1)).chunk(2, -1)
        x = x * (1 + .1 * scale[:, None].tanh()) + bias[:, None] + self.style(style)[:, None]
        for layer in self.address:
            x = layer(x, query + .02, memory)
        for layer in self.temporal:
            x = layer(x, inputs['canvas_valid'])
        x = self.norm(x)

        gate = self.count_gate(x)
        positive = self.count_positive(x).view(b, FRAMES, len(KINDS), SLOTS)
        rho = density[:, :2].mean(-1) * 64
        tilt = torch.where(density[:, 2] > .5,
            F.softplus(self.density_alpha)
            * (rho.clamp_min(1e-4) / self.density_reference).log(), 0.)
        gate = gate + tilt[:, None, None]
        positive = positive + tilt[:, None, None, None] * torch.arange(
            1, SLOTS + 1, device=x.device)
        return dict(x=x, gate=gate, positive=positive,
            count=hurdle_count_probabilities(gate, positive),
            count_log_probs=_hurdle_count_log_probabilities(gate, positive))

    def _teacher_group(self, x, frame, kind, count, offsets, deltas, allowed):
        """Teacher-force one canonical group, retaining complete prior members."""
        static = x[frame] + self.kind_ids.weight[kind] + self.count_ids.weight[count]
        previous = self.member_start
        hidden = x.new_zeros(GRU_WIDTH)
        members = []
        for slot in range(count):
            hidden = self.member_gru(
                torch.cat((static, previous), -1)[None], hidden[None])[0]
            onset_logits = self.onset_read(hidden).masked_fill(~allowed, -1e9)
            offset = int(offsets[slot])
            onset_embedding = self.onset_ids.weight[offset]
            tail = None
            if kind == KINDS.index('hold'):
                condition = self.tail_condition(torch.cat((hidden, onset_embedding), -1))
                tail = (self.tail_block(condition), self.tail_cell(condition),
                    self.tail_ms(condition))
                delta = int(deltas[slot])
                block, rest = divmod(delta, 2000)
                cell, ms = divmod(rest, 40)
                tail_embedding = (self.tail_block_ids.weight[block]
                    + self.tail_cell_ids.weight[cell] + self.tail_ms_ids.weight[ms])
            else:
                tail_embedding = self.no_tail
            previous = self.member_read(torch.cat((onset_embedding, tail_embedding), -1))
            members.append((onset_logits, tail))
        return members

    def _greedy_group(self, x, frame, kind, count, allowed, head_addresses,
            base_us, duration_us):
        """Emit a group with predicted count and complete predicted history."""
        if not bool(allowed.any()):
            raise ValueError('Predicted event has no legal onset address')
        static = x[frame] + self.kind_ids.weight[kind] + self.count_ids.weight[count]
        previous = self.member_start
        hidden = x.new_zeros(GRU_WIDTH)
        members = []
        for _ in range(count):
            hidden = self.member_gru(
                torch.cat((static, previous), -1)[None], hidden[None])[0]
            onset_logits = self.onset_read(hidden).masked_fill(~allowed, -1e9)
            offset = int(onset_logits.argmax())
            head = int(head_addresses[offset])
            onset_embedding = self.onset_ids.weight[offset]
            delta_ms = None
            if kind == KINDS.index('hold'):
                condition = self.tail_condition(torch.cat((hidden, onset_embedding), -1))
                fields = (self.tail_block(condition), self.tail_cell(condition),
                    self.tail_ms(condition))
                delta_us = best_tail(*(field.detach().float().cpu().numpy() for field in fields),
                    head - base_us, duration_us - base_us)
                delta_ms = delta_us // 1000
                block, rest = divmod(delta_ms, 2000)
                cell, ms = divmod(rest, 40)
                tail_embedding = (self.tail_block_ids.weight[block]
                    + self.tail_cell_ids.weight[cell] + self.tail_ms_ids.weight[ms])
            else:
                tail_embedding = self.no_tail
            previous = self.member_read(torch.cat((onset_embedding, tail_embedding), -1))
            members.append((head, delta_ms))
        return members


def _single_request(inputs, requests):
    if len(inputs['mode']) != 1 or len(requests) != 1:
        raise ValueError('EventPackageCanvas currently requires microbatch=1')
    return requests[0]


def _allowed_kinds(request):
    if request.allowed_types is None:
        return tuple(range(len(KINDS)))
    return tuple(KINDS.index(kind) for kind in request.allowed_types)


def package_loss(model, inputs, target, requests):
    """Count NLL plus teacher-forced member onset and legal joint-tail NLL.

    There is deliberately no independent existence, total-count, or interval
    auxiliary objective.  Only this training entry point receives target counts.
    """
    request = _single_request(inputs, requests)
    output = model(inputs)
    count = target['count'][0]
    valid = inputs['canvas_valid'][0]
    group_allowed = torch.zeros(len(KINDS), dtype=torch.bool, device=count.device)
    group_allowed[list(_allowed_kinds(request))] = True
    trained_groups = valid[:, None] & group_allowed[None]
    if bool(((count > 0) & ~trained_groups).any()):
        raise ValueError('Target has an event outside the valid cell/type support')
    if bool(((count < 0) | (count > SLOTS)).any()):
        raise ValueError('Target count outside hurdle support')
    count_nll = -output['count_log_probs'][0].gather(-1, count[..., None].long()).squeeze(-1)
    loss_count = ((count_nll * trained_groups).sum()
        / trained_groups.sum().clamp_min(1).float())

    zero = output['count'].sum() * 0
    onset_sum, tail_sum = zero, zero
    event_count, hold_count = 0, 0
    active = ((count > 0) & trained_groups).nonzero(as_tuple=False).tolist()
    for frame, kind in active:
        group_count = int(count[frame, kind])
        offsets = target['offset'][0, frame, kind, :group_count]
        allowed = inputs['canvas_head_allowed'][0, frame]
        if not bool(allowed[offsets.long()].all()):
            raise ValueError('Target onset outside the legal address support')
        deltas = []
        tail_domains = []
        if kind == KINDS.index('hold'):
            base_us = request.start_us + frame * CELL_US
            for slot in range(group_count):
                head = int(inputs['canvas_head_us'][0, frame, offsets[slot].long()])
                absolute = int(target['tail_block'][0, frame, slot] * 2000
                    + target['tail_cell'][0, frame, slot] * 40
                    + target['tail_ms'][0, frame, slot])
                delta = absolute - base_us // 1000
                if not 0 <= delta < TAIL_BLOCKS * 2000:
                    raise ValueError('Target tail outside the represented native-ms domain')
                lo = (head - base_us) // 1000 + 1
                hi = (request.duration_us - base_us) // 1000
                deltas.append(delta)
                tail_domains.append((lo, hi))
        members = model._teacher_group(output['x'][0], frame, kind, group_count,
            offsets, deltas, allowed)
        onset_sum = onset_sum + F.cross_entropy(
            torch.stack([member[0] for member in members]), offsets.long(), reduction='sum')
        event_count += group_count
        if kind == KINDS.index('hold'):
            for slot, (_, tail) in enumerate(members):
                lo, hi = tail_domains[slot]
                tail_sum = tail_sum + legal_tail_nll(*tail, lo, hi, deltas[slot])
            hold_count += group_count
    onset = onset_sum / max(event_count, 1)
    tail = tail_sum / max(hold_count, 1)
    return loss_count + onset + tail, dict(count=loss_count, onset=onset, tail=tail)


@torch.no_grad()
def generate_package(model, inputs, request):
    """Decode count modes and conditional-greedy members as ``RhythmEvent``s.

    No target, target count, or target-derived request attribute is accepted.
    Equal onset addresses remain separate members; no deduplication is applied.
    """
    _single_request(inputs, [request])
    model.eval()
    output = model(inputs)
    kind_allowed = torch.zeros(len(KINDS), dtype=torch.bool, device=output['count'].device)
    kind_allowed[list(_allowed_kinds(request))] = True
    probability = output['count'][0] * kind_allowed[None, :, None]
    valid = inputs['canvas_valid'][0]
    counts = probability.argmax(-1) * valid[:, None].long()
    total = int(counts.sum())
    if total > request.max_events:
        raise ValueError(f'Predicted event resource overflow: {total}')

    heads = inputs['canvas_head_us'][0]
    events = []
    for frame, kind in ((counts > 0).nonzero(as_tuple=False).tolist()):
        base_us = request.start_us + frame * CELL_US
        members = model._greedy_group(output['x'][0], frame, kind, int(counts[frame, kind]),
            inputs['canvas_head_allowed'][0, frame], heads[frame], base_us, request.duration_us)
        for head, delta_ms in members:
            end = head if delta_ms is None else base_us + delta_ms * 1000
            events.append(RhythmEvent(head, end, KINDS[kind]))
    return sorted(events)
