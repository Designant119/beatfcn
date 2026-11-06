"""Whole cell/type-package absorbing diffusion, distinct from slot diffusion.

A variable is count plus its complete ordered member multiset. Zero is one
atom; MASK hides count and every member. Observed packages are immutable during
reverse sampling. Audio is encoded once, while neighboring packages condition
each denoising step. This isolated module has no training driver or defaults.
"""
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .canvas_diffusion import _sample_tail, legal_tail_nll
from .event_canvas import RhythmEvent
from .full_event_model import RMSNorm
from .local_event_package import (
    EventPackageCanvas, GRU_WIDTH, _allowed_kinds, _hurdle_count_log_probabilities,
)
from .local_typed_canvas import CELL_US, FRAMES, KINDS, SLOTS, TemporalBlock


MASK = SLOTS + 1


class GroupSampleError(RuntimeError):
    """Keep an explicitly incomplete reverse state for failure diagnosis."""
    def __init__(self, message, state, trace, step):
        super().__init__(message)
        self.state = {key: value.detach().cpu().clone() for key, value in state.items()}
        self.trace = list(trace)
        self.step = step


def valid_groups(inputs, request):
    if len(inputs['mode']) != 1:
        raise ValueError('Whole-package diffusion currently uses microbatch one')
    allowed = torch.zeros(4, dtype=torch.bool, device=inputs['mode'].device)
    allowed[list(_allowed_kinds(request))] = True
    groups = inputs['canvas_valid'][..., None] & allowed[None, None]
    legal_hold_head = (inputs['canvas_head_us'] // 1000 + 1 <= request.duration_us // 1000)
    hold_supported = (inputs['canvas_head_allowed'] & legal_hold_head).any(-1)
    all_supported = torch.ones_like(hold_supported)
    # PyTorch2.6/MPS strided boolean in-place writes can corrupt other kinds
    # in partial windows. Construct the full mask without view mutation.
    return groups & torch.stack((all_supported, hold_supported, all_supported, all_supported), -1)


def clean_packages(target, request):
    """Convert native absolute tails to cell-relative native-ms endpoints."""
    if request.start_us % 1000:
        raise ValueError('This local probe requires a native-ms window origin')
    count, offset = target['count'], target['offset']
    base_ms = request.start_us // 1000 + torch.arange(FRAMES, device=count.device) * 40
    absolute = target['tail_block'] * 2000 + target['tail_cell'] * 40 + target['tail_ms']
    delta = absolute - base_ms[None, :, None]
    slot = torch.arange(SLOTS, device=count.device)
    active = slot < count[..., None]
    hold_tail = torch.where(active[:, :, 1], delta, 0)
    zero_tail = torch.zeros_like(hold_tail)
    tail = torch.stack((zero_tail, hold_tail, zero_tail, zero_tail), 2)
    return dict(count=count.long().clone(), offset=torch.where(active, offset, 0).long(), tail=tail.long())


def corrupt_packages(clean, groups, steps, total_steps, *, generator=None, mask=None):
    """No clean member/count survives inside a masked package."""
    if mask is None:
        probability = steps.float()[:, None, None] / total_steps
        mask = (torch.rand(groups.shape, device=groups.device, generator=generator) < probability) & groups
    elif mask.shape != groups.shape or bool((mask & ~groups).any()):
        raise ValueError('Invalid whole-package corruption mask')
    known = groups & ~mask
    return dict(count=torch.where(mask, MASK, torch.where(known, clean['count'], 0)),
                offset=torch.where(known[..., None], clean['offset'], 0),
                tail=torch.where(known[..., None], clean['tail'], 0))


class GroupDiffusionCanvas(EventPackageCanvas):
    def __init__(self, config, *, density_reference, diffusion_steps=32):
        if diffusion_steps < 1:
            raise ValueError('Diffusion steps must be positive')
        super().__init__(config, density_reference=density_reference)
        self.diffusion_steps = diffusion_steps
        d = config.d
        self.mask_package = nn.Parameter(torch.randn(d) * .02)
        self.noise_time = nn.Embedding(diffusion_steps + 1, d)
        self.package_read = nn.Sequential(RMSNorm(d), nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))
        self.denoise_temporal = nn.ModuleList(TemporalBlock(d, dilation) for dilation in (1, 2, 4, 8))
        self.denoise_norm = RMSNorm(d)

    def encode_audio(self, inputs):
        # The inherited forward's old count probabilities are unused. Density
        # is already removed from its audio/control representation, exactly as
        # in the frozen event-package interface.
        return EventPackageCanvas.forward(self, inputs)['x']

    def package_features(self, noisy):
        count, offsets, tail = (noisy[key] for key in ('count', 'offset', 'tail'))
        if count.ndim != 3 or count.shape[1:] != (FRAMES, 4):
            raise ValueError('Count state must be batch x200 x4')
        if offsets.shape != (*count.shape, SLOTS) or tail.shape != offsets.shape:
            raise ValueError('Complete member state shape mismatch')
        if bool(((count < 0) | (count > MASK)).any()):
            raise ValueError('Count outside clean/MASK support')
        known = count != MASK
        active = known[..., None] & (torch.arange(SLOTS, device=count.device) < count[..., None])
        safe_count = torch.where(known, count, 0)
        safe_offsets = torch.where(active, offsets, 0)
        hold_kind = torch.arange(4, device=count.device) == 1
        holds = active & hold_kind[None, None, :, None]
        safe_tail = torch.where(holds, tail, 0)
        if bool(((safe_offsets < 0) | (safe_offsets >= self.onset_ids.num_embeddings)).any()):
            raise ValueError('Observed onset outside address support')
        if bool(((safe_tail < 0) | (safe_tail >= self.tail_block_ids.num_embeddings * 2000)).any()):
            raise ValueError('Observed tail outside represented endpoint support')
        block, rest = safe_tail // 2000, safe_tail % 2000
        tail_embedding = (self.tail_block_ids(block) + self.tail_cell_ids(rest // 40)
                          + self.tail_ms_ids(rest % 40))
        members = self.onset_ids(safe_offsets) + torch.where(holds[..., None], tail_embedding, self.no_tail)
        pooled = (members * active[..., None]).sum(-2) / safe_count.clamp_min(1)[..., None]
        content = self.count_ids(safe_count) + pooled
        content = torch.where(known[..., None], content, self.mask_package)
        content = content + self.kind_ids.weight[None, None]
        return self.package_read(content).mean(2)

    def forward(self, inputs, noisy, steps, *, audio=None):
        if steps.shape != (len(inputs['mode']),) or bool(((steps < 1) | (steps > self.diffusion_steps)).any()):
            raise ValueError('Noise time outside the frozen diffusion schedule')
        if audio is None:
            audio = self.encode_audio(inputs)
        x = audio + self.package_features(noisy) + self.noise_time(steps)[:, None]
        for layer in self.denoise_temporal:
            x = layer(x, inputs['canvas_valid'])
        x = self.denoise_norm(x)
        gate = self.count_gate(x)
        positive = self.count_positive(x).view(len(x), FRAMES, 4, SLOTS)
        log_probability = _hurdle_count_log_probabilities(gate, positive)
        density = inputs['controls'][:, 0]
        rho = density[:, :2].mean(-1) * 64
        tilt = torch.where(density[:, 2] > .5, F.softplus(self.density_alpha)
            * (rho.clamp_min(1e-4) / self.density_reference).log(), 0.)
        tilted = log_probability + tilt[:, None, None, None] * torch.arange(SLOTS + 1, device=x.device)
        log_probability = tilted - tilted.logsumexp(-1, keepdim=True)
        return dict(x=x, count=log_probability.exp(), count_log_probs=log_probability)

    def ordered_members(self, x, frame, kind, count, inputs, request, *, rng=None, teacher=None):
        """One canonical group, with the same legal support in train/sample."""
        if not 1 <= count <= SLOTS or (rng is None) == (teacher is None):
            raise ValueError('Specify exactly one teacher or sampling RNG')
        allowed = inputs['canvas_head_allowed'][0, frame]
        heads = inputs['canvas_head_us'][0, frame]
        base_ms = request.start_us // 1000 + frame * 40
        static = x[frame] + self.kind_ids.weight[kind] + self.count_ids.weight[count]
        hidden, previous = x.new_zeros(GRU_WIDTH), self.member_start
        previous_head, previous_tail = None, None
        onset_nll, tail_nll = x.sum() * 0, x.sum() * 0
        members = []
        for slot in range(count):
            hidden = self.member_gru(torch.cat((static, previous), -1)[None], hidden[None])[0]
            support = allowed if previous_head is None else allowed & (heads >= previous_head)
            if kind == KINDS.index('hold'):
                support = support & (heads // 1000 + 1 <= request.duration_us // 1000)
            if not bool(support.any()):
                raise ValueError('No legal nondecreasing member onset')
            logits = self.onset_read(hidden).masked_fill(~support, -1e9)
            if teacher is not None:
                offset = int(teacher['offset'][slot])
                if not bool(support[offset]):
                    raise ValueError('Teacher group is outside canonical onset support')
                onset_nll = onset_nll - logits.log_softmax(-1)[offset]
            else:
                probabilities = logits.float().cpu().double().softmax(-1).numpy()
                offset = int(rng.choice(len(probabilities), p=probabilities / probabilities.sum()))
            head = int(heads[offset])
            onset_embedding = self.onset_ids.weight[offset]
            delta = 0
            if kind == KINDS.index('hold'):
                condition = self.tail_condition(torch.cat((hidden, onset_embedding), -1))
                fields = self.tail_block(condition), self.tail_cell(condition), self.tail_ms(condition)
                lo = head // 1000 + 1 - base_ms
                hi = request.duration_us // 1000 - base_ms
                if previous_head == head:
                    lo = max(lo, previous_tail)
                if hi < lo:
                    raise ValueError('No legal ordered native-ms hold endpoint')
                if teacher is not None:
                    delta = int(teacher['tail'][slot])
                    tail_nll = tail_nll + legal_tail_nll(*fields, lo, hi, delta)
                else:
                    delta = _sample_tail(*(f.float().cpu().double().numpy() for f in fields), lo, hi, rng)
                block, rest = divmod(delta, 2000)
                cell, ms = divmod(rest, 40)
                tail_embedding = (self.tail_block_ids.weight[block] + self.tail_cell_ids.weight[cell]
                                  + self.tail_ms_ids.weight[ms])
                previous_tail = delta
            else:
                tail_embedding = self.no_tail
            previous = self.member_read(torch.cat((onset_embedding, tail_embedding), -1))
            previous_head = head
            members.append((offset, delta))
        return members, onset_nll, tail_nll


def group_diffusion_loss(model, inputs, target, request, steps, *, noisy=None, generator=None):
    groups = valid_groups(inputs, request)
    clean = clean_packages(target, request)
    if bool(((clean['count'] > 0) & ~groups).any()):
        raise ValueError('Target package outside the request support')
    if noisy is None:
        noisy = corrupt_packages(clean, groups, steps, model.diffusion_steps, generator=generator)
    output = model(inputs, noisy, steps)
    masked = (noisy['count'] == MASK) & groups
    weight = model.diffusion_steps / steps.float()[0]
    count_nll = -output['count_log_probs'].gather(-1, clean['count'][..., None]).squeeze(-1)
    count_sum = (count_nll * masked).sum()
    onset_sum, tail_sum = output['x'].sum() * 0, output['x'].sum() * 0
    for frame, kind in (masked[0] & (clean['count'][0] > 0)).nonzero(as_tuple=False).tolist():
        count = int(clean['count'][0, frame, kind])
        _, onset, tail = model.ordered_members(output['x'][0], frame, kind, count, inputs, request,
            teacher={key: clean[key][0, frame, kind, :count] for key in ('offset', 'tail')})
        onset_sum, tail_sum = onset_sum + onset, tail_sum + tail
    denominator = groups.sum().clamp_min(1).float()
    parts = {name: value * weight / denominator for name, value in
             [('count', count_sum), ('onset', onset_sum), ('tail', tail_sum)]}
    return sum(parts.values()), dict(**parts, masked_groups=masked.sum().float(), valid_groups=denominator)


@torch.no_grad()
def sample_group_diffusion(model, inputs, request, *, seed, initial_state=None, start_step=None, deadline=None):
    if request.start_us % 1000:
        raise ValueError('This local probe requires a native-ms window origin')
    model.eval()
    groups = valid_groups(inputs, request)
    state = dict(count=torch.where(groups, MASK, 0),
        offset=torch.zeros(*groups.shape, SLOTS, dtype=torch.long, device=groups.device),
        tail=torch.zeros(*groups.shape, SLOTS, dtype=torch.long, device=groups.device))
    if initial_state is not None:
        state = {key: value.clone() for key, value in initial_state.items()}
    if state['count'].shape != groups.shape or bool(((state['count'] != 0) & ~groups).any()):
        raise ValueError('Initial packages outside the valid request support')
    rng = np.random.default_rng(seed)
    audio = model.encode_audio(inputs)
    trace = []
    start = model.diffusion_steps if start_step is None else start_step
    if not 1 <= start <= model.diffusion_steps:
        raise ValueError('Invalid reverse starting time')
    for step in range(start, 0, -1):
        if deadline is not None and time.monotonic() > deadline:
            raise GroupSampleError('Whole-package reverse deadline', state, trace, step)
        masked = (state['count'] == MASK) & groups
        before = {key: value.clone() for key, value in state.items()}
        output = model(inputs, state, torch.tensor([step], device=groups.device), audio=audio)
        probability = output['count'][0].float().cpu().double().numpy()
        unlocking = torch.from_numpy(rng.random(groups.shape) < 1 / step).to(groups.device) & masked
        for frame, kind in unlocking[0].nonzero(as_tuple=False).tolist():
            p = probability[frame, kind]
            count = int(rng.choice(len(p), p=p / p.sum()))
            state['count'][0, frame, kind] = count
            known_count = state['count'].masked_fill(state['count'] == MASK, 0).sum()
            if int(known_count) > request.max_events:
                raise GroupSampleError('Whole-package reverse event resource overflow', state, trace, step)
            if count:
                try:
                    members, _, _ = model.ordered_members(output['x'][0], frame, kind, count,
                        inputs, request, rng=rng)
                except ValueError as exc:
                    raise GroupSampleError(f'Group {frame}/{kind}: {exc}', state, trace, step) from exc
                for slot, (offset, tail) in enumerate(members):
                    state['offset'][0, frame, kind, slot] = offset
                    state['tail'][0, frame, kind, slot] = tail
        observed = groups & ~masked
        if any(not torch.equal(before[key][observed], state[key][observed]) for key in state):
            raise GroupSampleError('Reverse process rewrote an observed package', state, trace, step)
        trace.append(dict(step=step, unlocked_groups=int(unlocking.sum()),
                          remaining_mask=int(((state['count'] == MASK) & groups).sum())))
    if bool(((state['count'] == MASK) & groups).any()):
        raise GroupSampleError('Reverse process left MASK packages', state, trace, 0)
    heads = inputs['canvas_head_us'][0]
    events = []
    for frame, kind in (groups[0] & (state['count'][0] > 0)).nonzero(as_tuple=False).tolist():
        base_ms = request.start_us // 1000 + frame * 40
        for slot in range(int(state['count'][0, frame, kind])):
            head = int(heads[frame, state['offset'][0, frame, kind, slot]])
            end = (base_ms + int(state['tail'][0, frame, kind, slot])) * 1000 if kind == 1 else head
            events.append(RhythmEvent(head, end, KINDS[kind]))
    return sorted(events), dict(trace=trace, state=state)
