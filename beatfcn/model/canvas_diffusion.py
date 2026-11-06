"""Lossless slot-package absorbing diffusion on the typed event canvas.

Each slot package contains an onset (ABSENT or a native/COPY address) and, for
holds, its complete relative endpoint. A package is absorbed into MASK as a
whole; unlocked packages carry over unchanged. Reverse sampling uses the exact
categorical absorbing posterior and a legally normalized joint tail law.
"""
from dataclasses import asdict
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import encode_events
from beatfcn.model.local_bundle import read_grid
from beatfcn.model.local_typed_canvas import COPIES, FRAMES, KINDS, SLOTS, LocalTypedCanvas, best_tail


ABSENT = 0
OFFSET_VOCAB = 1 + 40 + COPIES
MASK = OFFSET_VOCAB
TAIL_VOCAB = dict(tail_block=601, tail_cell=50, tail_ms=40)


def corrupt_packages(clean, t, steps, *, mask_id=MASK, generator=None):
    """Independent absorbing marginal q(x_t|x_0), with package mask supplied."""
    if not 0 <= t <= steps:
        raise ValueError('Diffusion time outside the frozen schedule')
    if t == 0:
        return clean.clone()
    if t == steps:
        return torch.full_like(clean, mask_id)
    mask = torch.rand(clean.shape, device=clean.device, generator=generator) < (t / steps)
    return torch.where(mask, torch.full_like(clean, mask_id), clean)


def categorical_reverse(logits_x0, x_t, t):
    """Exact one-variable absorbing posterior, including MASK as the last class."""
    if t < 1:
        raise ValueError('Reverse posterior starts at t=1')
    classes = logits_x0.shape[-1]
    clean = F.softmax(logits_x0.float(), -1)
    masked = x_t == classes
    if bool((x_t > classes).any()):
        raise ValueError('Unknown discrete state')
    carry = F.one_hot(x_t.clamp_max(classes - 1), classes).to(clean.dtype)
    result = torch.where(masked[..., None], clean / t, carry)
    mask_probability = torch.where(masked, torch.full_like(x_t, (t - 1) / t, dtype=clean.dtype), 0.)
    return torch.cat((result, mask_probability[..., None]), -1)


def _tail_scores(block, cell, ms, lo, hi, *, block_size, cell_size):
    values = torch.arange(lo, hi + 1, device=block.device)
    return block[values // block_size] + cell[(values % block_size) // cell_size] + ms[values % cell_size]


def _legal_partition(block, cell, ms, lo, hi, *, block_size, cell_size):
    if lo < 0 or hi < lo or hi >= len(block) * block_size:
        raise ValueError('Tail endpoint outside the represented native-ms domain')
    first, last = lo // block_size, hi // block_size
    if first == last:
        return torch.logsumexp(_tail_scores(block, cell, ms, lo, hi,
            block_size=block_size, cell_size=cell_size), 0)
    pieces = [torch.logsumexp(_tail_scores(block, cell, ms, lo, first * block_size + block_size - 1,
        block_size=block_size, cell_size=cell_size), 0),
        torch.logsumexp(_tail_scores(block, cell, ms, last * block_size, hi,
        block_size=block_size, cell_size=cell_size), 0)]
    if last - first > 1:
        full = block[first + 1:last] + torch.logsumexp(cell, 0) + torch.logsumexp(ms, 0)
        pieces.append(torch.logsumexp(full, 0))
    return torch.logsumexp(torch.stack(pieces), 0)


def legal_tail_nll(block, cell, ms, lo, hi, target, *, block_size=2000, cell_size=40):
    """Negative log likelihood under the exact legal additive endpoint law."""
    lo, hi, target = int(lo), int(hi), int(target)
    if not lo <= target <= hi:
        raise ValueError('Clean target is outside its legal endpoint support')
    score = block[target // block_size] + cell[(target % block_size) // cell_size] + ms[target % cell_size]
    return _legal_partition(block, cell, ms, lo, hi,
        block_size=block_size, cell_size=cell_size) - score


class AbsorbingCanvasDiffusion(LocalTypedCanvas):
    """Attention canvas denoiser with noisy package and diffusion-time inputs."""

    def __init__(self, config, *, density_reference, steps=32):
        super().__init__(config, density_reference=density_reference)
        if steps < 2:
            raise ValueError('Ancestral diffusion needs at least two steps')
        self.steps = steps
        d = config.d
        self.time_ids = nn.Embedding(steps + 1, d)
        self.noisy_offset = nn.Embedding(MASK + 1, d)
        self.noisy_tail_block = nn.Embedding(TAIL_VOCAB['tail_block'] + 1, d)
        self.noisy_tail_cell = nn.Embedding(TAIL_VOCAB['tail_cell'] + 1, d)
        self.noisy_tail_ms = nn.Embedding(TAIL_VOCAB['tail_ms'] + 1, d)
        self.noisy_frame = nn.Linear(4 * d, d)
        self.offset0 = nn.Linear(d, OFFSET_VOCAB)

    def corrupt(self, target, t, *, generator=None):
        clean = torch.where(target['slot_valid'], target['offset'] + 1,
            torch.full_like(target['offset'], ABSENT))
        offset = corrupt_packages(clean, t, self.steps, generator=generator)
        masked = offset == MASK
        noisy = {'offset': offset}
        for name, vocab in TAIL_VOCAB.items():
            noisy[name] = torch.where(masked[:, :, 1], torch.full_like(target[name], vocab), target[name])
        return noisy

    def forward(self, inputs, noisy, t):
        if t.ndim != 1 or len(t) != len(inputs['mode']) or bool(((t < 1) | (t > self.steps)).any()):
            raise ValueError('One in-schedule diffusion time per request is required')
        memory = self.encode(inputs)
        b = len(t)
        query = torch.arange(FRAMES, device=t.device).float()[None] * .04
        query = query.expand(b, -1)
        four = query[..., None] + torch.tensor([.005, .015, .025, .035], device=query.device)
        n = inputs['local'].shape[2] * self.c.band_slots
        m = inputs['mel'].shape[1]
        mel = read_grid(memory['values'][:, n:n + m], inputs['mel_times'], inputs['mel_valid'], four, .01)
        x = self.frame_read(mel.flatten(-2))
        clock = torch.stack((query / 8, (query / 8).square(), torch.sin(query * math.pi),
            torch.cos(query * math.pi)), -1)
        x = x + self.clock_read(clock) + self.mode_ids(inputs['mode'])[:, None]
        visible = inputs['reference_valid'] & (inputs['reference'][..., 10] > 0)
        style = (inputs['reference'] * visible[..., None]).sum(1) / visible.sum(1, keepdim=True).clamp_min(1)
        scale, bias = self.direct(inputs['controls'].flatten(1)).chunk(2, -1)
        x = x * (1 + .1 * scale[:, None].tanh()) + bias[:, None] + self.style(style)[:, None]
        onset_embedding = self.noisy_offset(noisy['offset'])
        tail_embeddings = [getattr(self, 'noisy_' + name)(noisy[name]) for name in TAIL_VOCAB]
        summary = torch.cat([onset_embedding.mean((2, 3))] + [value.mean(2) for value in tail_embeddings], -1)
        x = x + self.noisy_frame(summary) + self.time_ids(t)[:, None]
        for layer in self.address:
            x = layer(x, query + .02, memory)
        for layer in self.temporal:
            x = layer(x, inputs['canvas_valid'])
        x = self.norm(x)
        slot = x[:, :, None, None] + self.kind_slot[None, None] + onset_embedding
        hold = slot[:, :, 1] + tail_embeddings[0] + tail_embeddings[1] + tail_embeddings[2]
        slot = torch.cat((slot[:, :, :1], hold[:, :, None], slot[:, :, 2:]), 2)
        slot = slot + self.slot_read(slot)
        offset0 = self.offset0(slot)
        allowed = inputs['canvas_head_allowed'][:, :, None, None]
        allowed = allowed & inputs['canvas_valid'][:, :, None, None, None]
        support = torch.cat((torch.ones_like(allowed[..., :1]), allowed), -1)
        offset0 = offset0.masked_fill(~support, -1e9)
        hold = slot[:, :, 1]
        return dict(offset0=offset0, tail_block=self.tail_block(hold),
            tail_cell=self.tail_cell(hold), tail_ms=self.tail_ms(hold))


def diffusion_loss(output, target, noisy, inputs, requests, t, *, auxiliary_weight=.001,
        balance_positive=False):
    """Masked-package KL, or an explicitly reweighted positive/absent objective."""
    valid = inputs['canvas_valid']
    clean = torch.where(target['slot_valid'], target['offset'] + 1,
        torch.full_like(target['offset'], ABSENT))
    masked = (noisy['offset'] == MASK) & valid[..., None, None]
    onset = F.cross_entropy(output['offset0'].flatten(0, 3), clean.flatten(), reduction='none').view_as(clean)
    inverse_t = t.float().reciprocal()[:, None, None, None]
    raw_onset_vb = (onset * masked * inverse_t).sum()
    raw_onset_auxiliary = (onset * masked).sum()
    denominator = (valid.sum() * 4 * SLOTS).clamp_min(1).float()
    holds = target['slot_valid'][:, :, 1] & valid[..., None] & (noisy['offset'][:, :, 1] == MASK)
    denominator = denominator + holds.sum().float()
    positive = masked & target['slot_valid']
    absent = masked & ~target['slot_valid']
    positive_count = positive.sum().clamp_min(1).float()
    absent_count = absent.sum().clamp_min(1).float()
    positive_vb = (onset * positive * inverse_t).sum() / positive_count
    positive_auxiliary = (onset * positive).sum() / positive_count
    absent_vb = (onset * absent * inverse_t).sum() / absent_count
    absent_auxiliary = (onset * absent).sum() / absent_count
    tail_vb_sum = output['offset0'].sum() * 0
    tail_aux_sum = output['offset0'].sum() * 0
    for b, f, s in holds.nonzero(as_tuple=False).tolist():
        request = requests[b]
        offset = int(target['offset'][b, f, 1, s])
        head = int(inputs['canvas_head_us'][b, f, offset])
        base = request.start_us + f * 40_000
        lo = (head - base) // 1000 + 1
        hi = (request.duration_us - base) // 1000
        delta = int(target['tail_block'][b, f, s] * 2000 + target['tail_cell'][b, f, s] * 40
            + target['tail_ms'][b, f, s])
        nll = legal_tail_nll(output['tail_block'][b, f, s], output['tail_cell'][b, f, s],
            output['tail_ms'][b, f, s], lo, hi, delta)
        tail_vb_sum = tail_vb_sum + nll / t[b].float()
        tail_aux_sum = tail_aux_sum + nll
    raw_vb = (raw_onset_vb + tail_vb_sum) / denominator
    raw_auxiliary = (raw_onset_auxiliary + tail_aux_sum) / denominator
    if balance_positive:
        has_positive = positive.any()
        vb = torch.where(has_positive, .5 * (positive_vb + absent_vb), absent_vb)
        auxiliary = torch.where(has_positive, .5 * (positive_auxiliary + absent_auxiliary), absent_auxiliary)
        vb = vb + tail_vb_sum / holds.sum().clamp_min(1).float()
        auxiliary = auxiliary + tail_aux_sum / holds.sum().clamp_min(1).float()
    else:
        vb, auxiliary = raw_vb, raw_auxiliary
    return vb + auxiliary_weight * auxiliary, dict(vb=vb, auxiliary=auxiliary,
        raw_vb=raw_vb, raw_auxiliary=raw_auxiliary,
        onset_vb=raw_onset_vb / denominator, onset_auxiliary=raw_onset_auxiliary / denominator,
        positive_vb=positive_vb, positive_auxiliary=positive_auxiliary,
        absent_vb=absent_vb, absent_auxiliary=absent_auxiliary,
        tail_vb=tail_vb_sum / denominator, tail_auxiliary=tail_aux_sum / denominator,
        balanced=torch.tensor(float(balance_positive), device=vb.device),
        masked_packages=masked.sum().float(), masked_positive_packages=positive.sum().float(),
        masked_holds=holds.sum().float())


def _numpy_logsumexp(values):
    peak = np.max(values)
    return float(peak + np.log(np.exp(values - peak).sum()))


def _sample_tail(block, cell, ms, lo, hi, rng):
    """Sample the exact legal additive law without a large Cartesian tensor."""
    first, last = lo // 2000, hi // 2000
    block_scores = []
    for index in range(first, last + 1):
        start = max(lo, index * 2000)
        end = min(hi, index * 2000 + 1999)
        values = np.arange(start, end + 1)
        scores = block[index] + cell[(values % 2000) // 40] + ms[values % 40]
        block_scores.append(_numpy_logsumexp(scores))
    block_scores = np.asarray(block_scores)
    weights = np.exp(block_scores - block_scores.max())
    chosen = first + int(rng.choice(len(weights), p=weights / weights.sum()))
    start, end = max(lo, chosen * 2000), min(hi, chosen * 2000 + 1999)
    values = np.arange(start, end + 1)
    scores = block[chosen] + cell[(values % 2000) // 40] + ms[values % 40]
    weights = np.exp(scores - scores.max())
    return int(values[rng.choice(len(values), p=weights / weights.sum())])


@torch.no_grad()
def reconstruction_probe(model, inputs, noisy, t, request):
    """One-shot x0/argmax diagnostic; never label this as ancestral generation."""
    started = time.monotonic()
    model.eval()
    output = {key: value.float().cpu().numpy() for key, value in model(inputs, noisy, t).items()}
    clean = output['offset0'][0].argmax(-1)
    valid = inputs['canvas_valid'][0].cpu().numpy()
    heads = inputs['canvas_head_us'][0].cpu().numpy()
    events = []
    failure = None
    tokens = []
    try:
        for f, k, s in np.argwhere(clean > ABSENT):
            if not valid[f]:
                continue
            if request.allowed_types is not None and KINDS[k] not in request.allowed_types:
                raise ValueError('Predicted disallowed type')
            head = int(heads[f, clean[f, k, s] - 1])
            if k == 1:
                base = request.start_us + f * 40_000
                delta = best_tail(output['tail_block'][0, f, s], output['tail_cell'][0, f, s],
                    output['tail_ms'][0, f, s], head - base, request.duration_us - base)
                end = int(base + delta)
            else:
                end = head
            events.append(RhythmEvent(head, end, KINDS[k]))
        events.sort()
        if len(events) > request.max_events:
            raise ValueError('Predicted event resource overflow')
        tokens = encode_events(events, request)
        if len(tokens) > request.max_tokens:
            raise ValueError('Native serialization token capacity overflow')
    except (ValueError, TimeoutError) as exc:
        failure = f'{type(exc).__name__}: {exc}'
    return dict(complete=failure is None, failure=failure, events=[asdict(event) for event in events],
        tokens=tokens, decode_kind='reconstruction_probe', predicted_count=len(events),
        wall_s=time.monotonic() - started)


@torch.no_grad()
def ancestral_sample(model, inputs, request, *, seed=0, deadline=None, initial_noisy=None, start_step=None):
    """Carry-over ancestral generation; optional authorized noised-target diagnostics."""
    started = time.monotonic()
    model.eval()
    device = inputs['mode'].device
    start = model.steps if start_step is None else start_step
    if not 1 <= start <= model.steps or (initial_noisy is None and start != model.steps):
        raise ValueError('A partial reverse chain needs an explicit noisy state and valid start step')
    shape = (1, FRAMES, 4, SLOTS)
    noisy = dict(offset=torch.full(shape, MASK, dtype=torch.long, device=device),
        tail_block=torch.full((1, FRAMES, SLOTS), 601, dtype=torch.long, device=device),
        tail_cell=torch.full((1, FRAMES, SLOTS), 50, dtype=torch.long, device=device),
        tail_ms=torch.full((1, FRAMES, SLOTS), 40, dtype=torch.long, device=device))
    if initial_noisy is not None:
        if set(initial_noisy) != set(noisy):
            raise ValueError('Noisy package fields are incomplete')
        noisy = {key: value.clone() for key, value in initial_noisy.items()}
    generator = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    trace = []
    for step in range(start, 0, -1):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError('Diffusion ancestral deadline')
        output = model(inputs, noisy, torch.tensor([step], device=device))
        probabilities = categorical_reverse(output['offset0'], noisy['offset'], step).float().cpu()
        sampled = torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]), 1,
            generator=generator).reshape(shape).to(device)
        was_masked = noisy['offset'] == MASK
        unlocked = was_masked & (sampled != MASK)
        next_noisy = {'offset': sampled}
        for name, vocab in TAIL_VOCAB.items():
            hold_state = sampled[:, :, 1]
            carry = torch.where(hold_state == MASK, torch.full_like(noisy[name], vocab), noisy[name])
            next_noisy[name] = torch.where(unlocked[:, :, 1], torch.zeros_like(carry), carry)
        arrays = {name: value.float().cpu().numpy() for name, value in output.items() if name.startswith('tail_')}
        heads = inputs['canvas_head_us'][0].cpu().numpy()
        for f, s in (unlocked[:, :, 1] & (sampled[:, :, 1] > ABSENT)).nonzero(as_tuple=False)[:, 1:].tolist():
            offset = int(sampled[0, f, 1, s]) - 1
            head = int(heads[f, offset])
            base = request.start_us + f * 40_000
            lo = (head - base) // 1000 + 1
            hi = (request.duration_us - base) // 1000
            delta = _sample_tail(arrays['tail_block'][0, f, s], arrays['tail_cell'][0, f, s],
                arrays['tail_ms'][0, f, s], lo, hi, rng)
            next_noisy['tail_block'][0, f, s], rest = delta // 2000, delta % 2000
            next_noisy['tail_cell'][0, f, s], next_noisy['tail_ms'][0, f, s] = rest // 40, rest % 40
        trace.append(dict(t=step, unlocked=int(unlocked.sum()), remaining_mask=int((sampled == MASK).sum())))
        noisy = next_noisy
    if bool((noisy['offset'] == MASK).any()):
        raise AssertionError('The t=1 absorbing posterior must clear every MASK')
    events = []
    failure = None
    tokens = []
    try:
        valid = inputs['canvas_valid'][0].cpu().numpy()
        heads = inputs['canvas_head_us'][0].cpu().numpy()
        for f, k, s in (noisy['offset'][0] > ABSENT).nonzero(as_tuple=False).tolist():
            if not valid[f]:
                continue
            if request.allowed_types is not None and KINDS[k] not in request.allowed_types:
                raise ValueError('Predicted disallowed type')
            offset = int(noisy['offset'][0, f, k, s]) - 1
            head = int(heads[f, offset])
            if k == 1:
                base = request.start_us + f * 40_000
                end = base + int(noisy['tail_block'][0, f, s] * 2000 + noisy['tail_cell'][0, f, s] * 40
                    + noisy['tail_ms'][0, f, s]) * 1000
            else:
                end = head
            events.append(RhythmEvent(head, end, KINDS[k]))
        events.sort()
        if len(events) > request.max_events:
            raise ValueError('Predicted event resource overflow')
        tokens = encode_events(events, request)
        if len(tokens) > request.max_tokens:
            raise ValueError('Native serialization token capacity overflow')
    except (ValueError, TimeoutError) as exc:
        failure = f'{type(exc).__name__}: {exc}'
    return dict(complete=failure is None, failure=failure, events=[asdict(event) for event in events],
        tokens=tokens, decode_kind='ancestral', steps=start, unlock_trace=trace,
        predicted_count=len(events), wall_s=time.monotonic() - started)
