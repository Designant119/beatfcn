"""Complete-window density supervision and incompatible-condition negatives.

Only the requested density changes in the negative view. Audio, observed notes,
permissions and all targets stay identical. The negative is ranked, never used
as a positive transcription label. No human notes or drag sequences are thinned.
"""
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import torch
from torch.nn import functional as NN

from ..events import controls as C
from ..events.canonical import MODE_TYPES, build_model_view
from ..events.requests import assert_no_leak, resolve
from ..model import features as F

FIELD = 'density_notes_per_second'
FIELD_INDEX = F.CONTROL_FIELDS.index(FIELD)


def complete_window(sample, chart, c0, duration, seed):
    """One fully supervised editable core; retain boundary holds whole as observed."""
    end = min(c0 + F.CORE_US, duration)
    if not 0 <= c0 < end <= duration or end - c0 < F.SEC:
        raise ValueError('density supervision needs a complete window of at least one second')
    regions = ([(0, c0, 'complete')] if c0 else []) + [(c0, end, 'editable')]
    if end < duration:
        regions.append((end, duration, 'complete'))
    observed, hidden, provenance = resolve(chart.events, regions)
    bands, raw = C.derive_controls(chart.mode, observed + hidden, [(c0, end)])
    band = bands.get(FIELD)
    if band is None:
        raise ValueError('density outside supported condition bands')
    edges = C.BAND_EDGES[FIELD]
    correct_index = edges.index(band[0])
    minimum = sum(c0 <= e.onset_us < end for e in observed) * F.SEC / (end - c0)
    candidates = [i for i in range(len(edges) - 1)
                  if abs(i - correct_index) >= 2 and minimum < edges[i + 1] <= 16.]
    if not candidates:
        raise ValueError('no feasible incompatible density request')
    rng = np.random.default_rng([seed, 0xD3517])
    # Balance increasing/decreasing requests when both directions are feasible.
    sides = [side for side in ([i for i in candidates if i < correct_index],
                               [i for i in candidates if i > correct_index]) if side]
    choices = sides[int(rng.integers(len(sides)))]
    wrong = int(rng.choice(choices))
    request = {**sample.request,
               'observed_events': [e.to_json(locked=True) for e in observed],
               'regions': [{'region_id': f'density-{i}', 'start_us': a, 'end_us': b, 'state': state}
                           for i, (a, b, state) in enumerate(regions)],
               'references': [],
               'controls': {'allowed_types': list(MODE_TYPES[chart.mode]), FIELD: band}}
    view = build_model_view(chart.mode, observed, hidden)
    if not view.ok:
        raise ValueError('density window has invalid quantized targets')
    assert_no_leak(request, hidden)
    return replace(sample, request=request, hidden=tuple(hidden), evidence=(), removed=(),
                   model_view=view, raw_controls=raw,
                   recipe={'kind': 'density_window', 'permission_provenance': provenance,
                           'density_scope_us': [c0, end], 'density_wrong_band': wrong,
                           'density_true_band': correct_index, 'density_raw': raw[FIELD]})


def decision_nll(logits, target, valid, batch_index, batch_size, mask=None):
    """Per-example mean NLL, so one long chart cannot dominate a condition pair."""
    if mask is not None:
        logits = logits.masked_fill(~mask, -1e9)
    ce = NN.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                          target.reshape(-1).clamp(0, logits.shape[-1] - 1), reduction='none')
    weight = valid.reshape(-1).float()
    index = batch_index.reshape(-1)
    counts = ce.new_zeros(batch_size).scatter_add(0, index, weight)
    sums = ce.new_zeros(batch_size).scatter_add(0, index, ce * weight)
    return sums / counts.clamp(min=1)


def ranking_loss(positive, negative, margin=.1):
    """Compare time and within-onset count decisions, not offsets or placement.

    This is a sum of two mean NLLs, not the joint chart log likelihood. A lower
    contrastive loss alone is insufficient: generated density must also respond.
    """
    pos = positive['time'] + positive['cont']
    neg = negative['time'] + negative['cont']
    return NN.softplus(margin + pos - neg).mean(), (neg > pos).sum(), pos.numel()


def rng_state(device):
    return (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


@contextmanager
def replay_rng(state, device):
    """Give both condition views identical dropout/SSL masks, without advancing RNG twice."""
    after = rng_state(device)
    try:
        torch.set_rng_state(state[0])
        if state[1] is not None:
            torch.cuda.set_rng_state(state[1], device)
        yield
    finally:
        torch.set_rng_state(after[0])
        if after[1] is not None:
            torch.cuda.set_rng_state(after[1], device)
