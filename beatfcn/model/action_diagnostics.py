"""Target-informed action diagnostics, never a product decoder or benchmark arm.

Compare free generation with true onsets, then with true onsets and true outer
history. The inner note decisions remain predictions in both oracle arms.
"""
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import torch

from . import features as F, sample
from .batch import collate
from .p1 import NEG
from ..train.loop import to_device


@contextmanager
def oracle_onsets(model, groups):
    """Force only time/offset choices; preserve legality and all action decisions."""
    times = [int(g['t_rel_us']) for g in groups]
    if any(g.get('copy', -1) >= 0 for g in groups):
        raise ValueError('oracle diagnostic requires a blank canvas without COPY targets')
    if any(t % F.MS or not 0 <= t < F.CORE_US for t in times) or times != sorted(set(times)):
        raise ValueError('oracle onsets must be unique, increasing legal millisecond times')
    state = dict(requested_groups=len(times), visited_groups=0, ended=False, pending_offset=None,
                 diagnostic_only=True, truth_supplied=['onsets'])
    original_time, original_offset = model.time_logits, model.offset_logits

    def time_logits(*args, **kwargs):
        logits, mask, ok = original_time(*args, **kwargs)
        if tuple(logits.shape[:2]) != (1, 1) or state['pending_offset'] is not None:
            raise ValueError('oracle time observer received an unexpected generation call')
        index = state['visited_groups']
        forced = torch.full_like(logits, NEG)
        if index == len(times):
            choice = 0
            state['ended'] = True
        else:
            cell, remainder = divmod(times[index], F.CELL_US)
            offset = remainder // F.MS
            choice = cell + 1
            if not bool(ok[0, 0, cell, offset]):
                raise ValueError('oracle onset is masked out; do not bypass the legal decoder')
            state['pending_offset'] = offset
            state['visited_groups'] += 1
        if not bool(mask[0, 0, choice]):
            raise ValueError('oracle time choice is masked out')
        forced[0, 0, choice] = 0
        return forced, mask, ok

    def offset_logits(*args, **kwargs):
        logits = original_offset(*args, **kwargs)
        offset = state['pending_offset']
        if offset is None:  # Hold endpoints remain predictions.
            return logits
        state['pending_offset'] = None
        forced = torch.full_like(logits, NEG)
        forced[0, offset] = 0
        return forced

    with patch.object(model, 'time_logits', time_logits), patch.object(model, 'offset_logits', offset_logits):
        yield state


def empty_decisions():
    return dict(type_decisions=0, raw_probability_sum=[0.] * 4, legal_probability_sum=[0.] * 4,
        allowed_decisions_by_type=[0] * 4, proposed_by_type=[0] * 4, emitted_by_type=[0] * 4,
        hold_end_success=0, hold_end_failures={}, hold_to_tap=0, hold_stopped_group=0, empty_group_attempts=0)


@torch.no_grad()
def actions_with_true_outer_history(model, ex, song, *, device='cpu', seed=20261025):
    """Predict each group independently given only earlier true groups and its onset.

    Full teacher-forced encoding is causal. Within-group type, lane, count and
    duration remain free predictions. Oracle earlier holds can make the combined
    diagnostic outputs globally inconsistent; callers must report that separately.
    """
    bt = to_device(collate([ex], [song], model.c.taps, model.c.top_tap), torch.device(device))
    h, pa, _ = model.encode(bt)
    holds, previous_onsets, output = [], [], []
    runs = [(ex.c0_us, ex.c0_us + F.CORE_US)]
    diagnostics = empty_decisions()
    rng = torch.Generator(device='cpu').manual_seed(seed)
    for i, target in enumerate(ex.groups):
        t_rel = int(target['t_rel_us']); t_abs = ex.c0_us + t_rel
        busy = np.zeros(4, dtype=bool)
        for onset, end, lane in holds:
            if lane >= 0 and onset < t_abs < end:
                busy[lane] = True
        right = np.array(sample._right(t_abs - 1, [], runs), dtype=np.float32)
        notes = sample._inner(model, bt, h[:, i], pa, t_rel, right, busy, song, ex, t_abs,
            holds, runs, rng, .8, .95, True, device, [], sorted(previous_onsets),
            history_row=i, diagnostics=diagnostics)
        if notes:
            output.append(dict(t_rel_us=t_rel, bin=t_rel // F.BIN_US, residual=t_rel % F.BIN_US // F.MS,
                               copy=-1, notes=notes, busy_lanes=busy, right_new=right))
        else:
            diagnostics['empty_group_attempts'] += 1
        previous_onsets.append(t_abs)
        for note in target['notes']:
            if 'end_rel_us' in note:
                holds.append((t_abs, t_abs + note['end_rel_us'], note['lane']))
    return output, dict(diagnostic_only=True, truth_supplied=['onsets', 'earlier_outer_groups'],
                       requested_groups=len(ex.groups), visited_groups=len(ex.groups), note_decisions=diagnostics)
