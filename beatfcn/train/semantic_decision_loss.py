"""D97 opt-in semantic decision objective; production event_loss is unchanged.

Six fixed families: initial activity, later groups, onset time, type/pattern,
Phira membership, hold end. A time decision is its joint digit likelihood.
Within each family, average actual optional decision units, then divide by six.
Missing families contribute zero; never renormalize by families present.
This is supervised learning, not evidence of free-generation controllability.
"""
import numpy as np
import torch
from torch.nn import functional as F

from beatfcn.model.full_event_codec import FIELDS

FAMILIES = ('start', 'groups', 'onset', 'type', 'members', 'hold_end')


def semantic_decision_weights(sequence):
    """CPU preparation for one already grammar-validated teacher sequence."""
    rows, valid, allowed, targets = (np.asarray(sequence[k]) for k in ('fields', 'valid', 'allowed', 'targets'))
    if (rows.ndim != 1 or valid.shape != rows.shape or targets.shape != rows.shape
            or allowed.ndim != 2 or len(allowed) != len(rows)
            or valid.dtype != np.bool_ or allowed.dtype != np.bool_):
        raise ValueError('Expected one CPU teacher sequence with boolean masks')
    count = int(valid.sum())
    if not count or not valid[:count].all() or valid[count:].any():
        raise ValueError('Teacher validity must be a nonempty prefix')
    if (not np.issubdtype(rows.dtype, np.integer) or not np.issubdtype(targets.dtype, np.integer)
            or np.any((rows[:count] < 0) | (rows[:count] >= len(FIELDS)))
            or np.any((targets[:count] < 0) | (targets[:count] >= allowed.shape[1]))
            or not allowed[np.arange(count), targets[:count]].all()):
        raise ValueError('Invalid or illegal teacher target')
    names = [FIELDS[int(f)] for f in rows[:count]]
    if names[0] != 'group':
        raise ValueError('Initial semantic decision must be group activity')
    choice = allowed.sum(-1) > 1
    units = {family: [] for family in FAMILIES}
    pos = 0
    while pos < count:
        field = names[pos]
        if field in ('head_block', 'tail_block'):
            family = 'onset' if field == 'head_block' else 'hold_end'
            suffix = ['head_cell', 'head_ms'] if family == 'onset' else ['tail_cell', 'tail_ms']
            stop = pos + 1
            if stop < count and names[stop] == suffix[0]:
                if names[stop:stop+2] != suffix:
                    raise ValueError('Incomplete time decision unit')
                stop += 2
            elif family == 'hold_end':
                raise ValueError('Hold-end decision is missing its time digits')
            positions = [p for p in range(pos, stop) if choice[p]]
        else:
            if field == 'group':
                family = 'start' if pos == 0 else 'groups'
            elif field == 'kind':
                family = 'type'
            elif field == 'member':
                family = 'members'
            else:
                raise ValueError('Orphan time digit in teacher sequence')
            positions = [pos] if choice[pos] else []
            stop = pos + 1
        if positions:
            units[family].append(positions)
        pos = stop
    weights = np.zeros(len(rows), np.float32)
    for family_units in units.values():
        if family_units:
            coefficient = 1. / (len(FAMILIES) * len(family_units))
            for positions in family_units:
                weights[positions] = coefficient
    return weights, {family: len(value) for family, value in units.items()}


def semantic_decision_loss(logits, sequence):
    """Mean over real requests, using CPU-prepared weights; no host work here."""
    weights = sequence['semantic_weights']
    allowed, valid, targets = (sequence[k] for k in ('allowed', 'valid', 'targets'))
    if weights.shape != targets.shape or weights.device != logits.device or not weights.is_floating_point():
        raise ValueError('Semantic weights must accompany the padded target batch')
    selected = logits.float().masked_fill(~allowed, -1e9)
    losses = F.cross_entropy(selected.flatten(0, 1), targets.flatten(), reduction='none').view_as(targets)
    per_request = (losses * valid * weights).sum(-1)
    return per_request.sum() / (valid.sum(-1) > 0).sum().clamp_min(1)
