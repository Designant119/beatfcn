"""Request-scoped controls and an opt-in full-event contrastive objective.

Label construction is CPU preparation, never inference from a hidden test chart.
The objective uses the existing event decoder; no attribute-prediction shortcut.
"""
import math

import numpy as np
import torch
from torch.nn import functional as F

from beatfcn.events import controls as C
from beatfcn.events.canonical import MODE_TYPES
from beatfcn.model.full_event_model import event_loss

CONTROL_FIELDS = tuple(C.BAND_EDGES)
CONTROL_COLUMNS = ('lower', 'upper', 'specified', 'applicable', 'start', 'end')
TIME_SCALE_US = 1_200_000_000


def _scope(scope):
    if (len(scope) != 2 or any(type(v) is not int for v in scope)
            or not 0 <= scope[0] < scope[1] <= TIME_SCALE_US):
        raise ValueError('Use one nonempty request interval in integer microseconds')
    return tuple(scope)


def training_labels(mode, events, scope):
    """Caller supplies the complete post-transform human target plus visible events.

    Incoming holds must be present for occupancy. This function neither filters
    fakes nor invents missing events; content admission happens before this call.
    """
    scope = _scope(scope)
    if mode not in C.MODE_FIELDS:
        raise ValueError('Unsupported mode')
    events = tuple(events)
    for event in events:
        if event.type not in MODE_TYPES[mode] or event.onset_us < 0:
            raise ValueError('Illegal event for condition supervision')
        if (event.is_hold and event.end_us <= event.onset_us) or (
                not event.is_hold and event.end_us != event.onset_us):
            raise ValueError('Invalid target duration')
        if mode == 'mania_4k' and getattr(event.location, 'lane', None) not in range(4):
            raise ValueError('mania control supervision requires real lane identity')
    return C.derive_controls(mode, events, [scope])


def encode_controls(mode, bands, scope, *, origin_us):
    """Encode only caller-supplied conditions, including true zero and unknown.

    Values use frozen field-specific table maxima; times are relative to the
    requested core, not a hidden chart statistic. Whole-song scope is not inferred.
    """
    scope = _scope(scope)
    if mode not in C.MODE_FIELDS or type(origin_us) is not int or not 0 <= origin_us <= TIME_SCALE_US:
        raise ValueError('Invalid mode or request clock origin')
    if not set(bands) <= set(C.MODE_FIELDS[mode]):
        raise ValueError('Unknown or inapplicable condition')
    out = np.zeros((len(CONTROL_FIELDS), len(CONTROL_COLUMNS)), np.float32)
    for i, field in enumerate(CONTROL_FIELDS):
        if field not in C.MODE_FIELDS[mode]:
            continue
        out[i, 3] = 1
        out[i, 4:] = [(t - origin_us) / TIME_SCALE_US for t in scope]
        if field in bands:
            band = bands[field]
            if not C.is_table_band(field, band):
                raise ValueError(f'Invalid frozen band for {field}')
            out[i, :2] = np.asarray(band) / C.BAND_EDGES[field][-1]
            out[i, 2] = 1
    return out


def in_band(field, value, band):
    if value is None or not math.isfinite(value) or not C.is_table_band(field, band):
        return False
    lo, hi = band
    return lo <= value < hi or (lo <= value == hi == 1.0 and C.BAND_EDGES[field][-1] == 1.0)


def paired_control_views(mode, events_a, events_b, scope, field, *, origin_us):
    """Two symmetric single-field views; both opposing labels must conflict.

    Audio, split and visible-context identity must be checked by the dataset
    join. These labels alone cannot establish that two charts form a valid pair.
    """
    if field not in C.MODE_FIELDS.get(mode, ()):
        raise ValueError('Inapplicable paired field')
    labels = [training_labels(mode, events, scope) for events in (events_a, events_b)]
    if any(field not in bands for bands, _ in labels):
        raise ValueError('Undefined or out-of-table condition')
    bands = [item[0][field] for item in labels]
    if any(in_band(field, labels[i][1][field], bands[1-i]) for i in range(2)):
        raise ValueError('Opposing condition does not contradict the real target')
    correct = np.stack([encode_controls(mode, {field: band}, scope, origin_us=origin_us)
                        for band in bands])
    return dict(controls=correct, opposing_controls=correct[::-1].copy(), field=field,
                actual=[item[1][field] for item in labels], bands=bands,
                scope_us=list(scope))


def request_log_probability(logits, sequence):
    """Mean legal-target log probability; padding contributes neither sum nor count."""
    allowed, valid, targets = (sequence[k] for k in ('allowed', 'valid', 'targets'))
    masked = logits.float().masked_fill(~allowed, -1e9)
    nll = F.cross_entropy(masked.flatten(0, 1), targets.flatten(), reduction='none').view_as(targets)
    return -(nll * valid).sum(-1) / valid.sum(-1).clamp_min(1)


def validate_pair_metadata(inputs, pair_rows, opposing_controls):
    """CPU preparation gate; semantic contradictions come from paired_control_views."""
    if any(v.device.type != 'cpu' for v in [*inputs.values(), pair_rows, opposing_controls]):
        raise ValueError('Validate condition pairs before GPU allocation')
    if pair_rows.ndim != 1 or pair_rows.dtype != torch.long or not len(pair_rows) or len(pair_rows) % 2:
        raise ValueError('Expected explicit symmetric pair indices')
    ids = pair_rows.tolist()
    if len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= len(inputs['mode']):
        raise ValueError('Repeated or out-of-bounds pair member')
    correct = inputs['controls'].index_select(0, pair_rows)
    if opposing_controls.shape != correct.shape or not torch.isfinite(opposing_controls).all():
        raise ValueError('Invalid opposing control shape or values')
    if not torch.equal(correct.reshape(-1, 2, *correct.shape[1:]).flip(1).reshape_as(correct), opposing_controls):
        raise ValueError('Each opposing condition must be the other real member view')
    for a, b in zip(ids[::2], ids[1::2]):
        for key, value in inputs.items():
            if key not in ('controls', 'opposing_controls', 'condition_pair_rows') and not torch.equal(value[a], value[b]):
                raise ValueError(f'Pair has different non-condition input: {key}')
        ca, cb = inputs['controls'][a], inputs['controls'][b]
        if not torch.isfinite(ca).all() or not torch.isfinite(cb).all() or not torch.equal(ca[:, 2:], cb[:, 2:]):
            raise ValueError('Pair masks, applicability and scopes must match')
        changed = (ca[:, :2] != cb[:, :2]).any(-1)
        if not bool(changed.any()) or not bool(((ca[changed, 2] == 1) & (ca[changed, 3] == 1)).all()):
            raise ValueError('Pair must change a specified applicable value')


def condition_match_loss(correct, opposing, sequence, *, margin):
    """Rows are consecutive symmetric pairs, each containing two nonempty requests.

    CPU tape validation guarantees pair identity, legal nonempty targets, and
    actual contradiction. Avoid per-step device synchronisation for those checks.
    """
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('A finite nonnegative preregistered margin is required')
    if correct.shape != opposing.shape or not len(correct) or len(correct) % 2:
        raise ValueError('Expected equally shaped symmetric request pairs')
    gap = request_log_probability(correct, sequence) - request_log_probability(opposing, sequence)
    return F.softplus(margin - gap).reshape(-1, 2).mean(-1).mean(), gap


def conditional_event_objective(model, inputs, sequence, *, pair_rows=None,
                                opposing_controls=None, weight=0., margin=.1):
    """Training objective with one shared audio/reference encode and two decodes.

    pair_rows/opposing_controls are CPU-validated tape metadata transferred with
    the batch. No chart parsing, random negative sampling or CPU copies occur here.
    A zero weight is precisely the existing event CE, not an enabled experiment.
    """
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('Condition weight must be finite and nonnegative')
    memory = model.encode(inputs)
    keys = ('tokens', 'fields', 'times', 'valid')
    logits = model.decode(*(sequence[k] for k in keys), memory)
    ce = event_loss(logits, sequence)
    if weight == 0:
        return ce, dict(event_ce=ce.detach(), condition_match=None, condition_gap=None)
    if pair_rows is None:
        pair_rows = inputs.get('condition_pair_rows')
    if opposing_controls is None:
        opposing_controls = inputs.get('opposing_controls')
    if pair_rows is None or opposing_controls is None or pair_rows.ndim != 1:
        raise ValueError('Enabled condition objective requires explicit symmetric pairs')
    if pair_rows.dtype != torch.long or not len(pair_rows) or len(pair_rows) % 2:
        raise ValueError('Pair indices must be consecutive pairs of integer batch indices')
    paired_memory = {k: v.index_select(0, pair_rows) for k, v in memory.items()}
    paired_sequence = {k: v.index_select(0, pair_rows) for k, v in sequence.items()}
    altered = model.replace_controls(paired_memory, opposing_controls)
    opposing = model.decode(*(paired_sequence[k] for k in keys), altered)
    rank, gap = condition_match_loss(logits.index_select(0, pair_rows), opposing,
                                     paired_sequence, margin=margin)
    return ce + weight * rank, dict(event_ce=ce.detach(), condition_match=rank.detach(),
                                    condition_gap=gap.detach())
