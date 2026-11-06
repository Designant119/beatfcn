"""Experimental scores of completed chart outcomes, not teacher-forced logits.

No training runner uses this module yet. Scores measure requested attributes,
not musical quality. Callers supply the full authorized final scope (including
retained notes and incoming holds), plus independent legality/completion checks.
Unknown fields never become zero targets; an undefined requested statistic fails.
"""
from collections import Counter
import math

from beatfcn.events import controls as C
from beatfcn.events.canonical import MODE_TYPES


def interval_violation(field, value, band):
    """0 in band; [1, 2) outside; 2 for an undefined statistic.

    The unit violation prevents a value exactly on an exclusive upper boundary
    receiving zero error. The remaining term is distance / (width + distance).
    This bounded scale is a declared candidate definition, not a tuned reward.
    """
    if not C.is_table_band(field, band):
        raise ValueError('Expected a frozen field-specific condition band')
    if value is None:
        return dict(value=None, defined=False, in_band=False, violation=2.)
    if not math.isfinite(value):
        raise ValueError('Non-finite outcome is a data error, not a missing label')
    lo, hi = band
    inside = lo <= value < hi or (lo <= value == hi == 1. and C.BAND_EDGES[field][-1] == 1.)
    distance = max(lo - value, value - hi, 0.)
    return dict(value=float(value), defined=True, in_band=inside,
                violation=0. if inside else 1. + distance / (hi - lo + distance))


def score_condition_outcome(mode, events, scope, bands, *, complete, legality_errors):
    """Score only caller-requested fields, with unchanged canonical statistics.

    This function neither derives requests from hidden targets nor repairs,
    filters or de-duplicates notes. Complete/legality must be independently
    established; no-condition requests return no scalar reward. A scalar alone
    cannot establish quality or controllability: retain every field and audit.
    """
    if mode not in C.MODE_FIELDS:
        raise ValueError('Unsupported mode')
    if (len(scope) != 2 or any(type(t) is not int for t in scope)
            or not 0 <= scope[0] < scope[1]):
        raise ValueError('Expected one nonempty scope in integer microseconds')
    if type(complete) is not bool:
        raise ValueError('Completion must be an explicit boolean')
    if not isinstance(legality_errors, (tuple, list)) or any(not isinstance(e, str) for e in legality_errors):
        raise ValueError('Independent legality findings must be a sequence of strings')
    if not set(bands) <= set(C.MODE_FIELDS[mode]):
        raise ValueError('Unknown or mode-inapplicable condition')
    events = tuple(events)
    for e in events:
        if (type(e.onset_us) is not int or type(e.end_us) is not int
                or e.onset_us < 0 or e.type not in MODE_TYPES[mode]
                or (e.is_hold and e.end_us <= e.onset_us)
                or (not e.is_hold and e.end_us != e.onset_us)):
            raise ValueError('Invalid canonical outcome event')
        if mode == 'mania_4k' and getattr(e.location, 'lane', None) not in range(4):
            raise ValueError('Mania outcome requires actual lane identity')
    raw = C.raw_controls(mode, events, [scope])
    fields = {field: dict(interval_violation(field, raw[field], band), requested_band=list(band))
              for field, band in bands.items()}
    eligible = complete and not legality_errors
    reward = None if not fields else (-sum(v['violation'] for v in fields.values()) / len(fields)
                                     if eligible else -2.)
    heads = [e for e in events if scope[0] <= e.onset_us < scope[1]]
    tap_times = sorted({e.onset_us for e in heads if e.type == 'tap'})
    gaps = [b-a for a, b in zip(tap_times, tap_times[1:])]
    return dict(scope_us=list(scope), mode=mode, complete=complete,
                legality_errors=list(legality_errors), eligible=eligible,
                fields=fields, realized=raw, requested_fields=len(fields), reward=reward,
                all_requested_satisfied=bool(fields) and eligible and all(v['in_band'] for v in fields.values()),
                audit=dict(heads=len(heads), type_counts=dict(Counter(e.type for e in heads)),
                           distinct_tap_gaps=len(gaps), positive_tap_gaps_lt10ms=sum(g < 10_000 for g in gaps),
                           positive_tap_gaps_lt10ms_fraction=sum(g < 10_000 for g in gaps)/len(gaps) if gaps else None),
                quality_passed=None, training_enabled=False)
