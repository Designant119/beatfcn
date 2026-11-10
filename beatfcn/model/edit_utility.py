"""Reference reconstruction effort, not measured human editing time or chart quality.

Insert/delete operations have explicit costs. A different but valid chart can be
penalized by this proxy, and moving a note may cost less than delete-and-reinsert.
Report this alongside music alignment, legality, completion and human calibration.
"""
import math

from .evaluation import hold_matches
from .sample import match_f1


def edit_utility(pred, truth, mode, *, tolerance_us=20_000, delete_cost=1., review_cost=0.):
    if mode not in ('mania_4k', 'phira'):
        raise ValueError('unsupported mode')
    if (not math.isfinite(tolerance_us) or tolerance_us < 0 or
            any(not math.isfinite(x) or x < 0 for x in (delete_cost, review_cost))):
        raise ValueError('costs and tolerance must be finite and nonnegative')

    def counts(tp, p, t):
        insert, delete = t - tp, p - tp
        cost = insert + delete_cost * delete + review_cost * p
        return {'matched': tp, 'pred': p, 'true': t, 'insert': insert, 'delete': delete,
                'edit_cost': cost, 'blank_cost': t,
                'saved_fraction': 1 - cost / t if t else None,
                'precision': tp / p if p else None, 'recall': tp / t if t else None}

    def match(events, targets, key):
        ps, ts = {}, {}
        for group, items in ((ps, events), (ts, targets)):
            for e in items:
                group.setdefault(key(e), []).append(e.onset_us)
        return sum(match_f1(ps.get(k, []), ts.get(k, []), tolerance_us)[0] for k in ps.keys() | ts.keys())

    p_times, t_times = sorted({e.onset_us for e in pred}), sorted({e.onset_us for e in truth})
    # mania drafts carry no lane decision: lane leaves the match keys
    lane_blind = mode == 'mania_4k' and any(getattr(e.location, 'lane', -1) < 0 for e in pred)
    lane_key = lambda e: (e.type, e.location.lane if mode == 'mania_4k' and not lane_blind else None)
    nonholds_p, nonholds_t = [e for e in pred if not e.is_hold], [e for e in truth if not e.is_hold]
    full_tp = match(nonholds_p, nonholds_t, lane_key) + hold_matches(
        pred, truth, tolerance_us, tolerance_us, match_lane=mode == 'mania_4k' and not lane_blind)['tp']
    return {
        'schema': 'beatfcn.reference_edit_proxy.v1', 'tolerance_us': tolerance_us,
        'costs': {'insert': 1., 'delete': delete_cost, 'review_each_prediction': review_cost},
        'timing_point': counts(match_f1(p_times, t_times, tolerance_us)[0], len(p_times), len(t_times)),
        'rhythm_head': counts(match(pred, truth, lambda e: e.type), len(pred), len(truth)),
        'typed_head': counts(match(pred, truth, lane_key), len(pred), len(truth)),
        'full_note': counts(full_tp, len(pred), len(truth)),
        'limitation': 'Reference insert/delete proxy, not observed human time; full_note requires both hold endpoints. '
                      'Phira spatial output is outside this contract. Timing tolerances forgive small adjustments.'}
