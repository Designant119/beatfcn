"""Hold-request and rhythm surrogate axes, without a compensating weighted sum.

These axes do not score hold-tail quality, audio causality, or whole-song quality.
They are not a complete admitted RL reward or a product promotion criterion.
"""
import math

from beatfcn.train.consensus_rl import event_f1


def hold_request_axes(events, reference, band):
    """Score complete raw event tuples; every native object counts separately."""
    low, high = band
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high <= 1):
        raise ValueError('Invalid hold-head request band')
    for group in (events, reference):
        for onset, end, kind, lane in group:
            if end < onset or (kind == 'hold' and end <= onset) or (kind != 'hold' and end != onset):
                raise ValueError('Malformed event duration')
    # An empty draft cannot satisfy a specified ratio, including a zero-hold request.
    ratio = sum(e[2] == 'hold' for e in events) / len(events) if events else None
    request = -1. if ratio is None else -max(low-ratio, ratio-high, 0.)
    # Retain multiplicity. Ignore marks only because this pilot changes hold heads.
    def onsets(group):
        return [(a, a, 'tap', -1) for a, _, _, _ in group]
    return dict(request_fit=request, onset_agreement=event_f1(onsets(events), onsets(reference)),
                hold_head_ratio=ratio, hold_duration_scored=False)


def pareto_preferences(rows, *, request_epsilon=0., onset_epsilon=.01, feasible=None):
    """Pairwise dominance scores; genuine tradeoffs remain incomparable.

    Exact request satisfaction has no preference between points inside the band.
    Musical agreement is only a frozen-reference surrogate, never an audio score.
    """
    if len(rows) < 2:
        raise ValueError('Need multiple candidates for the same request')
    if feasible is None:
        feasible = [True] * len(rows)
    if len(feasible) != len(rows) or any(type(v) is not bool for v in feasible):
        raise ValueError('One explicit feasibility flag per candidate is required')
    eps = (request_epsilon, onset_epsilon)
    if any(not math.isfinite(e) or e < 0 for e in eps):
        raise ValueError('Invalid indifference bounds')
    values = [(r['request_fit'], r['onset_agreement']) for r in rows]
    if any(not all(math.isfinite(x) for x in pair) for pair in values):
        raise ValueError('Nonfinite candidate axis')
    def dominates(a, b):
        return all(x >= y-e for x,y,e in zip(a,b,eps)) and any(x > y+e for x,y,e in zip(a,b,eps))
    def preference(i, j):
        if feasible[i] != feasible[j]:
            return 1 if feasible[i] else -1
        if not feasible[i]:
            return 0
        return int(dominates(values[i],values[j]))-int(dominates(values[j],values[i]))
    return [sum(preference(i,j) for j in range(len(values)) if i!=j)
            / (len(values)-1) for i,a in enumerate(values)]


def tail_prior_feasible(tail_nll, holds, threshold):
    """Use a frozen, per-request prior limit; never shorten or remove events."""
    if not math.isfinite(threshold) or threshold < 0 or holds < 0:
        raise ValueError('Invalid frozen tail limit')
    if holds == 0:
        return tail_nll is None
    return tail_nll is not None and math.isfinite(tail_nll) and 0 <= tail_nll <= threshold


def paired_tail_feasible(tail_nll, short_tail_nll, holds, minimum_margin):
    """A narrow anti-shortening test, not a general naturalness classifier."""
    if not math.isfinite(minimum_margin) or minimum_margin <= 0 or holds < 0:
        raise ValueError('Invalid paired tail guard')
    if holds == 0:
        return tail_nll is None and short_tail_nll is None
    scores=(tail_nll,short_tail_nll)
    if any(v is None or not math.isfinite(v) or v < 0 for v in scores):
        return False
    return short_tail_nll-tail_nll >= minimum_margin
