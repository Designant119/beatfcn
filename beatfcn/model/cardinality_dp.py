"""Full model cardinality, followed by conditional count MAP; CPU float64.

The cardinality polynomial is never truncated at a caller's resource limit.
This decision is not full-event MAP and does not receive a target/user count.
"""
import time

import numpy as np


def cardinality_counts(probabilities, *, max_events=None):
    started = time.monotonic()
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Counts require a finite nonnegative group x count law')
    if not np.allclose(p.sum(-1), 1, atol=2e-6, rtol=0):
        raise ValueError('Count laws must already be normalized')
    p = p / p.sum(-1, keepdims=True)
    polynomial = np.ones(1, np.float64)
    for row in p:
        polynomial = np.convolve(polynomial, row)
        polynomial /= polynomial.sum()
    total = int(polynomial.argmax())
    if max_events is not None and total > max_events:
        raise ValueError(f'Model cardinality resource overflow: {total}>{max_events}')
    logp = np.full_like(p, -np.inf)
    np.log(p, out=logp, where=p > 0)
    scores = np.full(total + 1, -np.inf)
    scores[0] = 0
    back = np.full((len(p), total + 1), -1, np.int16)
    for i, row in enumerate(logp):
        next_scores = np.full_like(scores, -np.inf)
        for count in range(min(len(row) - 1, total) + 1):
            candidate = scores[:total + 1 - count] + row[count]
            view = next_scores[count:]
            better = candidate > view
            view[better] = candidate[better]
            back[i, count:][better] = count
        scores = next_scores
    if not np.isfinite(scores[total]):
        raise ValueError('Cardinality mode has no feasible count allocation')
    counts = np.zeros(len(p), np.int64)
    remaining = total
    for i in range(len(p) - 1, -1, -1):
        counts[i] = back[i, remaining]
        if counts[i] < 0:
            raise ValueError('Missing conditional count backpointer')
        remaining -= int(counts[i])
    if remaining or int(counts.sum()) != total:
        raise AssertionError('Count backtrace changed model cardinality')
    return counts, dict(total=total, polynomial=polynomial,
        allocation_log_probability=float(scores[total]),
        support_max=len(polynomial) - 1, mass=float(polynomial.sum()),
        wall_s=time.monotonic() - started)
