"""Does a control actually steer generation? Request bands around a core's true value, generate, and measure
the ACHIEVED value with the same statistic that defined the band during training (events.controls.raw_controls).

Per field we report, over (core, requested band) pairs: the rate the achieved band equals the requested band,
the Spearman correlation between requested band midpoints and achieved values (the response at all), the
per-core monotonic rate, the mean achieved value per requested band, and the legality of every generated
core. A control whose response is flat (correlation ~0) is not usable for charting assistance and should
be removed from the request contract rather than advertised.
"""
from __future__ import annotations
import numpy as np
from ..events import controls as C


def bands_around(field: str, true_value, span: int = 2):
    """Requested bands: the true band and the bands `span` edges below / above it (clamped), as (lo, hi)."""
    edges = C.BAND_EDGES[field]
    band = C.band_of(field, true_value)
    i = edges.index(band[0]) if band else len(edges) // 2
    picks = sorted({max(0, min(len(edges) - 2, i + d)) for d in (-span, 0, span)})
    return [(edges[j], edges[j + 1]) for j in picks]


def band_mid(band):
    lo, hi = band
    return (lo + hi) / 2


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    rx, ry = x.argsort().argsort(), y.argsort().argsort()
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize(records):
    """records: [{'field', 'core', 'requested': (lo, hi), 'achieved': float|None, 'legal': bool}] -> per-field summary."""
    out = {}
    for field in sorted({r['field'] for r in records}):
        rs = [r for r in records if r['field'] == field]
        valid = [r for r in rs if r['achieved'] is not None]
        in_band = sum(C.band_of(field, r['achieved']) == list(r['requested']) for r in valid)
        corr = spearman([band_mid(r['requested']) for r in valid], [r['achieved'] for r in valid])
        by_core = {}
        for r in valid:
            by_core.setdefault(r['core'], []).append((band_mid(r['requested']), r['achieved']))
        mono = [sorted(v)[0][1] < sorted(v)[-1][1] and all(a[1] <= b[1] for a, b in zip(sorted(v), sorted(v)[1:]))
                for v in by_core.values() if len(v) >= 3]
        # response judged WITHIN cores: the pooled requested-vs-achieved correlation is inflated by the spread of
        # true values across cores (a dense core gets a high requested band and a high achieved value regardless)
        within = [c for c in (spearman([a for a, _ in v], [b for _, b in v]) for v in by_core.values() if len(v) >= 3) if c is not None]
        per_band = {}
        for r in valid:
            per_band.setdefault(tuple(r['requested']), []).append(r['achieved'])
        out[field] = {
            'pairs': len(valid), 'cores': len(by_core),
            'in_band_rate': round(in_band / max(len(valid), 1), 3),
            'spearman_requested_vs_achieved': None if corr is None else round(corr, 3),
            'within_core_spearman': round(float(np.mean(within)), 3) if within else None,
            'monotonic_rate': round(sum(mono) / max(len(mono), 1), 3) if mono else None,
            'mean_achieved_by_requested_band': {f'{lo}-{hi}': round(float(np.mean(v)), 3) for (lo, hi), v in sorted(per_band.items())},
            'illegal_cores': sum(not r['legal'] for r in rs),
            'usable': bool(mono) and sum(mono) / len(mono) >= 0.6 and in_band / max(len(valid), 1) >= 0.3,
        }
    return out
