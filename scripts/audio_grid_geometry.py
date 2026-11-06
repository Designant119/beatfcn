"""Audio-only piecewise tempo coordinates for diagnostics, never target snapping."""
from fractions import Fraction
import numpy as np


def grid_points(segments_ms, duration_us, denominators, *, phase_shift_us=0):
    segments = np.asarray(segments_ms, dtype=np.float64)
    if (segments.ndim != 2 or segments.shape[1] != 2 or not len(segments)
            or not np.isfinite(segments).all() or np.any(segments[:, 1] <= 0)
            or np.any(np.diff(segments[:, 0]) <= 0) or duration_us <= 0):
        raise ValueError('invalid piecewise timing geometry')
    if not denominators or any(type(n) is not int or n < 1 or n > 64 for n in denominators):
        raise ValueError('invalid subdivision denominator')
    phases = np.array(sorted({Fraction(k, n) for n in denominators for k in range(n)}), dtype=float)
    points = []
    for i, (origin_ms, period_ms) in enumerate(segments):
        origin, period = origin_ms * 1000 + phase_shift_us, period_ms * 1000
        left = 0 if i == 0 else max(0, origin_ms * 1000)
        right = min(duration_us, segments[i + 1, 0] * 1000) if i + 1 < len(segments) else duration_us
        if right <= left:
            continue
        beats = np.arange(np.floor((left - origin) / period) - 1,
                          np.ceil((right - origin) / period) + 1)
        raw = origin + period * (beats[:, None] + phases[None, :])
        # Every new segment starts a new phase anchor. Extending a continuous
        # beat counter across this boundary would move its subdivision lines.
        valid = (raw >= left - 1e-6) & ((raw < right - 1e-6) if i + 1 < len(segments)
                                      else (raw <= right + 1e-6))
        points.extend(np.rint(raw[valid]).astype(np.int64).tolist())
    return np.unique(points).astype(np.int64)


def residuals(times_us, points_us):
    times = np.asarray(times_us, dtype=np.int64)
    points = np.asarray(points_us, dtype=np.int64)
    if not len(points) or np.any(np.diff(points) <= 0):
        raise ValueError('grid must have strictly increasing points')
    k = np.searchsorted(points, times)
    left = points[np.clip(k - 1, 0, len(points) - 1)]
    right = points[np.clip(k, 0, len(points) - 1)]
    nearest = np.where(abs(times - left) <= abs(times - right), left, right)
    return times - nearest, nearest


def error_summary(errors_us):
    e = np.asarray(errors_us, dtype=np.int64)
    if not len(e):
        return dict(n=0, within_1ms=None, within_10ms=None, within_20ms=None,
                    median_abs_ms=None, p95_abs_ms=None, p99_abs_ms=None, signed_median_ms=None)
    a = abs(e) / 1000
    return dict(n=len(e), within_1ms=float(np.mean(a <= 1)), within_10ms=float(np.mean(a <= 10)),
                within_20ms=float(np.mean(a <= 20)), median_abs_ms=float(np.median(a)),
                p95_abs_ms=float(np.quantile(a, .95)), p99_abs_ms=float(np.quantile(a, .99)),
                signed_median_ms=float(np.median(e) / 1000))


def drag_burst_indices(events, *, max_gap_us=80000, minimum_distinct_onsets=4):
    """A temporal proxy, not a judgment of artistic intent or chart quality."""
    onsets = sorted({e['onset_us'] for e in events if e['type'] == 'drag'})
    groups = []
    for t in onsets:
        if not groups or t - groups[-1][-1] > max_gap_us:
            groups.append([t])
        else:
            groups[-1].append(t)
    included = {t for group in groups if len(group) >= minimum_distinct_onsets for t in group}
    return np.array([i for i, e in enumerate(events) if e['type'] == 'drag' and e['onset_us'] in included], dtype=np.int64)
