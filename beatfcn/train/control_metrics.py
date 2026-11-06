"""Control-response metrics for chart generation.

Loss is a training diagnostic.  These metrics answer the product questions:
does the output follow the requested intensity envelope, does it preserve note
types, and does it collapse to an empty readout?
"""
from collections import Counter
import math

import numpy as np

NOTE_TYPES = ('tap', 'hold', 'drag', 'flick')


def _events(value):
    if isinstance(value, dict):
        value = value.get('events', ())
    return list(value)


def _event_type(event):
    return event['type'] if isinstance(event, dict) else event.type


def _event_onset(event):
    return int(event['onset_us'] if isinstance(event, dict) else event.onset_us)


def curve_counts(events, start_us, cells, cell_us=500_000):
    counts = np.zeros(cells, np.float64)
    for event in _events(events):
        onset = _event_onset(event)
        if start_us <= onset < start_us + cells * cell_us:
            counts[(onset - start_us) // cell_us] += 1
    return counts


def curve_metrics(predicted, requested, start_us, *, cell_us=500_000, scale=16.0):
    """Pearson/MAE between generated notes/s and the requested 0..1 envelope."""
    requested = np.asarray(requested, np.float64).reshape(-1)
    counts = curve_counts(predicted, start_us, len(requested), cell_us)
    actual = counts / (cell_us / 1_000_000.0) / scale
    if len(requested) < 2 or actual.std() == 0 or requested.std() == 0:
        pearson = 0.0 if actual.std() != requested.std() else 1.0
    else:
        pearson = float(np.corrcoef(actual, requested)[0, 1])
    return dict(cells=len(requested), pearson=pearson,
                mae=float(np.abs(actual - requested).mean()),
                actual_density=float(actual.mean()), requested_density=float(requested.mean()))


def curve_shuffle_gap(predicted, requested, shuffled, start_us, *, cell_us=500_000, scale=16.0):
    """Positive gap means the output follows the real shape better than a shuffled one."""
    real = curve_metrics(predicted, requested, start_us, cell_us=cell_us, scale=scale)
    fake = curve_metrics(predicted, shuffled, start_us, cell_us=cell_us, scale=scale)
    return dict(pearson_gap=real['pearson'] - fake['pearson'],
                mae_gap=fake['mae'] - real['mae'], real=real, shuffled=fake)


def type_metrics(predicted, truth):
    pred = Counter(_event_type(e) for e in _events(predicted))
    target = Counter(_event_type(e) for e in _events(truth))
    per_type = {}
    for name in NOTE_TYPES:
        p, t = pred[name], target[name]
        ratio = 1.0 if t == 0 and p == 0 else (p / t if t else None)
        per_type[name] = dict(predicted=p, target=t, count_ratio=ratio,
                              coverage=(min(ratio, 1.0) if ratio is not None else 0.0))
    measured = [per_type[name]['coverage'] for name in NOTE_TYPES if target[name] > 0]
    macro = sum(measured) / len(measured) if measured else 0.0
    return dict(per_type=per_type, macro_type_count_coverage=macro,
                scope='Type counts only; no time/type matching or event recall',
                predicted_types=dict(pred), target_types=dict(target),
                js_divergence=_js_divergence(pred, target))


def _js_divergence(left, right):
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    a = np.asarray([left.get(key, 0) for key in keys], np.float64)
    b = np.asarray([right.get(key, 0) for key in keys], np.float64)
    a = a / max(a.sum(), 1.0)
    b = b / max(b.sum(), 1.0)
    m = (a + b) / 2
    def kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log(x[mask] / np.maximum(y[mask], 1e-12))))
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def readout_metrics(songs):
    songs = list(songs)
    nonempty = sum(bool(_events(song)) for song in songs)
    counts = [len(_events(song)) for song in songs]
    return dict(songs=len(songs), nonempty=nonempty,
                nonempty_rate=(nonempty / len(songs) if songs else 0.0),
                notes_per_song=(float(np.mean(counts)) if counts else 0.0),
                min_notes=(min(counts) if counts else 0),
                max_notes=(max(counts) if counts else 0))


def control_report(predicted, truth, requested, start_us, *, shuffled=None,
                   cell_us=500_000, scale=16.0):
    result = dict(curve=curve_metrics(predicted, requested, start_us,
                                      cell_us=cell_us, scale=scale),
                  types=type_metrics(predicted, truth),
                  readout=readout_metrics([predicted]))
    if shuffled is not None:
        result['curve_shuffle'] = curve_shuffle_gap(predicted, requested, shuffled,
            start_us, cell_us=cell_us, scale=scale)
    return result


def density_request_metrics(predicted, segments):
    """Measure caller segments in heads/s; no fabricated Pearson for flat curves."""
    from .density_curve import requested_density_curve
    events = _events(predicted)
    rows = []
    for s in segments:
        requested_density_curve([s], s['start_us'], s['end_us'])
        lo, hi = s['start_us'], s['end_us']
        count = sum(lo <= _event_onset(e) < hi for e in events)
        actual = count/((hi-lo)/1e6)
        rows.append(dict(start_us=lo, end_us=hi, requested_heads_per_s=s['density'],
            actual_heads_per_s=actual, absolute_error=abs(actual-s['density']),
            unwanted_events_in_silence=count if s['density'] == 0 else 0))
    if not rows:
        return dict(specified_segments=0, mae_heads_per_s=None, pearson=None, segments=[])
    requested = np.asarray([r['requested_heads_per_s'] for r in rows])
    actual = np.asarray([r['actual_heads_per_s'] for r in rows])
    pearson = (float(np.corrcoef(actual, requested)[0, 1]) if len(rows) > 1
        and actual.std() > 0 and requested.std() > 0 else None)
    return dict(specified_segments=len(rows),
        mae_heads_per_s=float(np.mean(np.abs(actual-requested))), pearson=pearson, segments=rows)
