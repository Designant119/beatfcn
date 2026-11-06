"""Complete-event and shared-physical-interval diagnostics; never changes outputs."""
from collections import Counter
import numpy as np


def physical_events(events, identity):
    shift = identity['source_clip_start_us']
    return [dict(e, onset_us=e['onset_us'] + shift, end_us=e['end_us'] + shift) for e in events]


def in_interval(events, interval):
    lo, hi = interval
    return [e for e in events if lo <= e['onset_us'] < hi]


def event_metrics(pred, truth, mode):
    edges = [[j for j, b in enumerate(truth) if a['type'] == b['type'] and a['lane'] == b['lane']
              and abs(a['onset_us'] - b['onset_us']) <= 20_000
              and abs(a['end_us'] - b['end_us']) <= 20_000] for a in pred]
    matches = {}
    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in matches or augment(matches[j], seen):
                matches[j] = i
                return True
        return False
    for i in range(len(pred)):
        augment(i, set())
    def groups(events):
        result = {}
        for e in events:
            result.setdefault(e['onset_us'], []).append((e['type'], e['lane']))
        return Counter((head, tuple(sorted(notes))) for head, notes in result.items())
    pg, tg = groups(pred), groups(truth)
    onset = [(pred[i]['onset_us'] - truth[j]['onset_us']) / 1000 for j, i in matches.items()]
    tails = [(pred[i]['end_us'] - truth[j]['end_us']) / 1000 for j, i in matches.items()
             if truth[j]['type'] == 'hold']
    conflicts = 0
    if mode == 'mania_4k':
        for lane in range(4):
            last_head, furthest_tail = -1, -1
            for e in sorted((e for e in pred if e['lane'] == lane), key=lambda e: (e['onset_us'], e['end_us'])):
                conflicts += int(e['onset_us'] == last_head or e['onset_us'] < furthest_tail)
                last_head, furthest_tail = e['onset_us'], max(furthest_tail, e['end_us'])
    per_type = {kind: dict(pred=sum(e['type'] == kind for e in pred),
                          true=sum(e['type'] == kind for e in truth),
                          matched=sum(truth[j]['type'] == kind for j in matches))
                for kind in ('tap', 'hold', 'flick', 'drag')}
    def error_stats(values):
        return dict(n=len(values), signed_ms=values,
            median_abs_ms=float(np.median(np.abs(values))) if values else None,
            p90_abs_ms=float(np.percentile(np.abs(values), 90)) if values else None)
    return dict(pred=len(pred), true=len(truth), matched=len(matches),
        full_note_f1=2 * len(matches) / (len(pred) + len(truth)) if pred or truth else 1.,
        full_event_false_positive=len(pred) - len(matches), full_event_false_negative=len(truth) - len(matches),
        exact_onset_group_recall=sum((pg & tg).values()) / len(tg) if tg else float(not pg),
        mania_lane_conflicts=conflicts, per_type=per_type,
        matched_onset_errors=error_stats(onset), matched_hold_tail_errors=error_stats(tails))
