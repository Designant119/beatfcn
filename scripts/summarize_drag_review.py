"""Type-separated Phira diagnostics; temporal clusters are not gesture labels."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beatfcn.events.canonical import Event
from beatfcn.model.evaluation import score_events


def dense_windows(events, gap_ms):
    """Reference-only clusters with >=3 distinct drag times, clipped to endpoints."""
    times = sorted({e.onset_us for e in events if e.type == 'drag'})
    groups = []
    for t in times:
        if not groups or t - groups[-1][-1] > gap_ms * 1000:
            groups.append([t])
        else:
            groups[-1].append(t)
    return [(g[0], g[-1]) for g in groups if len(g) >= 3]


def score(pred, truth):
    if not pred and not truth:
        return None
    return score_events(pred, truth, 'phira')['note_20ms']


def details(record):
    pred = [Event.from_json(e) for e in record['generated_events']]
    truth = [Event.from_json(e) for e in record['truth_events']]
    out = {k: record[k] for k in ('checkpoint', 'song', 'chart_id', 'request_kind')}
    out['counts'] = {'generated': dict(Counter(e.type for e in pred)),
                     'reference': dict(Counter(e.type for e in truth))}
    out['generated_drag_fraction'] = sum(e.type == 'drag' for e in pred) / len(pred) if pred else None
    out['reference_drag_fraction'] = sum(e.type == 'drag' for e in truth) / len(truth) if truth else None
    out['scores'] = {'all': score(pred, truth)}
    for label in ('drag', 'non_drag', 'tap', 'hold', 'flick'):
        keep = (lambda e: e.type != 'drag') if label == 'non_drag' else (lambda e, label=label: e.type == label)
        out['scores'][label] = score(list(filter(keep, pred)), list(filter(keep, truth)))
    out['dense_drag_sensitivity'] = {}
    for gap in (80, 160, 320):
        windows = dense_windows(truth, gap)
        inside = lambda e: any(a <= e.onset_us <= b for a, b in windows)
        p, t = list(filter(inside, pred)), list(filter(inside, truth))
        out['dense_drag_sensitivity'][str(gap)] = {
            'reference_cluster_count': len(windows),
            'reference_cluster_duration_s': sum(b-a for a, b in windows) / 1e6,
            'reference_notes_in_windows': len(t),
            'generated_notes_in_windows': len(p),
            'inside_note_20ms': score(p, t),
            'outside_note_20ms': score([e for e in pred if not inside(e)], [e for e in truth if not inside(e)]),
        }
    return out


def mean(values):
    values = [v for v in values if v is not None]
    return {'mean': float(np.mean(values)) if values else None, 'n': len(values)}


def main():
    directory = Path(sys.argv[1])
    raw = [json.loads(p.read_text()) for p in sorted((directory / 'records').glob('*.json'))]
    rows = [details(r) for r in raw if r['mode'] == 'phira' and r['condition'] == 'baseline' and r.get('complete')]
    output = {'schema': 'beatfcn.drag_review.v1', 'records': rows, 'groups': {}, 'paired': {},
              'limitations': [
                  'Uses the frozen generated outputs; no training or new inference.',
                  'Type F1 measures reference reproduction, not musical or gesture correctness.',
                  'Temporal clusters ignore spatial motion and cannot identify gestures.',
                  'Clusters require three distinct drag onsets; simultaneous stacks alone do not qualify.',
                  'Dense windows come only from reference labels and use inclusive endpoints; results vary with gap thresholds.',
                  'Empty/empty type subsets are undefined and excluded, with denominators reported.',
                  'Only complete requests are analyzed; consult main report for all failures.',
              ]}
    for kind in ('from_scratch', 'middle_holes', 'reference'):
        for ck in ('3500', '4500'):
            rs = [r for r in rows if r['checkpoint'] == ck and r['request_kind'] == kind]
            group = {'completed': len(rs)}
            for feature in ('generated_drag_fraction', 'reference_drag_fraction'):
                group[feature] = mean([r[feature] for r in rs])
            group['note_f1_20ms'] = {label: mean([r['scores'][label]['f1'] if r['scores'][label] else None for r in rs])
                                    for label in ('all', 'drag', 'non_drag', 'tap', 'hold', 'flick')}
            group['dense_drag_sensitivity'] = {}
            for gap in ('80', '160', '320'):
                samples = [r['dense_drag_sensitivity'][gap] for r in rs]
                group['dense_drag_sensitivity'][gap] = {
                    'requests_with_clusters': sum(s['reference_cluster_count'] > 0 for s in samples),
                    **{part: mean([s[part]['f1'] if s[part] else None for s in samples])
                       for part in ('inside_note_20ms', 'outside_note_20ms')},
                }
            output['groups'][f'{ck}/{kind}'] = group
        a = {r['song']: r for r in rows if r['checkpoint'] == '3500' and r['request_kind'] == kind}
        b = {r['song']: r for r in rows if r['checkpoint'] == '4500' and r['request_kind'] == kind}
        result = {'both_complete': len(a.keys() & b.keys())}
        for label in ('all', 'drag', 'non_drag', 'tap', 'hold', 'flick'):
            values = [b[s]['scores'][label]['f1'] - a[s]['scores'][label]['f1'] for s in sorted(a.keys() & b.keys())
                      if a[s]['scores'][label] is not None and b[s]['scores'][label] is not None]
            draws = np.random.default_rng(20261024).choice(values, (5000, len(values)), replace=True).mean(1) if values else None
            result[label] = {'delta_4500_minus_3500': mean(values),
                             'song_bootstrap_95': np.quantile(draws, [.025, .975]).tolist() if len(values) >= 2 else None}
        output['paired'][kind] = result
    path = ROOT / 'reports/capability_drag_review.json'
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({'output': str(path), 'completed_requests_analyzed': len(rows)}))


if __name__ == '__main__':
    main()
