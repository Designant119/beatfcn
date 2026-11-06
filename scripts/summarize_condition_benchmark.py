"""Retrieve and summarize the frozen condition comparison without dropping failed requests."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beatfcn.events.controls import band_of
from beatfcn.model.benchmark import scorecard

REMOTE = 'evolution/capability-evolution-20261024/condition-v1/benchmark'
LOCAL = ROOT / 'reports/condition_v1_benchmark'


def interval(values):
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    ci = None
    if len(x) > 1:
        means = np.random.default_rng(20261024).choice(x, (5000, len(x)), replace=True).mean(1)
        ci = np.quantile(means, [.025, .975]).tolist()
    return {'mean': float(x.mean()) if len(x) else None, 'song_bootstrap_95': ci, 'songs': len(x)}


def summarize(protocol, records):
    indexed = {r['id']: r for r in records}
    if len(indexed) != len(records) or set(indexed) - {q['id'] for q in protocol['expected']}:
        raise ValueError('duplicate or unexpected benchmark record')
    rows = [{**q, 'complete': False, 'legality': [], 'failure': None}
            if q['id'] not in indexed else indexed[q['id']] for q in protocol['expected']]
    groups, densities, pairs = defaultdict(list), defaultdict(list), defaultdict(dict)
    for r in rows:
        if r['request_kind'].startswith('density_'):
            scope = 'whole_song' if r['request_kind'].startswith('density_song_') else 'window'
            densities[(r['checkpoint'], r['mode'], scope)].append(r)
        else:
            groups[(r['checkpoint'], r['mode'], r['request_kind'])].append(r)
            pairs[(r['mode'], r['request_kind'], r['song'])][r['checkpoint']] = r
    good = lambda r: r['complete'] and not r['legality']
    cells = {}
    for key, cases in groups.items():
        recorded = [r for r in cases if r['id'] in indexed]
        valid = [r for r in cases if good(r)]
        cell = {'requested': len(cases), 'recorded': len(recorded), 'not_recorded': len(cases) - len(recorded),
                'complete_legal': len(valid), 'failed_recorded': sum(not good(r) for r in recorded),
                'failure_reasons': [r.get('failure') for r in recorded if not good(r)]}
        for ms in ('20ms', '50ms', '100ms'):
            cell[ms] = {metric: interval([r['editing'][ms][metric]['saved_fraction'] for r in valid])
                        for metric in ('timing_point', 'rhythm_head', 'typed_head', 'full_note')}
            cell[ms]['audio_baseline_delta'] = {
                baseline: interval([r['editing'][ms]['timing_point']['saved_fraction'] - r['audio_only_baselines'][baseline][ms]['saved_fraction']
                                    for r in valid if r['editing'][ms]['timing_point']['saved_fraction'] is not None])
                for baseline in ('onsets_us', 'strong_onsets_us')}
        cell['count_ratio'] = interval([r['quality']['density_curve']['count_ratio'] for r in valid])
        cell['note_f1_20ms'] = interval([r['scores']['note_20ms']['f1'] for r in valid])
        cell['hold_joint_f1_20ms'] = interval([r['scores']['hold_joint_20ms']['f1'] for r in valid])
        cell['non_drag_saved_fraction'] = interval([r['subsets']['non_drag']['full_note']['saved_fraction'] for r in valid])
        cells['/'.join(key)] = cell
    control, response_pairs = {}, defaultdict(dict)
    for key, cases in densities.items():
        by_song = defaultdict(dict)
        for r in cases:
            by_song[r['song']][r['request_kind']] = r
        prefix = 'density_song_' if key[-1] == 'whole_song' else 'density_'
        orders = tuple(prefix + band for band in ('low', 'mid', 'high'))
        valid, feasible, monotonic, flat, in_band = [], 0, 0, 0, 0
        for song, cases_by_kind in by_song.items():
            current = [cases_by_kind.get(k) for k in orders]
            if not all(r and good(r) and r.get('achieved') is not None for r in current):
                continue
            if any(r['minimum_from_observed'] >= r['requested'][1] for r in current):
                continue
            feasible += 1
            values = [r['achieved'] for r in current]
            monotonic += values[0] < values[1] < values[2]
            flat += max(values) - min(values) < 1e-6
            in_band += sum(band_of('density_notes_per_second', r['achieved']) == r['requested'] for r in current)
            valid.append(values)
            response_pairs[(key[1], key[2], song)][key[0]] = {
                'high_minus_low': values[2] - values[0],
                'monotonic_fraction': float(values[0] < values[1] < values[2]),
                'in_band_fraction': sum(band_of('density_notes_per_second', r['achieved']) == r['requested'] for r in current) / 3,
            }
        control['/'.join(key)] = {'requested_songs': len(by_song), 'complete_feasible_sweeps': feasible,
                                  'positive_monotonic_songs': monotonic, 'flat_songs': flat,
                                  'in_band': in_band, 'in_band_denominator': feasible * 3,
                                  'achieved_by_band': {k: interval([v[i] for v in valid]) for i, k in enumerate(orders)},
                                  'high_minus_low': interval([v[2] - v[0] for v in valid])}
    changes = defaultdict(lambda: defaultdict(list))
    comparisons = (('3500', 'r02-density-supervised'), ('3500', 'r02-density-contrastive'),
                   ('r02-density-supervised', 'r02-density-contrastive'))
    for (mode, kind, song), labels in pairs.items():
        for baseline, candidate in comparisons:
            a, b = labels.get(baseline), labels.get(candidate)
            if a is None or b is None or not good(a) or not good(b):
                continue
            label = '/'.join((candidate + '-minus-' + baseline, mode, kind))
            for metric in ('timing_point', 'full_note'):
                av, bv = [r['editing']['20ms'][metric]['saved_fraction'] for r in (a, b)]
                if av is not None and bv is not None:
                    changes[label][metric].append(bv - av)
            av, bv = [r['scores']['note_20ms']['f1'] for r in (a, b)]
            if av is not None and bv is not None:
                changes[label]['note_f1_20ms'].append(bv - av)
    control_changes = defaultdict(lambda: defaultdict(list))
    for (mode, scope, song), labels in response_pairs.items():
        for baseline, candidate in comparisons:
            if baseline not in labels or candidate not in labels:
                continue
            label = '/'.join((candidate + '-minus-' + baseline, mode, scope))
            for metric in labels[baseline]:
                control_changes[label][metric].append(labels[candidate][metric] - labels[baseline][metric])
    labels = sorted({r['checkpoint'] for r in rows})
    return {'complete': all(good(r) for r in rows), 'expected': len(rows), 'received': len(records),
            'not_recorded': len(rows) - len(records), 'failed_recorded': sum(not good(r) for r in records),
            'complete_legal': sum(good(r) for r in records),
            'protocol_identity': {k: protocol.get(k) for k in ('source_sha256', 'manifest_sha256', 'evaluator_sha256', 'checkpoints')},
            'cells': cells, 'controls': control,
            'paired_changes': {k: {metric: interval(v) for metric, v in metrics.items()} for k, metrics in changes.items()},
            'paired_control_changes': {k: {metric: interval(v) for metric, v in metrics.items()} for k, metrics in control_changes.items()},
            'scorecards': {label: scorecard([r for r in records if r['checkpoint'] == label and not r['request_kind'].startswith('density_')])
                          for label in labels},
            'limitations': ['Development panel; no sealed-test or human-time claim.',
                           'Quality means exclude failed requests, but requested denominators and failures remain explicit.',
                           'Unrecorded requests are unexecuted or interrupted, not evidence of model failure. Scorecards cover recorded requests only; cells retain planned denominators.',
                           'Condition feasibility excludes bands already violated by locked heads, with coverage retained.',
                           'Reference reconstruction cost is a proxy; valid alternate charts can be penalized.',
                           'Drag/non-drag separation is not a gesture-quality verdict.'],
            'promotion': 'requires joint review; no automatic pass'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pull', action='store_true')
    parser.add_argument('--available-only', action='store_true', help='Retrieve only committed records; retain all planned requests in the summary denominator.')
    args = parser.parse_args()
    if args.available_only and not args.pull:
        parser.error('--available-only requires --pull')
    LOCAL.mkdir(exist_ok=True)
    if args.pull:
        import modal
        volume = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
        def fetch(name):
            payload = bytearray()
            for chunk in volume.read_file(REMOTE + '/' + name):
                payload.extend(chunk)
                if len(payload) > 8 * 2**20:
                    raise ValueError('benchmark record exceeds transfer bound')
            json.loads(payload)
            path = LOCAL / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_bytes() != payload:
                raise ValueError('immutable benchmark artifact changed: ' + name)
            temporary = path.with_suffix('.json.tmp')
            temporary.write_bytes(payload)
            temporary.replace(path)
        fetch('protocol.json')
        protocol = json.loads((LOCAL / 'protocol.json').read_text())
        names = ['records/' + q['id'] + '.json' for q in protocol['expected']]
        if args.available_only:
            available = {'records/' + Path(entry.path).name for entry in volume.listdir(REMOTE + '/records')
                         if entry.path.endswith('.json')}
            if available - set(names):
                raise ValueError('committed record is not in the frozen protocol')
            names = [name for name in names if name in available]
        missing = [name for name in names if not (LOCAL / name).exists()]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch, missing))
        (LOCAL / 'retrieval.json').write_text(json.dumps({'at_unix': time.time(), 'available_only': args.available_only,
                                                        'expected': len(protocol['expected']), 'selected': len(names),
                                                        'downloaded': len(missing)}, indent=2) + '\n')
    protocol = json.loads((LOCAL / 'protocol.json').read_text())
    files = sorted((LOCAL / 'records').glob('*.json'))
    result = summarize(protocol, [json.loads(p.read_text()) for p in files])
    result['source_files_sha256'] = {str(p.relative_to(LOCAL)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    target = ROOT / 'reports/condition_v1_benchmark_summary.json'
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'path': str(target), 'complete': result['complete'], 'expected': result['expected'], 'received': result['received']}))


if __name__ == '__main__':
    main()
