"""Summarize a frozen-checkpoint diagnostic; optionally fetch its saved result."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beatfcn.events.controls import band_of
from beatfcn.model.control_eval import spearman


def ci(values):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return {'mean': None, 'song_bootstrap_95': None, 'songs': 0}
    interval = None
    if len(x) > 1:
        draws = np.random.default_rng(20261024).choice(x, (5000, len(x)), replace=True).mean(1)
        interval = np.quantile(draws, [.025, .975]).tolist()
    return {'mean': float(x.mean()), 'song_bootstrap_95': interval, 'songs': len(x)}


def summarize(raw):
    result = {'source_sha256': raw['source_sha256'], 'complete': raw.get('complete', False),
              'controls_received': len(raw['controls']), 'controls_expected': raw['expected_control_records'],
              'history_received': len(raw['history']), 'history_expected': raw['expected_history_records'],
              'failures': raw['failures'], 'checkpoints': raw['checkpoints'], 'controls': {}, 'history': {},
              'training_updates': 0, 'limitations': [
                  'Density measured on one core with human preceding context; not whole-song adherence.',
                  'Only density was tested; the other seven numeric conditions remain unverified here.',
                  'History ablation sensitivity is not the causal benefit of training with an adapter.',
                  'Small development panel; intervals are descriptive, not human-calibrated acceptance.']}
    groups = defaultdict(list)
    for row in raw['controls']:
        groups[row['checkpoint'] + '/' + row['mode']].append(row)
    for key, rows in groups.items():
        by_song = defaultdict(list)
        for row in rows:
            if row['complete'] and row['legal'] and row['achieved'] is not None:
                by_song[row['song']].append(row)
        paired = [sorted(rs, key=lambda r: r['requested'][0]) for rs in by_song.values() if len(rs) == 3]
        deltas = [rs[-1]['achieved'] - rs[0]['achieved'] for rs in paired]
        correlations = [spearman([sum(r['requested']) / 2 for r in rs], [r['achieved'] for r in rs]) for rs in paired]
        monotonic = [rs[0]['achieved'] < rs[-1]['achieved'] and all(a['achieved'] <= b['achieved'] for a, b in zip(rs, rs[1:]))
                     for rs in paired]
        valid = [r for rs in by_song.values() for r in rs]
        bands = defaultdict(list)
        for r in valid:
            bands[str(r['requested'])].append(r['achieved'])
        result['controls'][key] = {
            'attempted': len(rows), 'complete_legal': len(valid), 'paired_songs': len(paired),
            'positive_monotonic_songs': sum(monotonic), 'monotonic_rate': sum(monotonic) / len(paired) if paired else None,
            'high_minus_low_density': ci(deltas),
            'within_song_spearman': ci([r for r in correlations if r is not None]),
            'flat_response_songs': sum(max(r['achieved'] for r in rs) == min(r['achieved'] for r in rs) for rs in paired),
            'in_band_local_core': sum(band_of(r['field'], r['achieved']) == r['requested'] for r in valid) / len(valid) if valid else None,
            'achieved_by_band': {b: {'mean': float(np.mean(v)), 'n': len(v)} for b, v in bands.items()}}
    by_checkpoint = defaultdict(list)
    for row in raw['history']:
        by_checkpoint[row['checkpoint']].append(row)
    for checkpoint, rows in by_checkpoint.items():
        baseline = [r['conditions']['baseline'] for r in rows]
        n = sum(r['choice_targets'] for r in baseline)
        report = {'cores_or_positions': len(rows), 'choice_targets': n,
                  'choice_accuracy': sum(r['choice_correct'] for r in baseline) / n if n else None,
                  'first_free_accuracy': sum(r['first_free_correct'] for r in baseline) / n if n else None,
                  'conditions': {}}
        for condition in sorted({c for r in rows for c in r['conditions']} - {'baseline'}):
            by_song = defaultdict(list)
            for row in rows:
                current, base = row['conditions'].get(condition), row['conditions']['baseline']
                if current is not None and current['choice_nll'] is not None:
                    by_song[row['song']].append(current['choice_nll'] - base['choice_nll'])
            report['conditions'][condition] = {'choice_nll_delta': ci([float(np.mean(v)) for v in by_song.values()]),
                'mean_probability_tv': float(np.mean([r['conditions'][condition]['lane_probability_total_variation']
                                                      for r in rows if condition in r['conditions']]))}
        result['history'][checkpoint] = report
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', default='history-density-v1')
    parser.add_argument('--pull', action='store_true')
    args = parser.parse_args()
    if Path(args.stage).name != args.stage:
        raise ValueError('stage must be a basename')
    raw_path = ROOT / f'reports/{args.stage}_raw.json'
    if args.pull:
        import modal
        volume = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
        remote = f'evolution/capability-evolution-20261024/rhythm-diagnostics/{args.stage}/result.json'
        payload = bytearray()
        for chunk in volume.read_file(remote):
            payload.extend(chunk)
            if len(payload) > 16 * 2**20:
                raise ValueError('diagnostic download exceeded 16 MiB bound')
        json.loads(payload)
        raw_path.write_bytes(payload)
    raw = json.loads(raw_path.read_text())
    report = summarize(raw)
    report['raw_sha256'] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    target = ROOT / f'reports/{args.stage}_summary.json'
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
