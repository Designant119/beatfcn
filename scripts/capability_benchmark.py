"""Freeze a held-out benchmark, validate metrics on local audio/charts, or rescore saved rollouts.

This entry point never starts training or a cloud job. Run from the repository root.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from beatfcn.events.canonical import Event, LaneLoc, MODE_TYPES
from beatfcn.events.requests import build_request
from beatfcn.model.benchmark import VERSION, audio_features, capability_metrics, scorecard, metric_source_sha256
from beatfcn.train.data import Corpus, selected_for_training


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def eligible(row):
    return (row.get('kept') and selected_for_training(row) and row.get('mode') in MODE_TYPES
            and row.get('offset_decision', {}).get('decision_version') == 2 and row.get('audio_sec', 0) >= 20)


def select_rows(rows, count, seed):
    """Round robin over mode and density bands, audio hash disjoint across BOTH modes."""
    if count <= 0:
        raise ValueError('song count must be positive')
    pools = {}
    for row in sorted(rows, key=lambda r: digest([seed, r['chart_id']])):
        density = row['n_events'] / row['audio_sec']
        band = 'density_lt2' if density < 2 else 'density_2to5' if density < 5 else 'density_ge5'
        pools.setdefault((row['mode'], band), []).append({**row, 'benchmark_stratum': band})
    chosen, seen = [], set()
    # Interleave modes first; bands are balanced within each mode when available.
    keys = sorted(pools, key=lambda k:(k[1], k[0]))
    while len(chosen) < count and any(pools.values()):
        for key in keys:
            while pools[key] and pools[key][-1]['audio_sha256'] in seen:
                pools[key].pop()
            if pools[key] and len(chosen) < count:
                row = pools[key].pop()
                chosen.append(row)
                seen.add(row['audio_sha256'])
    return chosen


def freeze(args):
    rows = read_rows(args.rows)
    # Validation is the iterative development suite; test split stays sealed.
    chosen = select_rows([r for r in rows if eligible(r) and r.get('split') == 'val'], args.songs, args.seed)
    if len(chosen) != args.songs:
        raise ValueError(f'only {len(chosen)} eligible distinct held-out audio files')
    chosen = [{**r, 'benchmark_seed':args.seed+i} for i, r in enumerate(chosen)]
    payload = {'schema': VERSION, 'purpose': 'checkpoint_regression_val_not_final_test',
               'seed': args.seed, 'source_rows_sha256': digest(rows), 'rows_sha256': digest(chosen),
               'request_kinds': ['from_scratch', 'middle_holes', 'reference'],
               'controls': 'allowed_types_only; explicit density conditions are a separate paired suite',
               'audio_clock': 'Corpus.model_shift_us with frozen offset-v2; no generated-chart alignment fit',
               'requests': [{'chart_id':r['chart_id'], 'song':r['audio_sha256'], 'mode':r['mode'],
                             'kind':kind, 'seed':args.seed+i, 'stratum':r['benchmark_stratum']}
                            for i, r in enumerate(chosen) for kind in ('from_scratch', 'middle_holes', 'reference')],
               'rows': chosen}
    payload['manifest_sha256'] = digest(payload)
    return payload


def sanity(args):
    """Training charts are allowed ONLY here: test the evaluator, never checkpoint quality."""
    root = Path(args.root)
    rows = read_rows(root / 'manifest/charts.jsonl')
    available = [r for r in rows if eligible(r) and
                 (root / 'canon' / r['source'] / f"{r['native_id']}.json").exists() and
                 (root / 'mel10' / f"{r['audio_sha256']}.npy").exists()]
    chosen = select_rows(available, args.songs, args.seed)
    corpus = Corpus(root=str(root))
    corpus.train_rows(chosen)
    records, failures = [], []
    for i, row in enumerate(chosen):
        try:
            duration = int(round(row['audio_sec'] * 1e6))
            sample = build_request(corpus.chart(row['chart_id']), kind='from_scratch', seed=args.seed+i,
                                   audio_id=row['audio_sha256'], audio_duration_us=duration, control_dropout=False)
            if not sample.ok:
                raise ValueError('quarantined model view')
            truth = [e for g in sample.model_view.target_groups for e in g.events]
            af = audio_features(corpus.mel(row['audio_sha256']), duration)
            rng = np.random.default_rng(args.seed+i)
            def move(e, t):
                length = e.end_us-e.onset_us
                t = min(max(int(t), 0), duration-length-1)
                return replace(e, onset_us=t, end_us=t+length)
            variants = {'human_self':truth,
                        'shift_137ms':[move(e, (e.onset_us+137_000) % duration) for e in truth],
                        'random_time':[move(e, rng.integers(0, duration)) for e in truth],
                        'empty':[]}
            if row['mode'] == 'mania_4k':
                variants['single_lane'] = [replace(e, location=LaneLoc(0)) for e in truth]
            for condition, pred in variants.items():
                records.append({'song':row['audio_sha256'], 'chart_id':row['chart_id'], 'mode':row['mode'],
                                'source_split':row['split'], 'condition':condition,
                                'benchmark':capability_metrics(pred, truth, sample.request, af)})
        except (OSError, ValueError) as exc:
            failures.append({'chart_id':row['chart_id'], 'reason':f'{type(exc).__name__}:{exc}'})
    summary = {}
    for mode, condition in sorted({(r['mode'], r['condition']) for r in records}):
        rs = [r for r in records if r['mode']==mode and r['condition']==condition]
        fields = ('anchor_20ms_precision', 'anchor_20ms_shift_excess', 'density_flux_spearman',
                  'rapid_same_lane_fraction', 'hand_onset_imbalance', 'lane_bigram_entropy')
        summary[f'{mode}/{condition}'] = {'songs':len(rs)}
        for name in fields:
            values = [r['benchmark']['generated'][name] for r in rs if r['benchmark']['generated'][name] is not None]
            summary[f'{mode}/{condition}'][name] = float(np.median(values)) if values else None
    return {'schema':VERSION, 'purpose':'metric_negative_controls_only_NOT_model_evaluation', 'seed':args.seed,
            'metric_source_sha256':metric_source_sha256(), 'numpy_version':np.__version__,
            'requested':args.songs, 'selected':len(chosen), 'failures':failures,
            'warning':'These local charts may be training data; do not use these ranges as acceptance thresholds.',
            'summary':summary, 'records':records}


def rescore(args):
    source = json.loads(Path(args.evaluation).read_text())
    records, cache = source['songs'], {}
    for r in records:
        if 'generated_events' not in r:
            r.pop('benchmark', None)
            continue  # Failure retained in the scorecard denominator.
        if 'request' not in r or 'truth_events' not in r:
            raise ValueError('rescore needs a v3 rollout with persisted request and truth; do not reconstruct old random masks')
        key = (r['song'], r['request']['audio_duration_us'])
        if key not in cache:
            try:
                mel = np.load(Path(args.root) / 'mel10' / f"{r['song']}.npy", allow_pickle=False)
                cache[key] = (audio_features(mel, key[1]), None)
            except (OSError, ValueError) as exc:
                cache[key] = (None, f'{type(exc).__name__}:{exc}')
        af, error = cache[key]
        r['benchmark'] = capability_metrics([Event.from_json(e) for e in r['generated_events']],
                                            [Event.from_json(e) for e in r['truth_events']], r['request'],
                                            af, audio_error=error)
    return {'schema':VERSION, 'input_evaluation_sha256':hashlib.sha256(Path(args.evaluation).read_bytes()).hexdigest(),
            'metric_source_sha256':metric_source_sha256(), 'numpy_version':np.__version__,
            'checkpoint': {k:source.get(k) for k in ('run','step','checkpoint_sha256','model_source_sha256')},
            'scorecard':scorecard(records), 'songs':records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('freeze', 'sanity', 'rescore'):
        p = sub.add_parser(name)
        p.add_argument('--output', required=True)
        if name != 'rescore':
            p.add_argument('--songs', type=int, default=40 if name=='freeze' else 12)
            p.add_argument('--seed', type=int, default=20261024)
        if name == 'freeze': p.add_argument('--rows', required=True)
        else: p.add_argument('--root', required=True)
        if name == 'rescore': p.add_argument('--evaluation', required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    result = {'freeze':freeze, 'sanity':sanity, 'rescore':rescore}[args.command](args)
    if args.command != 'freeze': result['cpu_wall_s'] = time.perf_counter()-started
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({'output':str(dest), 'schema':result['schema'], 'summary':result.get('summary'),
                      'selected':result.get('selected', len(result.get('rows', []))),
                      'failures':result.get('failures', [])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
