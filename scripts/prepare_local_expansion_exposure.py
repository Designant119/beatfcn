"""Freeze a larger request pool and reproduce the historical sampling policy."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from beatfcn.train.local_sampling import LocalRequestSampler
from scripts.run_local_learning_curve import condition_view

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def write_new(path, value):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def records(charts):
    # Same flattening contract as LocalRequestPanel, without opening audio caches.
    return [dict(q, chart_index=i, chart_id=c['chart_id'], audio_sha256=c['audio_sha256'],
        component=c['component'], split=c['split'], mode=c['mode'], stratum=c['stratum'])
        for i, c in enumerate(charts) for q in c['requests']]


def sample(pool, seed, steps):
    sampler = LocalRequestSampler(pool, seed=seed)
    views = random.Random(seed)
    out = []
    for step in range(1, steps + 1):
        ids = sampler.next_batch()
        specified = [np.flatnonzero(condition_view(pool[i], views)[0][:, 2]).tolist() for i in ids]
        out.append(dict(effective_update=step, indices=ids, specified_condition_fields=specified))
    return out, sampler.pool_identity


def exposure_summary(pool, steps):
    counts = {k: Counter() for k in ('mode', 'song', 'chart', 'stratum', 'request', 'specified_fields')}
    tokens = 0
    for step in steps:
        for i, fields in zip(step['indices'], step['specified_condition_fields'], strict=True):
            r = pool[i]
            if r['split'] != 'train':
                raise ValueError('Exposure reaches a validation song')
            tokens += r['tokens']
            for field, value in [('mode', r['mode']), ('song', r['component']), ('chart', r['chart_id']),
                                 ('stratum', r['mode'] + '/' + r['stratum']), ('request', str(i))]:
                counts[field][value] += 1
            counts['specified_fields'].update(map(str, fields))
    return dict(valid_tokens=tokens, requests=4 * len(steps),
                independent_songs=len(counts['song']), distinct_charts=len(counts['chart']),
                distinct_requests=len(counts['request']), counts={k: dict(v) for k, v in counts.items()})


def run():
    plan_path = 'bench/rhythm-v2/d94-exposure-preparation-v1.json'
    plan = read(plan_path)
    for p, expected in plan['source_sha256'].items():
        if sha(p) != expected:
            raise ValueError('Frozen preparation source changed: ' + p)
    baseline, all_tape = read(plan['baseline_tape']), read(plan['all_source_tape'])
    previous = read(plan['baseline_exposure'])
    old_pool = records(baseline['charts'])
    replay, identity = sample(old_pool, previous['seed'], len(previous['steps']))
    if identity != previous['pool_identity'] or replay != previous['steps']:
        raise ValueError('Historical sampling and condition views did not reproduce exactly')

    inventory, old_features = read(plan['inventory']), read(plan['old_features'])
    if not inventory['complete'] or any(r['error_type'] != 'NotFoundError' for r in inventory['failures']):
        raise ValueError('A complete inventory with explicit absence reasons is required')
    counts = Counter(r['audio_sha256'] for r in inventory['records'])
    if any(n != 3 for n in counts.values()):
        raise ValueError('Partially available audio must not enter the panel')
    available = set(counts) | {r['audio_sha256'] for r in old_features['records'] if r['passed']}
    charts = [c for c in all_tape['charts'] if c['audio_sha256'] in available]
    if charts[:len(baseline['charts'])] != baseline['charts']:
        raise ValueError('The old request/chart indices must stay identical')
    by_split = {s: {c['component'] for c in charts if c['split'] == s} for s in ('train', 'val')}
    if by_split['train'] & by_split['val']:
        raise ValueError('Song split leakage')
    expected_val = {r['chart_id'] for r in read(plan['evaluation20'])['records']}
    if not expected_val <= {c['chart_id'] for c in charts if c['split'] == 'val'}:
        raise ValueError('Existing twenty-song evaluation changed')

    tape = dict(id='D94-expanded-target-tape-v1', complete=True, failures=[], charts=charts,
        training_ready=False, feature_admission_pending=True,
        excluded_missing_feature_chart_ids=[c['chart_id'] for c in all_tape['charts'] if c['audio_sha256'] not in available],
        plan_sha256=sha(plan_path), source_policy=all_tape['source_policy'],
        evaluation20=plan['evaluation20'], optimizer_updates=0, cloud_compute_calls=0,
        summary={s: dict(charts=sum(c['split'] == s for c in charts), songs=len(by_split[s]),
            mode_charts=dict(Counter(c['mode'] for c in charts if c['split'] == s))) for s in by_split},
        meaning='Target preparation only. Availability does not certify downloaded tensors or authorize a run.')
    pool = records(charts)
    steps, identity = sample(pool, previous['seed'], len(previous['steps']))
    # Both arms start at the same verified step2. Its eight actual old requests
    # are copied; the new sampler's first two draws were never trained on.
    steps[:2] = previous['steps'][:2]
    if any(pool[i] != old_pool[i] for step in steps[:2] for i in step['indices']):
        raise ValueError('Shared step2 refers to changed requests')
    old_summary, new_summary = exposure_summary(old_pool, previous['steps']), exposure_summary(pool, steps)
    write_new(plan['output_tape'], tape)
    exposure = dict(id='D94-expanded-exposure-1000-v1', request_tape=plan['output_tape'],
        request_tape_sha256=sha(plan['output_tape']), pool_identity=identity,
        seed=previous['seed'], steps=steps, effective_batch_requests=4, microbatch_requests=1,
        phira_fraction=.75, condition_view_probabilities=previous['condition_view_probabilities'],
        shared_prefix_updates=2, shared_prefix_exposure=plan['baseline_exposure'],
        sampler_first_two_candidate_draws_discarded=True, exposure=new_summary,
        sampler_code_sha256=sha('beatfcn/train/local_sampling.py'),
        view_code_sha256=sha('scripts/run_local_learning_curve.py'),
        training_started=False, training_ready=False, feature_admission_pending=True)
    write_new(plan['output_exposure'], exposure)
    report = dict(id='D94-exposure-preparation-v1', complete=True, optimizer_updates=0,
        new_cloud_compute_calls=0, quality_passed=False, feature_admission_pending=True,
        plan_sha256=sha(plan_path), baseline_full_1000_step_replay_exact=True,
        old_target_records_unchanged=len(baseline['charts']),
        all_evaluation20_retained=True, split_leakage=0, pool_summary=tape['summary'],
        excluded_missing_feature_chart_ids=tape['excluded_missing_feature_chart_ids'],
        baseline_exposure=old_summary, expanded_exposure=new_summary,
        common_parent_checkpoint_sha256=sha(plan['common_parent_checkpoint']),
        source_sha256={**plan['source_sha256'], plan['output_tape']:sha(plan['output_tape']),
                       plan['output_exposure']:sha(plan['output_exposure'])},
        limitation='Fixed request budget does not equal token/FLOP budget; both totals are explicit. No scaling or convergence claim.')
    write_new(plan['output_report'], report)
    print(json.dumps(dict(complete=True, replay_exact=True, pool=tape['summary'],
        old_tokens=old_summary['valid_tokens'], new_tokens=new_summary['valid_tokens'],
        old_songs=old_summary['independent_songs'], new_songs=new_summary['independent_songs'])))


if __name__ == '__main__':
    run()
