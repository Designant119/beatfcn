"""Freeze new-song health cases, then join completed feature admission evidence.

Selection uses source labels only, before any model predictions. No target is
shortened to satisfy the health gate. This script never trains or uses Modal.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.prepare_local_expansion_exposure import records

ROOT = Path(__file__).resolve().parents[1]
SELECTION = 'reports/d94_probe_selection_v1.json'
FEATURES = 'reports/d94_merged_feature_admission_v1.json'
PLAN = 'bench/rhythm-v2/d94-new-data-probe-v1.json'
TAPE = 'reports/d94_expanded_target_tape_v1.json'
OLD_TAPE = 'reports/d67_local_request_tape_v1.json'
EXPOSURE = 'inputs/d94_local_requests/learning-exposure-1000-v1.json'
BASE_SHAPES = dict(local=400, mel=1600, song=896, reference=1)


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    h = hashlib.sha256()
    with (ROOT / path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def padding_shapes(song_frames):
    """Retain the old shape for old lengths; add one bounded, masked bucket."""
    if not isinstance(song_frames, int) or not 0 < song_frames <= 1024:
        raise ValueError('Song context outside the registered full-length buckets')
    return dict(BASE_SHAPES, song=896 if song_frames <= 896 else 1024)


def choose_case(pool, predicate, used_components, role):
    eligible = [(i, r) for i, r in enumerate(pool) if r['split'] == 'train'
                and r['component'] not in used_components and predicate(r)]
    eligible.sort(key=lambda item: (item[1]['tokens'], item[0]))
    if not eligible:
        raise ValueError('No real, distinct training song for role: ' + role)
    rank = (len(eligible) - 1) // 2
    index, row = eligible[rank]
    used_components.add(row['component'])
    return index, row, dict(role=role, eligible_requests=len(eligible), rank=rank,
        rule='lower median by (full target token count, record index); distinct new train song')


def select_cases():
    tape, old = read(TAPE), read(OLD_TAPE)
    pool = records(tape['charts'])
    used = {c['component'] for c in old['charts']}
    exposure = read(EXPOSURE)
    views = {}
    for step in exposure['steps']:
        for i, fields in zip(step['indices'], step['specified_condition_fields'], strict=True):
            views.setdefault(i, fields)
    predicates = [
        ('phira_empty', lambda r: r['mode'] == 'phira' and r['event_count'] == 0),
        ('phira_low_tap', lambda r: r['mode'] == 'phira' and r['stratum'].startswith('EZ/')
            and set(r['types']) == {'tap'}),
        ('phira_hold_tail_carry', lambda r: r['mode'] == 'phira' and r['types'].get('hold', 0) > 0
            and r['incoming_holds'] > 0 and r['request']['end_us'] == r['request']['duration_us']),
        ('phira_flick', lambda r: r['mode'] == 'phira' and r['types'].get('flick', 0) > 0),
        ('phira_drag', lambda r: r['mode'] == 'phira' and r['types'].get('drag', 0) > 0),
        ('phira_all_types', lambda r: r['mode'] == 'phira'
            and all(r['types'].get(k, 0) > 0 for k in ('tap', 'hold', 'flick', 'drag'))),
    ]
    # Check real lane coverage before ranking; metadata does not encode lanes.
    arrays = {}
    all_four = set()
    for i, row in enumerate(pool):
        if row['split'] != 'train' or row['component'] in used or row['mode'] != 'mania_4k':
            continue
        ci = row['chart_index']
        if ci not in arrays:
            chart = tape['charts'][ci]
            if sha(chart['target_path']) != chart['target_sha256']:
                raise ValueError('Target source changed')
            with np.load(ROOT / chart['target_path'], allow_pickle=False) as archive:
                arrays[ci] = archive['lane'].copy()
        if set(arrays[ci][row['target_start']:row['target_stop']].tolist()) == {0, 1, 2, 3}:
            all_four.add((row['chart_id'], row['request']['start_us']))
    has_four = lambda r: (r['chart_id'], r['request']['start_us']) in all_four
    predicates += [
        ('mania_long_song_hold', lambda r: r['mode'] == 'mania_4k'
            and r['request']['duration_us'] > 448_000_000 and r['types'].get('hold', 0) > 0 and has_four(r)),
        ('mania_four_lanes_hold', lambda r: r['mode'] == 'mania_4k'
            and r['types'].get('hold', 0) > 0 and has_four(r)),
    ]
    selected = []
    for role, predicate in predicates:
        index, row, selection = choose_case(pool, predicate, used, role)
        # Reuse its first scheduled condition view if exposed; otherwise a
        # predeclared full, known training-label view for diagnostic fitting.
        fields = views.get(index, np.flatnonzero(np.asarray(row['controls'])[:, 2]).tolist())
        selected.append(dict(selection, record_index=index, chart_id=row['chart_id'],
            component=row['component'], audio_sha256=row['audio_sha256'], mode=row['mode'],
            stratum=row['stratum'], request=row['request'], tokens=row['tokens'],
            events=row['event_count'], types=row['types'], incoming_holds=row['incoming_holds'],
            outgoing_holds=row['crossing_outgoing_holds'], specified_fields=fields,
            condition_view_source='first actual exposure' if index in views else 'all known train-label fields'))
    return dict(id='D94-probe-selection-v1', complete=True, training_ready=False,
        optimizer_updates=0, new_cloud_compute_calls=0, cases=selected,
        source_sha256={p: sha(p) for p in (TAPE, OLD_TAPE, EXPOSURE,
            'scripts/prepare_local_data_scale_probe.py')},
        selection_independent_of_model_and_download_order=True,
        target_tokens=sum(c['tokens'] for c in selected),
        modes=dict(Counter(c['mode'] for c in selected)),
        qualification='Eight-request memorisation gate only; not a quality or condition response benchmark')


def merge_admission(reports, expected_audio):
    merged = {}
    scale = reports[0]['scales']
    for report in reports:
        if not report['complete'] or report['failures'] or report['scales'] != scale:
            raise ValueError('Feature audit incomplete, failed, or incompatible scales')
        for row in report['records']:
            if not row['passed'] or not row['legacy_prefix_parity']['interior_exact_equal']:
                raise ValueError('A feature row did not pass admission')
            key = row['audio_sha256']
            if key in merged and row != merged[key]:
                raise ValueError('Duplicate audio has conflicting admission evidence')
            merged[key] = row
    if set(merged) != set(expected_audio):
        raise ValueError('Merged feature identity differs from the full frozen tape')
    return dict(id='D94-merged-feature-admission-v1', complete=True, failures=[],
        records=list(merged.values()), scales=scale, expected_songs=len(merged),
        completed_songs=len(merged), training_ready=False, optimizer_updates=0, cloud_compute_calls=0)


def run(select_only=False):
    selection = select_cases()
    if (ROOT / SELECTION).exists():
        if read(SELECTION) != selection:
            raise ValueError('Frozen health selection changed')
    else:
        write_new(SELECTION, selection)
    if select_only:
        print(json.dumps(dict(selection=SELECTION, tokens=selection['target_tokens'],
            cases=[{k: c[k] for k in ('role', 'chart_id', 'tokens', 'record_index')} for c in selection['cases']]), ensure_ascii=False))
        return
    sources = ['reports/d67_local_feature_audit_v1.json', 'reports/d93_feature_audit_v1.json']
    tape = read(TAPE)
    merged = merge_admission([read(p) for p in sources], {c['audio_sha256'] for c in tape['charts']})
    merged['source_sha256'] = {p: sha(p) for p in sources + [TAPE]}
    if (ROOT / FEATURES).exists():
        if read(FEATURES) != merged:
            raise ValueError('Merged feature evidence changed')
    else:
        write_new(FEATURES, merged)
    base = read('bench/rhythm-v2/d88-relative-head-probe-v3.json')
    cases = selection['cases']
    frows = {r['audio_sha256']: r for r in merged['records']}
    source_hashes = {p: sha(p) for p in [TAPE, OLD_TAPE, EXPOSURE, FEATURES, SELECTION,
        base['parent'], 'design/D94_DATA_SCALE_LOCAL_v1.md',
        'scripts/prepare_local_data_scale_probe.py', 'scripts/probe_local_data_scale.py',
        'scripts/probe_local_sequence_likelihood.py', 'scripts/run_local_song_baseline.py',
        'scripts/run_local_event_experiment.py', 'tests/test_local_data_scale_probe.py']}
    source_hashes.update({str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'beatfcn').rglob('*.py')})
    source_hashes[merged['scales']] = sha(merged['scales'])
    for case in cases:
        chart = next(c for c in tape['charts'] if c['chart_id'] == case['chart_id'])
        feature = frows[case['audio_sha256']]
        case['input_shapes'] = padding_shapes(feature['summary']['shape'][0])
        for field in ('target', 'token'):
            p, expected = chart[field + '_path'], chart[field + '_sha256']
            if sha(p) != expected:
                raise ValueError('Fixed batch target bytes changed')
            source_hashes[p] = expected
        source_hashes.update(feature['sha256'])
    for p, expected in source_hashes.items():
        if sha(p) != expected:
            raise ValueError('Fixed batch feature/source changed: ' + p)
    plan = dict(id='D94-new-data-probe-v1', tape=TAPE, features=FEATURES,
        parent=base['parent'], config=base['config'], learning_rate=.001,
        steps=100, microbatch_requests=1, effective_batch_requests=8, cases=cases,
        max_wall_s=900, max_request_mean_ce=.05, generation_token_cap=4096,
        padding_parity_atol=.001, padding_parity_threshold_source='Existing D68 real MPS shape parity gate',
        output='reports/d94_new_data_probe_v1.json', checkpoint='artifacts/d94-probe/diagnostic-step100.pt',
        source_sha256=source_hashes, quality_promotion_allowed=False, auto_resume=False,
        regularization=dict(dropout=0, weight_decay=0, augmentation=False,
                            soft_targets=False, auxiliary_losses=False),
        scope='New-data implementation health; weights never used for the independent-song quality run',
        fit_gate='Every request CE<=.05, correct BOS, exact full free-generated tokens and events; no pruning')
    write_new(PLAN, plan)
    print(json.dumps(dict(plan=PLAN, feature_songs=len(merged['records']), cases=len(cases), training_started=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--select-only', action='store_true')
    run(parser.parse_args().select_only)
