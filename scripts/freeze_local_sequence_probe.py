"""Freeze D86's train-only constants and real diagnostic requests before running."""
from collections import defaultdict
import json
from pathlib import Path

from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT/'bench/rhythm-v2/d86-sequence-likelihood-probe-v2.json'
    weights_output = ROOT/'reports/d86_sequence_weight_contract_v2.json'
    if output.exists() or weights_output.exists():
        raise FileExistsError('Do not overwrite a frozen experiment')
    audit_path = 'reports/d85_decision_support_v1.json'
    exposure_path = 'inputs/d67_local_requests/initial100/learning-exposure-1000-v1.json'
    audit = json.loads((ROOT/audit_path).read_text())
    exposure = json.loads((ROOT/exposure_path).read_text())
    records = {r['record_index']:r for r in audit['records']}
    constants = {m:audit['by_mode_and_view'][m]['mean_tokens'] for m in ('phira', 'mania_4k')}
    first_masks = {}
    totals = defaultdict(lambda:dict(request_exposures=0, original_token_coefficient_mass=0.,
        candidate_token_coefficient_mass=0., original_bos_end=0., original_bos_new=0.,
        candidate_bos_end=0., candidate_bos_new=0.))
    for step in exposure['steps']:
        for index, specified in zip(step['indices'], step['specified_condition_fields'], strict=True):
            first_masks.setdefault(index, specified)
            row = records[index]
            keys = [('mode', row['mode']), ('song', row['mode']+'/'+row['component']),
                    ('stratum', row['mode']+'/'+row['stratum']),
                    ('condition_view', row['mode']+'/'+('specified' if any(row['control_known'][f] for f in specified) else 'unknown'))]
            # Type memberships overlap. These are exposure/coefficient totals of
            # requests containing each type, not a probability distribution.
            keys += [('request_contains_type', row['mode']+'/'+kind) for kind in row['types']]
            for category, key in keys:
                target = totals[category+'::'+key]
                target['request_exposures'] += 1
                target['original_token_coefficient_mass'] += 1
                target['candidate_token_coefficient_mass'] += row['tokens']/constants[row['mode']]
                label = row['bos_label'].lower()
                if row['bos_support_size'] > 1:
                    target['original_bos_'+label] += 1/row['tokens']
                    target['candidate_bos_'+label] += 1/constants[row['mode']]
    for value in totals.values():
        value['coefficient_mass_ratio'] = value['candidate_token_coefficient_mass']/value['original_token_coefficient_mass']
    for mode in constants:
        value = totals['mode::'+mode]
        if abs(value['coefficient_mass_ratio']-1) > 1e-12:
            raise ValueError('Frozen mode coefficient mass is not preserved')
    weights = dict(id='D86-sequence-weight-contract-v2', complete=True, optimizer_updates=0,
        constants=constants, source_sha256={p:sha(ROOT/p) for p in (audit_path, exposure_path)},
        groups=dict(totals),
        limits='Coefficient masses are not gradients. Song/length weights change; type memberships overlap. All labels, forced tokens and valid empty windows remain.')
    write_json(weights_output, weights)
    criteria = [
        ('phira_empty', lambda r:r['mode']=='phira' and not r['events']),
        ('phira_tap', lambda r:r['mode']=='phira' and set(r['types'])=={'tap'}),
        ('phira_hold', lambda r:r['mode']=='phira' and 'hold' in r['types']),
        ('phira_flick', lambda r:r['mode']=='phira' and 'flick' in r['types']),
        ('phira_drag', lambda r:r['mode']=='phira' and 'drag' in r['types']),
        ('phira_all4', lambda r:r['mode']=='phira' and len(r['types'])==4),
        ('mania_empty', lambda r:r['mode']=='mania_4k' and not r['events']),
        ('mania_hold', lambda r:r['mode']=='mania_4k' and 'hold' in r['types'])]
    cases, used = [], set()
    for role, predicate in criteria:
        candidates = sorted([r for r in records.values() if predicate(r) and r['record_index'] not in used],
            key=lambda r:(r['tokens'], r['record_index']))
        selected = candidates[(len(candidates)-1)//4]
        used.add(selected['record_index'])
        cases.append(dict(selected, role=role, specified_fields=first_masks[selected['record_index']],
            selection='floor lower-quartile rank by (token length,index), from exposed real train requests, without duplicates'))
    base = json.loads((ROOT/'bench/rhythm-v2/d68-local-baseline-v3.json').read_text())
    parent = 'artifacts/d68-local-baseline/checkpoints/step-0002.pt'
    source_paths = [audit_path, exposure_path, str(weights_output.relative_to(ROOT)), parent,
        'bench/rhythm-v2/d68-local-baseline-v3.json', 'reports/d85_sequence_supervision_research.md',
        'reports/d67_rating_policy_impact_v1.json', base['tape'], base['features'],
        'scripts/freeze_local_sequence_probe.py', 'scripts/probe_local_sequence_likelihood.py',
        'scripts/run_local_song_baseline.py', 'scripts/run_local_event_experiment.py',
        'beatfcn/train/sequence_likelihood.py', 'beatfcn/train/local_request_panel.py',
        'beatfcn/train/local_audio_audit.py', 'beatfcn/train/local_features.py',
        'beatfcn/train/local_shapes.py', 'beatfcn/train/local_panel.py',
        'beatfcn/train/event_reference.py', 'beatfcn/model/full_event_model.py',
        'beatfcn/model/full_event_codec.py', 'beatfcn/model/full_event_generate.py',
        'beatfcn/model/event_sequence.py', 'beatfcn/model/event_canvas.py',
        'tests/test_sequence_likelihood.py', 'tests/test_local_decision_support.py']
    sources = {path:sha(ROOT/path) for path in source_paths}
    tape = json.loads((ROOT/base['tape']).read_text())
    chart_ids = {r['chart_id'] for r in cases}
    audio = set()
    for chart in tape['charts']:
        if chart['chart_id'] in chart_ids:
            audio.add(chart['audio_sha256'])
            sources[chart['token_path']] = chart['token_sha256']
            sources[chart['target_path']] = chart['target_sha256']
    features = json.loads((ROOT/base['features']).read_text())
    for record in features['records']:
        if record['audio_sha256'] in audio:
            sources.update(record['sha256'])
    plan = dict(id='D86-sequence-likelihood-real-fixed-batch-v2', dimension='Objective',
        hypothesis='Changing only 1/N_i to frozen 1/C_mode removes request-length dependence of BOS supervision without dropping rests; test exact implementation and fit before quality claims.',
        normalizers=dict(mania_mean_tokens=constants['mania_4k'], phira_mean_tokens=constants['phira']),
        parent=parent, config=base['config'], learning_rate=base['learning_rate'],
        input_shapes=base['input_shapes'], tape=base['tape'], features=base['features'], cases=cases,
        steps=100, arms=['request_mean_control', 'constant_mode_candidate'],
        physical_microbatch=1, effective_requests=8, max_wall_s=900,
        max_request_mean_ce=.05, generation_token_cap=1024,
        fit_gate='Every candidate request CE <=.05, correct first decision, and exact complete free token sequence; control measured on same batch/initial state. No health claim from training objective magnitude.',
        baseline_scope='D68 model with common step2; no D83 Key, affine normalizer or interaction. No heldout quality attribution from this probe.',
        output='reports/d86_sequence_likelihood_probe_v1.json',
        checkpoint_dir='artifacts/d86-sequence-likelihood-probe-v1',
        source_sha256=sources, quality_promotion_allowed=False, auto_resume=False,
        failure_counts_unchanged_before_dispatch=dict(objective=3, architecture=3, data_representation=3, training_dynamics=1),
        new_cloud_compute_calls=0, budget_reservations_unchanged=True)
    write_json(output, plan)
    print(json.dumps(dict(plan=str(output), sources=len(sources), cases=[(r['role'],r['record_index'],r['tokens']) for r in cases])))


if __name__ == '__main__':
    main()
