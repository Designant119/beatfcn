"""Bind D94 execution to completed new-data health and the historical control.

Creates immutable local plans only. No training, inference, or cloud calls.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from scripts.prepare_local_data_scale_probe import FEATURES, PLAN as HEALTH_PLAN, TAPE, EXPOSURE
from scripts.prepare_local_expansion_exposure import records
from scripts.run_local_song_baseline import sha

ROOT = Path(__file__).resolve().parents[1]
QUALITY_PLAN = 'bench/rhythm-v2/d94-data-scale-quality-v1.json'


def read(path):
    return json.loads((ROOT / path).read_text())


def write_new(path, value):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def validate_health(health, health_plan, identity):
    if (health.get('id') != 'D94-new-data-probe-v1' or not health.get('complete')
            or not health.get('passed') or health.get('optimizer_updates') != 100
            or health.get('plan_sha256') != identity or not health.get('parent_unchanged')):
        raise ValueError('The actual D94 new-data 100-step health gate has not passed')
    cases = health_plan['cases']
    generated = health.get('generation', [])
    if (len(cases) != 8 or len(health.get('final', [])) != 8 or len(generated) != 8
            or {g['record_index'] for g in generated} != {c['record_index'] for c in cases}
            or not all(g.get('complete') and g.get('exact_tokens') and g.get('exact_events') for g in generated)
            or not all(v['request_mean_ce'] <= health_plan['max_request_mean_ce'] and v['bos_correct']
                       for v in health['final'])
            or not health.get('padding_proofs') or not all(v['passed'] for v in health['padding_proofs'])):
        raise ValueError('Per-request fitting or long-song padding evidence is incomplete')


def run():
    if (ROOT / QUALITY_PLAN).exists():
        raise FileExistsError('Quality contract already frozen; inspect it instead of rebuilding')
    health_plan = read(HEALTH_PLAN)
    health = read(health_plan['output'])
    validate_health(health, health_plan, sha(ROOT / HEALTH_PLAN))
    for p, expected in health_plan['source_sha256'].items():
        if sha(ROOT / p) != expected:
            raise ValueError('Healthy implementation/data changed: ' + p)
    features = read(FEATURES)
    if not features['complete'] or features['failures']:
        raise ValueError('Full feature admission required')
    prep = read('reports/d94_exposure_preparation_v1.json')
    if not prep['complete'] or not prep['baseline_full_1000_step_replay_exact'] or prep['split_leakage']:
        raise ValueError('Fixed exposure and split evidence is incomplete')
    old_plan_path = 'bench/rhythm-v2/d87-sequence-control-v1.json'
    control = read(old_plan_path)
    plan = deepcopy(control)
    tape, old_tape = read(TAPE), read(control['tape'])
    new_pool, old_pool = records(tape['charts']), records(old_tape['charts'])
    if new_pool[:len(old_pool)] != old_pool:
        raise ValueError('Original train/validation request indices changed')
    exposure, old_exposure = read(EXPOSURE), read(control['exposure'])
    if exposure['steps'][:2] != old_exposure['steps'][:2] or len(exposure['steps']) != 1000:
        raise ValueError('Common parent or total exposure changed')
    if not set(r['audio_sha256'] for r in new_pool) <= {r['audio_sha256'] for r in features['records']}:
        raise ValueError('Quality tape includes unadmitted audio')
    # Keep model/codec/generator mechanics exactly as in the matched control.
    invariant = ['beatfcn/model/full_event_model.py', 'beatfcn/model/full_event_codec.py',
        'beatfcn/model/full_event_generate.py', 'beatfcn/model/local_song_rollout.py',
        'beatfcn/train/local_features.py', 'beatfcn/train/local_request_panel.py',
        'beatfcn/train/local_shapes.py', 'beatfcn/train/event_conditions.py',
        'beatfcn/train/event_reference.py', 'beatfcn/train/local_panel.py']
    for path in invariant:
        if sha(ROOT / path) != control['code_sha256'][path]:
            raise ValueError('A non-data mechanism differs from the matched control: ' + path)
    stage = read(control['stage_protocol'])
    stage.update(id='D94-stage-panel-v1', early_stop=dict(
        at_step=None, evidence='D89: control empty at500 but nonempty at1000; no automatic empty-at500 rejection',
        immediate_stop=['nonfinite values', 'invalid target or permission leakage', 'resource limit']),
        no_quality_promotion_from_small_panel=True)
    stage_path = 'bench/rhythm-v2/d94-stage-protocol-v1.json'
    stage_text = json.dumps(stage, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    stage_digest = hashlib.sha256(stage_text.encode()).hexdigest()
    initial = control['initialize_from']
    initial_plan = read(initial['plan'])
    source_inputs = dict(initial_plan['inputs_sha256'])
    inputs = [TAPE, FEATURES, EXPOSURE, HEALTH_PLAN, health_plan['output'],
        'reports/d94_exposure_preparation_v1.json', 'reports/d94_probe_selection_v1.json',
        'design/D94_DATA_SCALE_LOCAL_v1.md', 'bench/rhythm-v2/fullscale-data-contract-v4.json',
        old_plan_path, control['output'], control['generation_output'],
        'reports/d87_control_review_1000_v1.json', 'reports/d87_control_assistant_review_1000_v1.json',
        'reports/d87_control_stage_1000_v1.json',
        control['evaluation20'], control['tape'], control['features'], control['exposure'],
        stage_path, 'bench/rhythm-v2/d87-condition-protocol-v1.json',
        'bench/rhythm-v2/d87-audio-protocol-v1.json', *initial.values()]
    for p in inputs:
        source_inputs[p] = stage_digest if p == stage_path else sha(ROOT / p)
    control_generation = read(control['generation_output'])
    source_inputs[control_generation['checkpoint']] = sha(ROOT / control_generation['checkpoint'])
    if source_inputs[control_generation['checkpoint']] != control_generation['checkpoint_sha256']:
        raise ValueError('Historical control checkpoint changed')
    for p in control_generation['song_files'].values():
        source_inputs[p] = sha(ROOT / p)
    reviewed = read('bench/rhythm-v2/d87-control-review20-v1.json')['inputs_sha256']
    for path in [control['generation_output'], *control_generation['song_files'].values()]:
        if source_inputs[path] != reviewed[path]:
            raise ValueError('Historical twenty-song control evidence changed: ' + path)
    paths = [str(p.relative_to(ROOT)) for p in (ROOT / 'beatfcn').rglob('*.py')]
    paths += ['scripts/run_local_data_scale.py', 'scripts/prepare_local_data_scale_probe.py',
        'scripts/prepare_local_expansion_exposure.py', 'scripts/freeze_local_data_scale.py',
        'scripts/run_local_song_baseline.py', 'scripts/run_local_event_experiment.py',
        'scripts/audit_local_sequence_stage.py', 'scripts/review_local_song_rollouts.py',
        'scripts/probe_local_data_scale.py', 'tests/test_local_data_scale_probe.py']
    codes = {p: sha(ROOT / p) for p in paths}
    plan.update(id='D94-data-scale-1000-v1', tape=TAPE, features=FEATURES, exposure=EXPOSURE,
        health=health_plan['output'], policy='bench/rhythm-v2/fullscale-data-contract-v4.json',
        output='reports/d94_data_scale_quality_v1.json',
        checkpoint_dir='artifacts/d94-data-scale-v1/checkpoints',
        generation_output='reports/d94_data_scale_generation_v1.json',
        generation_dir='artifacts/d94-data-scale-v1/generation-1000',
        source_snapshot='inputs/code_baselines/d94-data-scale-v1',
        inputs_sha256=source_inputs, code_sha256=codes, stage_protocol=stage_path,
        stage_report_pattern='reports/d94_stage_{step}_v1.json',
        baseline='D87 request-mean matched control: 100songs/1000updates/20full songs; macro F1 .076097527',
        hypothesis='Only expand independently sampled train songs100→333; unchanged loss/model/optimizer/decoding',
        failure_policy='Correctness/resource faults stop. Empty-at500 is not a sufficient rejection. Review20 at1000; no automatic extension/scaling.',
        correctness_revision='Old896 song padding retained; new>896 contexts padded to1024 after actual MPS equivalence; no truncation',
        quality_promotion_allowed=False, normalizers=None,
        local_health_weights_never_resumed=True,
        comparison_scope='Equal4000requests, not equal tokens/FLOPs:1079403 versus1016477',
        actual_exposure=prep['expanded_exposure'],
        prior_quality_failures=dict(objective=4,architecture=3,data_representation=4,training_dynamics=2))
    for key in ('config','seed','learning_rate','steps','validation_indices','evaluation20',
                'generation_batch_size','generation_token_cap','input_shapes','physical_microbatch_requests',
                'initialize_from','loss_mode','quality_gates'):
        if plan[key] != control[key]:
            raise ValueError('Single-variable comparison changed ' + key)
    # Preserve the historical source-input identity gate for the common step2.
    if any(source_inputs[k] != v for k, v in initial_plan['inputs_sha256'].items()):
        raise ValueError('Initial driver inputs changed')
    write_new(stage_path, stage)
    snapshot = ROOT / plan['source_snapshot']
    if snapshot.exists():
        raise FileExistsError(snapshot)
    for path in paths:
        dest = snapshot / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, dest)
    write_new(QUALITY_PLAN, plan)
    write_new('reports/d94_quality_preparation_v1.json', dict(
        id='D94-quality-preparation-v1', complete=True, dispatched=False, optimizer_updates=0,
        cloud_compute_calls=0, plan=QUALITY_PLAN, plan_sha256=sha(ROOT / QUALITY_PLAN),
        copied_control_mechanisms=invariant, unchanged_comparison_fields=True,
        old_request_prefix_exact=True, health=health_plan['output'],
        next_action='Run native MPS common2→4 driver; verify optimizer/RNG/exposure before continuation'))
    print(json.dumps(dict(plan=QUALITY_PLAN, dispatched=False, quality_passed=False)))


if __name__ == '__main__':
    run()
