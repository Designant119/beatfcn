"""Freeze a width-only D95 quality comparison after real new-data health."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import shutil

from scripts.prepare_local_capacity import width_only
from scripts.prepare_local_data_scale_probe import read, sha, write_new

ROOT = Path(__file__).resolve().parents[1]
BASE = 'bench/rhythm-v2/d94-data-scale-quality-v1.json'
PLAN = 'bench/rhythm-v2/d95-capacity-quality-v1.json'
HEALTH = 'bench/rhythm-v2/d95-new-data-probe-v1.json'


def validate_health(health, plan, identity):
    if (not health.get('complete') or not health.get('passed')
            or health.get('optimizer_updates') != 100 or health.get('plan_sha256') != identity
            or health.get('id') != plan['id'] or not health.get('parent_unchanged')):
        raise ValueError('The width256 new-data health gate has not passed')
    expected = [c['record_index'] for c in plan['cases']]
    if (len(expected) != 8 or len(health.get('final', [])) != 8
            or [g['record_index'] for g in health.get('generation', [])] != expected
            or not all(g['complete'] and g['exact_tokens'] and g['exact_events'] for g in health['generation'])
            or not all(v['request_mean_ce'] <= plan['max_request_mean_ce'] and v['bos_correct'] for v in health['final'])
            or not health.get('padding_proofs') or not all(v['passed'] for v in health['padding_proofs'])):
        raise ValueError('Full target fitting or long-song equivalence evidence is incomplete')


def execution_ast(source):
    """Extract the unchanged training, padding, checkpoint, eval and generation code."""
    tree = ast.parse(source)
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    names = ('batch', 'save', 'checkpoint', 'evaluate')
    result = {n.name: ast.dump(n, include_attributes=False) for n in run.body
              if isinstance(n, ast.FunctionDef) and n.name in names}
    result['training_and_generation'] = ast.dump(next(n for n in run.body if isinstance(n, ast.Try)), include_attributes=False)
    if set(result) != set(names) | {'training_and_generation'}:
        raise ValueError('Missing a registered execution mechanism')
    return result


def run():
    if (ROOT / PLAN).exists():
        raise FileExistsError('Quality plan already frozen')
    base, hp = read(BASE), read(HEALTH)
    health = read(hp['output'])
    validate_health(health, hp, sha(HEALTH))
    for path, expected in hp['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Healthy implementation or input changed: ' + path)
    initialization = 'reports/d95_initialization_v2.json'
    ip = 'bench/rhythm-v2/d95-initialization-v2.json'
    startup = read(initialization)
    if not startup['passed'] or startup['checkpoint'] != hp['parent']:
        raise ValueError('Quality initialization differs from healthy width256 architecture')
    width_only(base['config'], hp['config'])
    old_script = ROOT / base['source_snapshot'] / 'scripts/run_local_data_scale.py'
    if sha(old_script) != base['code_sha256']['scripts/run_local_data_scale.py']:
        raise ValueError('Original execution archive changed')
    runner = 'scripts/run_local_capacity.py'
    if execution_ast(old_script.read_text()) != execution_ast((ROOT / runner).read_text()):
        raise ValueError('A non-capacity training/evaluation mechanism changed')
    for path, expected in base['code_sha256'].items():
        if path.startswith('beatfcn/') and sha(path) != expected:
            raise ValueError('Production model/data/codec implementation changed: ' + path)
    generation = read(base['generation_output'])
    if not generation['execution_complete'] or generation['completed_songs'] != 20:
        raise ValueError('Completed20 control outputs are missing')
    reviewed = read('bench/rhythm-v2/d94-review20-v1.json')
    inputs = deepcopy(base['inputs_sha256'])
    paths = [BASE, HEALTH, hp['output'], initialization, ip, startup['checkpoint'],
        'design/LOCAL_CAPACITY_D95.md', base['output'], base['generation_output'],
        generation['checkpoint'], 'reports/d94_review_1000_v1.json',
        'reports/d94_assistant_review_1000_v1.json', 'reports/d94_candidate_disposition_v1.json',
        'reports/d94_sequence_condition_response_v1.json', 'reports/d94_exposure_coverage_v1.json']
    paths += list(generation['song_files'].values())
    for path in paths:
        inputs[path] = sha(path)
    for path in [base['generation_output'], *generation['song_files'].values()]:
        if inputs[path] != reviewed['inputs_sha256'][path]:
            raise ValueError('Control20 outputs differ from reviewed originals')
    code = {p: h for p, h in base['code_sha256'].items() if p != 'scripts/run_local_data_scale.py'}
    for path in [runner, 'scripts/prepare_local_capacity.py', 'scripts/freeze_local_capacity_health.py',
                 'scripts/freeze_local_capacity_quality.py', 'tests/test_local_capacity_startup.py',
                 'tests/test_local_capacity_quality.py']:
        code[path] = sha(path)
    for path, expected in {**inputs, **code}.items():
        if sha(path) != expected:
            raise ValueError('Frozen comparator input or source changed: ' + path)
    plan = deepcopy(base)
    plan.update(id='D95-capacity-1000-v1', config=hp['config'], health=hp['output'],
        inputs_sha256=inputs, code_sha256=code,
        initialize_from=dict(checkpoint=startup['checkpoint'], report=initialization, plan=ip),
        output='reports/d95_capacity_quality_v1.json', checkpoint_dir='artifacts/d95-capacity-v1/checkpoints',
        generation_output='reports/d95_capacity_generation_v1.json', generation_dir='artifacts/d95-capacity-v1/generation-1000',
        source_snapshot='inputs/code_baselines/d95-capacity-v1', stage_report_pattern='reports/d95_stage_{step}_v1.json',
        baseline='D94 width128:333songs/1000updates/20full songs; macro F1 .0613482128',
        hypothesis='Only width128/FFN384 to width256/FFN768; same data, exposure, audio, objective and optimizer',
        comparison_scope='Identical4000requests and1079403valid tokens; compute and wall time differ with width',
        prior_quality_failures=dict(objective=4, architecture=3, data_representation=5, training_dynamics=2))
    invariant = ('tape', 'features', 'exposure', 'seed', 'learning_rate', 'steps', 'cpu_threads',
        'checkpoints', 'validation_indices', 'evaluation20', 'generation_batch_size', 'generation_token_cap',
        'max_train_wall_s', 'max_generation_wall_s', 'input_shapes', 'physical_microbatch_requests',
        'loss_mode', 'normalizers', 'quality_gates', 'stage_protocol', 'actual_exposure')
    if any(plan[k] != base[k] for k in invariant):
        raise ValueError('The quality comparison changes more than capacity')
    snapshot = ROOT / plan['source_snapshot']
    if snapshot.exists():
        raise FileExistsError(snapshot)
    for path in code:
        dest = snapshot / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, dest)
    write_new(PLAN, plan)
    write_new('reports/d95_quality_preparation_v1.json', dict(id='D95-quality-preparation-v1',
        complete=True, dispatched=False, plan=PLAN, plan_sha256=sha(PLAN),
        quality_passed=False, training_execution_ast_equal=True, invariant_fields=list(invariant),
        health=hp['output'], initialization=initialization, cloud_compute_calls=0,
        next_action='Bounded step2-to4 driver; verify Adam/RNG/exposure before continuing'))
    print(json.dumps(dict(plan=PLAN, dispatched=False, quality_passed=False)))


if __name__ == '__main__':
    run()
