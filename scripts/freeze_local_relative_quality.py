"""Freeze D88's one-variable comparison only after real runtime health passes."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT/'bench/rhythm-v2/d88-relative-head-quality-v1.json'
    if output.exists():
        raise FileExistsError('Do not overwrite an existing frozen experiment')
    control_path = 'bench/rhythm-v2/d87-sequence-control-v1.json'
    control = json.loads((ROOT/control_path).read_text())
    health_path = 'reports/d88_relative_head_probe_v3.json'
    health = json.loads((ROOT/health_path).read_text())
    health_plan_path = 'bench/rhythm-v2/d88-relative-head-probe-v3.json'
    health_plan = json.loads((ROOT/health_plan_path).read_text())
    if (not health['complete'] or not health['passed'] or health.get('error')
            or health['optimizer_updates'] != 100 or len(health['cache_proofs']) != 16
            or health['plan_sha256'] != sha(ROOT/health_plan_path)):
        raise ValueError('Real100-step and complete CPU/MPS cache/tensor health required')
    for path, expected in health_plan['source_sha256'].items():
        if sha(ROOT/path) != expected:
            raise ValueError('Passed health source changed: '+path)
    for path, expected in control['inputs_sha256'].items():
        if sha(ROOT/path) != expected:
            raise ValueError('Original control input changed: '+path)
    archived = 'inputs/code_baselines/d87-closed-v1/scripts/run_local_sequence_quality.py'
    retired = 'scripts/run_local_sequence_quality.py'
    for path, expected in control['code_sha256'].items():
        if sha(ROOT/(archived if path == retired else path)) != expected:
            raise ValueError('Original shared control source changed: '+path)
    same = ('config', 'learning_rate', 'input_shapes', 'tape', 'features')
    if any(control[k] != health_plan[k] for k in same):
        raise ValueError('Health uses another model or input contract')
    if health_plan['parent'] != control['initialize_from']['checkpoint']:
        raise ValueError('Health starts from another parent')
    disposition = json.loads((ROOT/'reports/d87_candidate_disposition_v1.json').read_text())
    if disposition['quality_passed'] or not disposition['rollback']['production_model_unchanged']:
        raise ValueError('Require completed D87 closeout before changing dimension')
    plan = deepcopy(control)
    plan.pop('normalizers')
    additions = ['design/D88_RELATIVE_HEAD_QUALITY_v1.md', control_path, health_path,
        health_plan_path, 'reports/d88_padding_diagnosis_v1.json',
        'reports/d88_relative_fidelity_v1.json', 'reports/d87_candidate_disposition_v1.json',
        'reports/d87_sequence_control_v1.json', 'reports/d87_control_review_1000_v1.json',
        'reports/d87_control_assistant_review_1000_v1.json', 'reports/d87_control_stage_1000_v1.json']
    plan['inputs_sha256'].update({p:sha(ROOT/p) for p in additions})
    code = {p:h for p,h in control['code_sha256'].items() if p != retired}
    code[archived] = control['code_sha256'][retired]
    for path in ['beatfcn/model/relative_event_codec.py', 'beatfcn/model/relative_event_generate.py',
                 'beatfcn/train/relative_request_panel.py', 'scripts/run_local_relative_quality.py',
                 'scripts/freeze_local_relative_quality.py', 'scripts/audit_local_relative_stage.py',
                 'tests/test_relative_event_codec.py', 'tests/test_relative_event_runtime.py',
                 'scripts/probe_relative_head_runtime.py']:
        code[path] = sha(ROOT/path)
    plan.update(id='D88-relative-head-quality-1000-v1', representation='relative_head_v1',
        loss_mode='request_mean', health=health_path, code_sha256=code,
        output='reports/d88_relative_head_quality_v1.json',
        checkpoint_dir='artifacts/d88-relative-head-quality-v1/checkpoints',
        generation_output='reports/d88_relative_head_generation_v1.json',
        generation_dir='artifacts/d88-relative-head-quality-v1/generation-1000',
        source_snapshot='inputs/code_baselines/d88-relative-quality-v1',
        stage_report_pattern='reports/d88_relative_stage_{step}_v1.json',
        hypothesis='Ordinary head coordinates relative to previous completed group improve interval learning without changing events, model, request-mean objective or exposure.',
        baseline='Immutable D87 request-mean control1000; macro F1 .0760975265482696, short tap fraction .37971684649197923, rest mean .75.',
        correctness_revision='Same original event model, common step2, physical microbatch1/shapes; relative teacher/state adapter only. Recomputed step2 teacher CE kept separate from inherited absolute history.',
        failure_policy='250/500 frozen8 stage and early-stop gates unchanged;1000 actual20-song review. No automatic extension beyond1000 or cloud scaling.',
        previous_dimension_failures=disposition['failure_counters_after'],
        quality_promotion_allowed=False,
        control_identity=dict(plan=control_path, plan_sha256=sha(ROOT/control_path),
            report=control['output'], report_sha256=sha(ROOT/control['output']),
            archived_runner=archived, archived_runner_sha256=sha(ROOT/archived)))
    for k in ['exposure', 'evaluation20', 'steps', 'seed', 'checkpoints', 'checkpoint_every',
              'generation_batch_size', 'generation_token_cap', 'input_shapes', 'quality_gates',
              'physical_microbatch_requests', 'initialize_from']:
        if plan[k] != control[k]:
            raise ValueError('Accidental second intervention: '+k)
    snapshot = ROOT/plan['source_snapshot']
    snapshot.mkdir(parents=True, exist_ok=False)
    for path in code:
        dest = snapshot/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/path, dest)
    write_json(output, plan)
    receipt = dict(id='D88-quality-preparation-v1', complete=True, optimizer_updates=0,
        new_cloud_compute_calls=0, plan=str(output.relative_to(ROOT)), plan_sha256=sha(output),
        health=health_path, health_sha256=sha(ROOT/health_path),
        independent_quality_winner=None, fixed_core_and_data_verified=True,
        single_intervention='relative ordinary head coordinates and corresponding decoder query/state',
        quality_gates=plan['quality_gates'])
    write_json(ROOT/'reports/d88_quality_preparation_v1.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
