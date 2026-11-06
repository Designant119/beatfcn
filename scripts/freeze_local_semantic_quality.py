"""Freeze D97 A only after its actual new-objective probe has passed."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def main():
    target = ROOT/'bench/rhythm-v2/d97-semantic-quality-v1.json'
    if target.exists():
        raise FileExistsError(target)
    health_path = 'reports/d97_semantic_objective_probe_v1.json'
    health = json.loads((ROOT/health_path).read_text())
    health_plan_path = 'bench/rhythm-v2/d97-semantic-objective-probe-v1.json'
    health_plan = json.loads((ROOT/health_plan_path).read_text())
    if (not health.get('passed') or not health.get('complete') or health.get('optimizer_updates') != 100
            or health['plan_sha256'] != sha(health_plan_path)):
        raise ValueError('The actual new-objective 100-step gate has not passed')
    for p, h in health_plan['source_sha256'].items():
        if sha(p) != h:
            raise ValueError('Probe source changed after validation: '+p)
    original = 'bench/rhythm-v2/d94-data-scale-quality-v1.json'
    plan = json.loads((ROOT/original).read_text())
    for p, h in plan['inputs_sha256'].items():
        if sha(p) != h:
            raise ValueError('Matched baseline input changed: '+p)
    allowed_history = {'beatfcn/audio/mert2.py', 'scripts/run_local_data_scale.py'}
    historical_changes = []
    for p, h in plan['code_sha256'].items():
        actual = sha(p)
        if actual != h:
            if p not in allowed_history:
                raise ValueError('Unexpected baseline implementation change: '+p)
            historical_changes.append(dict(path=p, baseline_sha256=h, current_sha256=actual,
                reason='D96 unused MERT adapter resource fix' if 'mert2' in p else 'D94 failed runner is closed; separate D97 runner used'))
    plan.update(id='D97-semantic-objective-1000-v1', health=health_path,
        loss_mode='semantic_six_family_mean', normalizers=None,
        output='reports/d97_semantic_quality_v1.json',
        checkpoint_dir='artifacts/d97-semantic-v1/checkpoints',
        generation_output='reports/d97_semantic_generation_v1.json',
        generation_dir='artifacts/d97-semantic-v1/generation-1000',
        source_snapshot='inputs/code_baselines/d97-semantic-v1',
        stage_report_pattern='reports/d97_stage_{step}_v1.json',
        baseline='D94 exact matching2.37M/333songs/4000requests/1079403tokens; macroF1 .0613482128',
        hypothesis='Semantic decision-family likelihood alone raises heldout generation beyond the original full quality gates; same targets, exposure, audio, model, optimizer, decoding.',
        comparison_scope='Same1079403base tokens and4000requests; only supervised weights change after commonstep2. No conditionRL or reference training.',
        prior_quality_failures=dict(objective=4, architecture=4, data_representation=5, training_dynamics=2),
        historical_code_changes=historical_changes, deadline_utc='2026-10-26T18:02:55Z',
        deadline_unix=datetime.fromisoformat('2026-10-26T18:02:55+00:00').timestamp())
    for p in [original, health_path, health_plan_path, 'reports/d97_condition_outcome_audit_v1.json',
              'design/CONDITION_OUTCOME_SPRINT_D97.md', 'reports/d94_data_scale_quality_v1.json',
              'reports/d94_data_scale_generation_v1.json', 'reports/d94_candidate_disposition_v1.json']:
        plan['inputs_sha256'][p] = sha(p)
    code = dict(plan['code_sha256'])
    del code['scripts/run_local_data_scale.py']
    for p in ['scripts/run_local_semantic_objective.py', 'scripts/freeze_local_semantic_quality.py',
              'beatfcn/train/semantic_decision_loss.py', 'tests/test_semantic_decision_loss.py']:
        code[p] = sha(p)
    plan['code_sha256'] = {p: sha(p) for p in code}
    snapshot = ROOT/plan['source_snapshot']
    snapshot.mkdir(exist_ok=False)
    for p in plan['code_sha256']:
        dst = snapshot/p;dst.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(ROOT/p, dst)
    with target.open('x') as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(plan=str(target.relative_to(ROOT)), sha256=sha(target.relative_to(ROOT)),
                         quality_dispatched=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
