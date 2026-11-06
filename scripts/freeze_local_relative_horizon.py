"""Freeze the evidence-based500-to1000 continuation, preserving old verdicts."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

from scripts.run_local_song_baseline import sha, write_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    path=ROOT/'bench/rhythm-v2/d90-relative-horizon-v1.json'
    if path.exists():raise FileExistsError(path)
    parent_plan='bench/rhythm-v2/d88-relative-head-quality-v1.json'
    parent=json.loads((ROOT/parent_plan).read_text())
    state=json.loads((ROOT/parent['output']).read_text())
    disposition=json.loads((ROOT/'reports/d88_candidate_disposition_v1.json').read_text())
    audit=json.loads((ROOT/'reports/d89_exposure_curve_audit_v1.json').read_text())
    if (state['updates']!=500 or state.get('error') or not disposition['early_stop_triggered']
            or not audit['complete'] or audit.get('error') or len(audit['rows'])!=387):
        raise ValueError('Require closed500 and completed new evidence audit')
    if state['checkpoint_sha256']!=sha(ROOT/state['checkpoint']):raise ValueError('Parent checkpoint changed')
    plan=deepcopy(parent)
    plan.update(id='D90-relative-matched-horizon-1000-v1',dimension='training_dynamics',
        hypothesis='Only extend original relative trajectory500to1000 after matched control proved500all-END can recover. Same final quality gates; noextra hyperparameters.',
        output='reports/d90_relative_horizon_v1.json',checkpoint_dir='artifacts/d90-relative-horizon-v1/checkpoints',
        generation_output='reports/d90_relative_generation_v1.json',generation_dir='artifacts/d90-relative-horizon-v1/generation-1000',
        source_snapshot='inputs/code_baselines/d90-relative-horizon-v1',
        stage_report_pattern='reports/d90_relative_stage_{step}_v1.json',checkpoints=[1000],
        max_train_wall_s=900,max_generation_wall_s=1800,
        failure_policy='No500all-END rule: controlcounterexample falsified itsselectivity. Stop correctness/resource failure; final1000all20quality gates unchanged. No2000extension.',
        initialize_from=dict(checkpoint=state['checkpoint'],checkpoint_sha256=state['checkpoint_sha256'],
            report=parent['output'],plan=parent_plan),
        timing_scope='D90only500additionalupdates; D88parentcost andfailure remainseparate.',
        runtime_health_reused='Exactunchanged representation/model/loss/data fromD88; diagnostic weightsnotused.')
    for p in list(plan['code_sha256']):
        if p=='scripts/run_local_relative_quality.py':
            archived='inputs/code_baselines/d88-closed-v1/'+p
            expected=plan['code_sha256'].pop(p)
            if sha(ROOT/archived)!=expected:raise ValueError('Oldrunner archive differs')
            plan['code_sha256'][archived]=expected
    for p in ['scripts/run_local_relative_horizon.py','scripts/freeze_local_relative_horizon.py',
              'design/D90_MATCHED_HORIZON_v1.md']:
        plan['code_sha256'][p]=sha(ROOT/p)
    for p in [parent_plan,parent['output'],state['checkpoint'],'reports/d88_candidate_disposition_v1.json',
              'reports/d89_exposure_curve_audit_v1.json']:
        plan['inputs_sha256'][p]=sha(ROOT/p)
    for p,h in {**plan['inputs_sha256'],**plan['code_sha256']}.items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen dependency changed: '+p)
    control=json.loads((ROOT/'bench/rhythm-v2/d87-sequence-control-v1.json').read_text())
    assert plan['quality_gates']==control['quality_gates']
    for k in ['tape','features','exposure','seed','learning_rate','steps','config','input_shapes','evaluation20','validation_indices','physical_microbatch_requests']:
        assert plan[k]==parent[k],k
    folder=ROOT/plan['source_snapshot']
    if folder.exists():raise FileExistsError(folder)
    for p in plan['code_sha256']:
        dst=folder/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dst)
    write_json(path,plan)
    write_json(ROOT/'reports/d90_preparation_v1.json',dict(complete=True,plan_sha256=sha(path),
        parent_updates=500,additional_updates=500,model_data_loss_optimizer_unchanged=True,
        final_gates_unchanged=True,new_cloud_compute_calls=0,health_reused=True,driver_probe_pending=True))
    print(json.dumps(dict(plan=str(path.relative_to(ROOT)),sha256=sha(path))))


if __name__=='__main__':main()
