"""Validate a fully-applied health checkpoint, then finish diagnostics only."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import shutil
import subprocess

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.train.managed_mps import ManagedMPS
from beatfcn.train.native_cell_chain import cell_batch,run_members,decode_cells
from beatfcn.train.native_field_replay_batched import replay_fields
from scripts.run_local_evolution import load_data
from scripts.run_local_portfolio_scale import write,sha,weight_digest
from scripts.run_local_typed_canvas import score
from scripts.run_mps_adamw_admission import fresh
from scripts.run_native_cell_chain import closure,audio_variant
from scripts.run_native_cell_recovery import rows_for_health,zero_audio,input_digest


ROOT=Path(__file__).resolve().parents[1];ID='c13-frozen-health-completion-v1'
PLAN=ROOT/f'bench/rhythm-v2/{ID}.json';ART=ROOT/f'artifacts/{ID}'


def prepare():
    if PLAN.exists():raise FileExistsError('Preserve completion registration')
    parent='bench/rhythm-v2/c12-native-cell-recovery-v1.json';p=json.loads((ROOT/parent).read_text())
    checkpoint='artifacts/c12-native-cell-recovery-v1/health-diagnostic-state.pt'
    contract=dict(id=ID,parent_plan=parent,parent_plan_sha256=sha(ROOT/parent),checkpoint=checkpoint,
        checkpoint_sha256='d709783072a65474f63714b2fd807366bd30c9f12c13ea1c292036b5ac1ea5db',
        optimizer_updates=0,inherited_health_updates=100,wall_s=600,runtime='2.14.0',
        runtime_target=p['runtime_target'],runtime_records=p['runtime_records'],
        scope='Checkpointstructure + all5 baseline/prefix/teacher/all12fields/wholezero-shift;nohealthretraining,noquality',
        controls_do_not_gate_effect=True,old_status_unchanged=True,quality_authorized=False,cloud_compute_calls=0,
        source_sha256={f:sha(ROOT/f) for f in closure('scripts/complete_frozen_health.py')})
    if sha(ROOT/checkpoint)!=contract['checkpoint_sha256']:raise ValueError('Checkpoint changed before admission')
    write(PLAN,contract)
    for name in contract['source_sha256']:
        dest=ART/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    shutil.copy2(PLAN,ART/'source/frozen-plan.json')


def structure(model,state,parent,plan):
    errors=[]
    expected=dict(updates=100,complete=False,accumulation_phase='applied',optimizer_step_may_be_partial=False,
        resume_authorized=False,plan_sha256=plan['parent_plan_sha256'])
    for k,v in expected.items():
        if state.get(k)!=v:errors.append('Checkpoint '+k)
    for k in ('model','optimizer','rng_cpu','rng_mps','rng_python'):
        if k not in state:errors.append('Missing '+k)
    params=list(model.parameters());baseline=model.state_dict();weights=state.get('model',{})
    if set(weights)!=set(baseline):errors.append('Model state keys')
    for k,v in weights.items():
        if k not in baseline or v.shape!=baseline[k].shape or v.dtype!=baseline[k].dtype:errors.append('Tensor identity '+k)
        if not bool(torch.isfinite(v).all()):errors.append('Nonfinite '+k)
    opt=state.get('optimizer') or {};groups=opt.get('param_groups',[])
    ids=[p for g in groups for p in g['params']]
    if len(ids)!=len(params) or len(set(ids))!=len(params):errors.append('Optimizer parameter coverage')
    steps=[]
    for ident,parameter in zip(ids,params):
        s=opt.get('state',{}).get(ident,{})
        if not s:errors.append('Missing Adam state');continue
        steps.append(float(s['step']))
        for k in ('exp_avg','exp_avg_sq'):
            if k not in s or s[k].shape!=parameter.shape:errors.append('Adam moment shape')
            elif not bool(torch.isfinite(s[k]).all()):errors.append('Nonfinite Adam moment')
    if steps and any(s!=100 for s in steps):errors.append('Adam steps not100')
    if not steps:errors.append('No Adam steps')
    if sum(p.numel() for p in params)!=10240592:errors.append('Parameter count')
    if parent['model']['dropout']!=0:errors.append('Dropout not0')
    return dict(status='structure_rejected' if errors else 'structure_passed',errors=errors,
        updates=state.get('updates'),parameter_count=sum(p.numel() for p in params),optimizer_parameters=len(ids),
        optimizer_steps_min=min(steps) if steps else None,optimizer_steps_max=max(steps) if steps else None,
        complete=state.get('complete'),accumulation_phase=state.get('accumulation_phase'),expected=expected)


@torch.no_grad()
def worker():
    plan=json.loads(PLAN.read_text());parent=json.loads((ROOT/plan['parent_plan']).read_text())
    report=dict(id=ID,status='running',diagnostic_optimizer_updates=0,inherited_health_updates=100,
                health_weights_used_for_quality=False,quality_authorized=False,cloud_compute_calls=0,rows=[])
    guard=ManagedMPS(ROOT,ART,plan['wall_s'])
    try:
        if torch.__version__!='2.14.0' or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('Runtime orfallback')
        torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5)
        path=ROOT/plan['checkpoint']
        if sha(path)!=plan['checkpoint_sha256']:raise ValueError('Checkpoint SHA')
        state=torch.load(path,map_location='cpu',weights_only=True)
        model=fresh(parent,'cpu')
        audit=structure(model,state,parent,plan);write(ART/'structure.json',audit);report['structure']=audit
        if audit['status']!='structure_passed':report['status']='structure_rejected';return
        model.load_state_dict(state['model'],strict=True);model.to('mps').eval()
        torch.set_rng_state(state['rng_cpu']);torch.mps.set_rng_state(state['rng_mps']);random.setstate(state['rng_python'])
        rng_cpu=torch.get_rng_state().clone();rng_mps=torch.mps.get_rng_state().clone();rng_python=random.getstate()
        before=weight_digest(model);report['weight_digest_before']=before;report['checkpoint_sha256_before']=sha(path)
        del state;guard.clear()
        for name,expected in parent['input_payload_sha256'].items():
            if sha(ROOT/name)!=expected:raise ValueError('Health input cache changed '+name)
            guard.check('input_identity')
        panel,row=load_data();data=rows_for_health(parent,panel,row);deadline=guard.start+guard.wall_s
        for rr in data:
            x,t=cell_batch(rr,'mps');out=model(x)
            pred=decode_cells(model,x,rr['request'],{k:t[k] for k in ('heads','allowed')},trace=True,output=out,deadline=deadline)
            events=[RhythmEvent(**e) for e in pred['events']];variants={}
            for name,counts in [('true_count_true_prefix',t['counts']),('machine_count_available_true_prefix',out['count'][0].argmax(-1).cpu().numpy())]:
                try:
                    ee,_,_,_=run_members(model,out,x,rr['request'],t,counts,teacher=False,teacher_prefix=True,deadline=deadline)
                    variants[name]=dict(events=[asdict(e) for e in ee],scores=score(ee,rr['targets']),oracle=True)
                except (ValueError,TimeoutError) as exc:variants[name]=dict(failure=str(exc),oracle=True)
            _,parts,teacher,_=run_members(model,out,x,rr['request'],t,t['counts'],teacher=True,trace=True,deadline=deadline)
            z=zero_audio(x);s,coverage=audio_variant(x,x,shift=.733)
            upstream=dict(zero=(model(z),z),shift=(model(s),s))
            field=None
            if pred['complete']:
                field=replay_fields(model,out,x,rr['request'],{k:t[k] for k in ('heads','allowed')},pred,rr['targets'],upstream,health=False,checkpoint=guard.clear)
                field['index']=rr['index'];write(ART/f'fields-{rr["index"]}.json',field)
            controls={}
            for name,changed in [('zero',z),('shift',s)]:
                alternative=decode_cells(model,changed,rr['request'],{k:t[k] for k in ('heads','allowed')},trace=True,deadline=deadline)
                controls[name]=dict(prediction=alternative,scores=score([RhythmEvent(**e) for e in alternative['events']],rr['targets']))
                guard.clear()
            rec=dict(index=rr['index'],view=rr['view'],request=asdict(rr['request']),input_sha256=input_digest(x),
                targets=[asdict(e) for e in rr['targets']],prediction=pred,scores=score(events,rr['targets']),
                count_probability=out['count'][0].softmax(-1).cpu().tolist(),count_target=t['counts'].tolist(),
                oracle_variants=variants,teacher_steps=teacher,teacher_nll={k:float(v) for k,v in parts.items()},
                controls=controls,shift_coverage=coverage,
                field_certificate=field['certificate'] if field else None,
                field_identity_max_diff=field['field_identity_max_diff'] if field else None,
                field_arms=list(field['arms']) if field else [])
            report['rows'].append(rec);write(ART/'report.json',report)
            print('completed',rr['index'],rec['scores']['full_note_f1'],len(events),rec['field_identity_max_diff'],flush=True)
            del x,t,out,z,s,upstream,field,teacher,parts,controls,variants;guard.clear()
        after=weight_digest(model);report['weight_digest_after']=after;report['checkpoint_sha256_after']=sha(path)
        report['rng_unchanged']=bool(torch.equal(rng_cpu,torch.get_rng_state()) and torch.equal(rng_mps,torch.mps.get_rng_state()) and rng_python==random.getstate())
        old=json.loads((ROOT/'artifacts/c12-native-cell-recovery-v1/health.json').read_text());gate=parent['health_gate']
        nonempty=[r for r in report['rows'] if r['targets']];empty=[r for r in report['rows'] if not r['targets']]
        checks=dict(loss=old['curve'][-1]['loss']<=old['curve'][0]['loss']*gate['loss_fraction'],
            mean_full=np.mean([r['scores']['full_note_f1'] for r in nonempty])>=gate['mean_full'],
            minimum_full=min(r['scores']['full_note_f1'] for r in nonempty)>=gate['minimum_full'],
            onset10=np.mean([r['scores']['onset_10ms']['f1'] or 0 for r in nonempty])>=gate['onset10'],
            empty=len(empty)==1 and all(not r['prediction']['events'] for r in empty),
            baseline_complete=len(report['rows'])==5 and all(r['prediction']['complete'] for r in report['rows']),
            all12_fields=all(len(r['field_arms'])==12 and r['field_identity_max_diff']<=1e-4 for r in report['rows']),
            controls_recorded=all(set(r['controls'])=={'zero','shift'} and len(r['oracle_variants'])==2 for r in report['rows']),
            cpu_mps=max(old['cpu_mps'].values())<=1e-4,
            weights_unchanged=before==after and report['checkpoint_sha256_after']==plan['checkpoint_sha256'],rng_unchanged=report['rng_unchanged'])
        report['checks']={k:bool(v) for k,v in checks.items()};report['status']='health_passed' if all(checks.values()) else 'health_failed'
    except BaseException as exc:report.update(status='stopped',failure=dict(type=type(exc).__name__,error=str(exc),phase=guard.phase))
    finally:
        report['telemetry']=guard.close();write(ART/'report.json',report)


def run():
    if (ART/'report.json').exists():raise FileExistsError('No completion rerun')
    plan=json.loads(PLAN.read_text())
    for f,h in {**plan['source_sha256'],**plan['runtime_records']}.items():
        if sha(ROOT/f)!=h:raise ValueError('Registered source/runtime drift '+f)
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/plan['runtime_target'])+os.pathsep+str(ROOT)
    with (ART/'run.log').open('w') as log:subprocess.run([str(ROOT/'.venv/bin/python'),'-u','-m','scripts.complete_frozen_health','worker'],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=plan['wall_s']+30,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','worker']);args=parser.parse_args();globals()[args.action]()
