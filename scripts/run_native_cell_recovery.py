"""C12 complete same-law recovery: cold preflight, health/P2, fresh quality, eval."""
import argparse
from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.sample import match_f1
from beatfcn.train.managed_mps import ManagedMPS
from beatfcn.train.native_cell_chain import cell_row,cell_batch,cell_loss,decode_cells,run_members,SUPPORT
from beatfcn.train.native_field_replay import replay_fields,array
from beatfcn.train.native_quality_metrics import teacher_scores,summarize
from beatfcn.train.multiset_endpoint_diagnostic import endpoint_diagnostic
from scripts.check_experiment_storage import admission,GIB
from scripts.run_c07_count_ancestral import specs
from scripts.run_local_evolution import load_data
from scripts.run_local_portfolio_scale import sha,write,weight_digest
from scripts.run_local_typed_canvas import score
from scripts.run_mps_adamw_admission import fresh
from scripts.run_native_cell_chain import closure,audio_variant,rate
from scripts.run_joint_native_set import audit_tensors


ROOT=Path(__file__).resolve().parents[1];ID='c12-native-cell-recovery-v1'
PLAN=ROOT/f'bench/rhythm-v2/{ID}.json';ART=ROOT/f'artifacts/{ID}'
REPORT=ROOT/'reports/local-integration-v1/native-cell-recovery.json'


def read(path):return json.loads((ROOT/path).read_text())


def publish(report):
    write(REPORT,report)
    path=ROOT/'reports/local-integration-v1/state.json';state=json.loads(path.read_text())
    stage=report.get('active_stage');state.update(status=report['status'],candidate=ID,
        report=str(REPORT.relative_to(ROOT)),training_process_running=report['status'].endswith('_running'),
        progress_file=f'artifacts/{ID}/{stage}.json' if stage else None,
        confirmation_scored=False,new_cloud_compute_calls=0,new_cloud_spend_usd=0)
    write(path,state)


def prepare():
    if PLAN.exists():raise FileExistsError('Preserve frozen trial')
    parent=read('bench/rhythm-v2/c11-native-cell-chain-v1.json')
    admission_plan=read('bench/rhythm-v2/c12-adamw-runtime-admission-v1.json')
    if read('artifacts/c12-adamw-runtime-admission-v1/report.json')['status']!='admission_passed':raise ValueError('Actual8update admission required')
    p=dict(parent)
    for k in ('source_sha256','pretraining_probe_receipts','id'):p.pop(k,None)
    p.update(id=ID,runtime='2.14.0',runtime_target=admission_plan['runtime_target'],runtime_records=admission_plan['runtime_records'],
        initial_state_path=admission_plan['initial_state_path'],initial_state_sha256=admission_plan['initial_state_sha256'],
        initial_weight_sha256=admission_plan['initial_weight_sha256'],memory_fraction=.5,
        preflight_wall_s=400,health_wall_s=1800,quality_wall_s=7200,evaluation_wall_s=2400,process_wall_s=12000,
        resource_prefix_steps=64,resource_prefix_wall_s=1200,free_check_steps=[8,32,64],
        p2_indices=parent['indices']['dev'][::4],confirmation_scored=False,
        objective=parent['objective'],decision=parent['decision'],
        field_schema='Machine baseline complete first;same count/state/query/head/history per arm;conditional endpoint true path follows arm-specific keys,not counterfactual history',
        health_control_effect_gate=False,
        p2_arms=['base','head_mel_zero','head_mel_shift','song_zero','song_shift','upstream_zero','upstream_shift','end_mel_zero','end_mel_shift','end_song_zero','end_song_shift','clock'],
        research_gate=dict(full50=.30,count_ratio=[.5,1.75],null10_gain=.05,positive_components=10,
            shifted10_gain=.03,foreign10_gain=.03,one_non_tap_f1=.15,non_tap_null_gain=.03,
            non_tap_tp=3,non_tap_components=2,proper_gain=.01),
        confirmation_policy='Keep16sealedthisround;dev research signal only,not independent confirmation or scaling',
        failure_state_policy='One diagnostic model+optimizer+RNG+accumulation-phase state,never automatic resume',
        predecessors=['c11-native-cell-chain-v1','c12-mps-runtime-probe-v1','c12-adamw-runtime-admission-v1'],
        shared_failure_modes=['total_process_memory','missing_health_field_controls','empty_free_coverage'],
        mechanism_changes=dict(
            total_process_memory=dict(change='Isolated2.14+perrequestcleanup preservesgrad+ownphysicalwatchdog+freshprefix64',prediction='Actual varied updates remain bounded',discriminating_control='8updateadmission,ongoing0.5stelemetry,prefixmedian/growth'),
            missing_health_field_controls=dict(change='Persist machine tape andbaseline/zero/shift/field replay before health release',prediction='All controls executed withidentity certificate andscope',discriminating_control='Probability/intidentity against untouched originaldecode;effectsreportednothealthqualitygate'),
            empty_free_coverage=dict(change='Actual100health nonempty/hold baseline andperiodicmachinefreepath resources',prediction='Qualify all actual generatedbranches ratherthanemptycall',discriminating_control='HealthF1/type/holdtape coverage plus8/32/64free execution')),
        pretraining_probe_receipts=[dict(status='completed',evidence='artifacts/c12-adamw-runtime-admission-v1/report.json',scope='new_runtime_realupdates'),
            dict(status='completed',evidence='reports/local-integration-v1/c12-field-replay-regression.json',scope='new_field_replay_exact_math_and_multiset_tests')],execution_ready=True)
    p['source_sha256']={f:sha(ROOT/f) for f in closure('scripts/run_native_cell_recovery.py')+['tests/test_native_field_replay.py','tests/test_native_quality_metrics.py']}
    write(PLAN,p)
    for name in p['source_sha256']:
        dest=ART/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    shutil.copy2(PLAN,ART/'source/frozen-plan.json')


def zero_audio(x):
    return {k:torch.zeros_like(v) if k in ('local','mel','song') else v for k,v in x.items()}


def input_digest(x):
    h=hashlib.sha256()
    for k,v in sorted(x.items()):h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(array(v).tobytes())
    return h.hexdigest()


def preflight(plan,panel,row):
    started=time.monotonic();original=read('artifacts/c11-native-cell-chain-v1/data-identity.json')['checks'];checks=[];types=Counter()
    for name,expected in plan['input_payload_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Input payload drift '+name)
    groups={k:{str(panel.records[i]['component']) for i in plan['indices'][k]} for k in ('train64','dev','confirm','empty_dev')}
    audio={k:{panel.records[i]['audio_sha256'] for i in plan['indices'][k]} for k in groups}
    for a,b in (('train64','dev'),('train64','confirm'),('dev','confirm'),('train64','empty_dev')):
        if groups[a]&groups[b] or audio[a]&audio[b]:raise ValueError('Audio/component overlap '+a+' '+b)
    if len(groups['dev'])!=16:raise ValueError('Dev component identity')
    for batch in plan['sample_stream']:
        for s in batch:
            rr=cell_row(panel,row(s['index'],s['view'],s['field_index']));counts,members=SUPPORT.encode(rr['targets'],rr['request'])
            if Counter(SUPPORT.decode(counts,members,rr['request']))!=Counter(rr['targets']):raise ValueError('Multiset support')
            digest=hashlib.sha256(json.dumps([asdict(e) for e in sorted(rr['targets'])],sort_keys=True).encode()).hexdigest()
            if digest!=original[len(checks)]['target_sha256']:raise ValueError('Exposure target identity')
            checks.append(dict(**s,target_sha256=digest,events=len(members)));types.update(e.type for e in rr['targets'])
            if time.monotonic()-started>plan['preflight_wall_s']:raise TimeoutError('Preflight wall')
    result=dict(status='completed',requests=len(checks),events=sum(types.values()),types=dict(types),checks=checks,
        groups={k:dict(windows=len(plan['indices'][k]),components=len(groups[k]),audio_ids=len(audio[k])) for k in groups},
        prior=[types[k]+1 for k in ('tap','hold','flick','drag')],wall_s=time.monotonic()-started,
        confirm_scope='Metadataidentityonly;no confirmation predictions or scoring')
    write(ART/'data-identity.json',result);return result


def rows_for_health(plan,panel,row):
    return [cell_row(panel,row(i,v)) for i,v in list(zip(plan['health_indices'],('draft','single_condition','random_groups','middle')))+[(plan['empty_health_index'],'draft')]]


@torch.no_grad()
def prediction(model,rr,x,t,guard,deadline,output=None):
    guard.check('free_decode')
    result=decode_cells(model,x,rr['request'],{k:t[k] for k in ('heads','allowed')},deadline=deadline,trace=True,output=output)
    guard.check('free_decode_done')
    return result


@torch.no_grad()
def field_panel(model,rr,x,t,pred,out,guard,prefix,health):
    upstream={}
    if not health:
        z=zero_audio(x);a,cov=audio_variant(x,x,shift=.733)
        upstream=dict(zero=(model(z),z),shift=(model(a),a))
    result=replay_fields(model,out,x,rr['request'],{k:t[k] for k in ('heads','allowed')},pred,rr['targets'],upstream,
        health=health,checkpoint=guard.clear)
    context={'output_'+k:array(v) for k,v in out.items() if k!='count'}
    context.update({'input_'+k:array(v) for k,v in x.items() if k.endswith(('_times','_valid')) or k in ('reference','controls','mode')})
    np.savez_compressed(ART/f'{prefix}-{rr["index"]}-context.npz',**context)
    result.update(index=rr['index'],request=asdict(rr['request']),input_sha256=input_digest(x),
        count_probability=array(out['count'][0].softmax(-1)).tolist(),context_file=f'{prefix}-{rr["index"]}-context.npz')
    write(ART/f'{prefix}-{rr["index"]}.json',result)
    return dict(index=rr['index'],members=len(result['tape']),holds=sum(r['event']['type']=='hold' for r in result['tape']),
                certificate=result['certificate'],field_identity_max_diff=result['field_identity_max_diff'],arms=list(result['arms']))


def initial_parity(model,rr,x,t,plan):
    cpu=fresh(plan,'cpu');cx={k:v.cpu() for k,v in x.items()}
    co=cpu(cx);cl,_=cell_loss(cpu,co,cx,rr['request'],t);cl.backward()
    mo=model(x);ml,_=cell_loss(model,mo,x,rr['request'],t);ml.backward();torch.mps.synchronize()
    diffs={k:float((co[k].detach()-mo[k].detach().cpu()).abs().max()) for k in ('count','x','song','reference')}
    diffs['loss']=abs(float(cl.detach())-float(ml.detach()));diffs['gradient']=0.
    for (name,p),(mn,mp) in zip(cpu.named_parameters(),model.named_parameters()):
        if name!=mn or (p.grad is None)!=(mp.grad is None):raise ValueError('Gradient support identity')
        if p.grad is not None:diffs['gradient']=max(diffs['gradient'],float((p.grad-mp.grad.cpu()).abs().max()))
    if max(diffs.values())>1e-4:raise ValueError('CPU/MPS parity')
    model.zero_grad(set_to_none=True)
    return diffs


def finite_optimizer(model,opt):
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError('Nonfinite parameter')
    if any(not bool(torch.isfinite(v).all()) for state in opt.state.values() for v in state.values() if isinstance(v,torch.Tensor)):
        raise FloatingPointError('Nonfinite optimizer state')


def health(model,opt,plan,panel,row,guard,report):
    data=rows_for_health(plan,panel,row);rr=data[0];x,t=cell_batch(rr,'mps')
    report['input_probe']=audit_tensors(x);report['cpu_mps']=initial_parity(model,rr,x,t,plan);del x,t;guard.clear();first=None
    for step in range(1,101):
        guard.check('health_update');model.train();opt.zero_grad(set_to_none=True);values=[]
        for micro,rr in enumerate(data):
            report['accumulation_phase']=micro;x,t=cell_batch(rr,'mps');out=model(x);loss,detail=cell_loss(model,out,x,rr['request'],t)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Health loss nonfinite')
            (loss/len(data)).backward();values.append(float(loss.detach()));del x,t,out,loss,detail;guard.clear()
        grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        if not math.isfinite(grad):raise FloatingPointError('Health gradient')
        report['accumulation_phase']='optimizer_in_progress';opt.step();finite_optimizer(model,opt)
        report['updates']=step;report['accumulation_phase']='applied';sample=guard.clear()
        first=float(np.mean(values)) if first is None else first
        if step==1 or step%10==0:
            report['curve'].append(dict(step=step,loss=float(np.mean(values)),gradient=grad,**sample));write(ART/'health.json',report)
            print('health',step,round(float(np.mean(values)),6),round(sample['wall_s'],2),flush=True)
    records=[];p2=[];deadline=guard.start+guard.wall_s
    for rr in data:
        x,t=cell_batch(rr,'mps');model.eval()
        with torch.no_grad():
            out=model(x);pred=prediction(model,rr,x,t,guard,deadline,out)
            variants={}
            for name,counts in [('true_count_true_prefix',t['counts']),('machine_count_available_true_prefix',out['count'][0].argmax(-1).cpu().numpy())]:
                try:
                    events,_,_,_=run_members(model,out,x,rr['request'],t,counts,teacher=False,teacher_prefix=True,deadline=deadline)
                    variants[name]=dict(events=[asdict(e) for e in events],scores=score(events,rr['targets']),oracle=True)
                except (ValueError,TimeoutError) as exc:variants[name]=dict(failure=str(exc),oracle=True)
            _,_,teacher,_=run_members(model,out,x,rr['request'],t,t['counts'],teacher=True,trace=True,deadline=deadline)
            events=[RhythmEvent(**e) for e in pred['events']]
            records.append(dict(index=rr['index'],target=[asdict(e) for e in rr['targets']],prediction=pred,scores=score(events,rr['targets']),
                endpoint_min_cost=endpoint_diagnostic(events,rr['targets']),count_probability=array(out['count'][0].softmax(-1)).tolist(),
                teacher_steps=teacher,oracle_variants=variants))
            if rr['targets'] and pred['complete']:p2.append(field_panel(model,rr,x,t,pred,out,guard,'health-p2',True))
        del out,x,t;guard.clear();write(ART/'health-diagnostics.json',records)
    nonempty=[r for r in records if r['target']];empty=[r for r in records if not r['target']];gate=plan['health_gate']
    checks=dict(loss=report['curve'][-1]['loss']<=first*gate['loss_fraction'],complete=all(r['prediction']['complete'] for r in records),
        mean_full=np.mean([r['scores']['full_note_f1'] for r in nonempty])>=gate['mean_full'],
        minimum_full=min(r['scores']['full_note_f1'] for r in nonempty)>=gate['minimum_full'],
        onset10=np.mean([r['scores']['onset_10ms']['f1'] or 0 for r in nonempty])>=gate['onset10'],
        empty=all(not r['prediction']['events'] for r in empty),p2_complete=len(p2)==4,
        free_hold_path=any(p['holds']>0 for p in p2))
    # Effects are recorded, never an additional fixed-window acoustic quality gate.
    controls={}
    for arm in ('zero','shift'):
        rows=[]
        for rr in data:
            x,t=cell_batch(rr,'mps');changed=zero_audio(x) if arm=='zero' else audio_variant(x,x,shift=.733)[0]
            with torch.no_grad():pred=prediction(model,rr,changed,t,guard,deadline)
            rows.append(dict(index=rr['index'],prediction=pred,target=[asdict(e) for e in rr['targets']],scores=score([RhythmEvent(**e) for e in pred['events']],rr['targets'])))
            del x,t,changed;guard.clear();write(ART/f'health-audio-{arm}.json',rows)
        controls[arm]=dict(attempted=len(rows),complete=sum(r['prediction']['complete'] for r in rows))
    report.update(checks={k:bool(v) for k,v in checks.items()},p2=p2,controls=controls,
        status='health_passed' if all(checks.values()) else 'health_failed',health_weights_used_for_quality=False)
    return all(checks.values())


def train(model,opt,plan,panel,row,guard,report):
    seen=Counter();applied=Counter();views=Counter()
    for step,batch in enumerate(plan['sample_stream'],1):
        model.train();opt.zero_grad(set_to_none=True);values=[];current=Counter()
        for micro,s in enumerate(batch):
            report['accumulation_phase']=micro;guard.check('quality_forward')
            rr=cell_row(panel,row(s['index'],s['view'],s['field_index']));x,t=cell_batch(rr,'mps')
            seen.update(e.type for e in rr['targets']);current.update(e.type for e in rr['targets'])
            out=model(x);loss,detail=cell_loss(model,out,x,rr['request'],t)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Quality loss')
            guard.check('quality_backward');(loss/4).backward();values.append(float(loss.detach()))
            del x,t,out,loss,detail,rr
            sample=guard.clear();report['requests'].append(dict(step=step,micro=micro,index=s['index'],**sample))
            report['seen_types']=dict(seen)
        gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        if not math.isfinite(gradient):raise FloatingPointError('Quality gradient')
        for g in opt.param_groups:g['lr']=rate(plan,step)
        report['accumulation_phase']='optimizer_in_progress';guard.check('quality_optimizer');opt.step()
        finite_optimizer(model,opt);report['updates']=step;report['accumulation_phase']='applied';applied.update(current);views.update(s['view'] for s in batch)
        report['applied_exposure']=dict(types=dict(applied),events=sum(applied.values()),requests=4*step,views=dict(views))
        sample=guard.clear()
        if step==1 or step%25==0:
            report['curve'].append(dict(step=step,loss=float(np.mean(values)),gradient=gradient,lr=rate(plan,step),**sample))
            write(ART/'quality.json',report);print('quality',step,round(float(np.mean(values)),6),round(sample['wall_s'],2),flush=True)
        if step in plan['free_check_steps']:
            checks=[]
            for rr in rows_for_health(plan,panel,row)[:4]:
                x,t=cell_batch(rr,'mps')
                with torch.no_grad():pred=prediction(model,rr,x,t,guard,guard.start+guard.wall_s)
                checks.append(dict(index=rr['index'],prediction=pred))
                del x,t;guard.clear()
            write(ART/f'quality-runtime-free-{step:04d}.json',checks)
        if step<=64 and time.monotonic()-guard.start>plan['resource_prefix_wall_s']:raise TimeoutError('First64 prefix wall')
        if step==64:
            last=report['requests'][-128:];a=last[:64];b=last[64:]
            growth=float(np.median([r['physical_footprint_bytes'] for r in b])-np.median([r['physical_footprint_bytes'] for r in a]))
            median=float(np.median([r['physical_footprint_bytes'] for r in b]));compressed=float(np.median([r['compressed_bytes'] for r in b]))
            result=dict(step=64,growth_bytes=growth,median_bytes=median,compressed_median=compressed,
                passed=growth<=256*2**20 and median<=4*GIB and compressed<=GIB)
            report['runtime_prefix']=result;write(ART/'runtime-prefix64.json',result)
            if not result['passed']:raise MemoryError('Prefix64 stability')
    report['status']='quality_completed'


def output_null(events,targets,request,shifts):
    result=[];start=request.start_us;width=request.end_us-start
    for shift in shifts:
        moved=[RhythmEvent(start+(e.onset_us-start+shift)%width,
                          start+(e.onset_us-start+shift)%width+e.end_us-e.onset_us,e.type) for e in events]
        valid=[e for e in moved if e.end_us<=request.duration_us]
        original=score(moved,targets);ss=score(valid,targets);types={}
        for kind in ('tap','hold','flick','drag'):
            tp=ss['per_type'][kind]['tp'];n=sum(e.type==kind for e in moved);true=sum(e.type==kind for e in targets)
            types[kind]=dict(tp=tp,pred=n,true=true,f1=2*tp/max(n+true,1))
        result.append(dict(shift_us=shift,onset10=original['onset_10ms']['f1'] or 0,
            full50=2*sum(r['tp'] for r in types.values())/max(len(moved)+len(targets),1),
            types=types,invalid_endpoint_events=len(moved)-len(valid),
            scope='Output null only;invalid endpoints kept in predicted denominator,no event repair'))
    return result


@torch.no_grad()
def evaluate(model,plan,panel,row,guard,report):
    prior=read(str((ART/'data-identity.json').relative_to(ROOT)))['prior'];tasks=specs(plan)
    tasks.update(audio_zero=[(i,'draft',None,False) for i in plan['indices']['dev']],
        audio_shift=[(i,'draft',None,False) for i in plan['indices']['dev']],audio_foreign=[(i,'draft',None,False) for i in plan['indices']['dev']])
    summaries={};p2=[];deadline=guard.start+guard.wall_s
    for task,items in tasks.items():
        records=[]
        for index,view,bands,masked in items:
            guard.check('evaluate_'+task);rr=cell_row(panel,row(index,view,0,bands),mask_reference=masked);x,t=cell_batch(rr,'mps');control={}
            if task=='audio_zero':x=zero_audio(x);control=dict(scope='Three feature streams zero,metadata fixed')
            if task=='audio_shift':x,cov=audio_variant(x,x,shift=.733);control=dict(shift_s=.733,coverage=cov)
            if task=='audio_foreign':
                dev=plan['indices']['dev'];p=dev.index(index)
                donor=next(dev[(p+j)%len(dev)] for j in range(1,len(dev)) if panel.records[dev[(p+j)%len(dev)]]['component']!=panel.records[index]['component'])
                dr=cell_row(panel,row(donor,'draft'));dx,_=cell_batch(dr,'mps')
                scale=(dr['request'].end_us-dr['request'].start_us)/(rr['request'].end_us-rr['request'].start_us)
                x,cov=audio_variant(x,dx,scale=scale);control=dict(donor=donor,scale=scale,coverage=cov);del dx,dr
            out=model(x);pred=prediction(model,rr,x,t,guard,deadline,out);events=[RhythmEvent(**e) for e in pred['events']]
            _,_,teacher,_=run_members(model,out,x,rr['request'],t,t['counts'],teacher=True,trace=True,deadline=deadline)
            null=output_null(events,rr['targets'],rr['request'],plan['shifts_us'])
            cp=array(out['count'][0].softmax(-1))
            rec=dict(index=index,component=panel.records[index]['component'],view=view,request=asdict(rr['request']),
                target=[asdict(e) for e in rr['targets']],prediction=pred,scores=score(events,rr['targets']),control=control,
                count_probability=cp.tolist(),count_target=t['counts'].tolist(),
                count_diagnostics=dict(p0_sum=float(cp[:,0].sum()),expected_without_overflow=float((cp[:,:65]*np.arange(65)[None]).sum()),
                    overflow_probability_sum=float(cp[:,65].sum()),overflow_argmax_cells=int((cp.argmax(-1)==65).sum()),
                    scope='Nonoverflow expectation is not renormalized;full failure outcomes remain explicit'),
                teacher_steps=teacher,teacher_scores=teacher_scores(teacher,t,rr['request'],array(out['count'][0].softmax(-1)),prior),
                output_null=null,output_null_onset10=[n['onset10'] for n in null],endpoint_min_cost=endpoint_diagnostic(events,rr['targets']))
            if task=='draft' and index in plan['p2_indices']:
                if pred['complete']:p2.append(field_panel(model,rr,x,t,pred,out,guard,'quality-p2',False))
                else:p2.append(dict(index=index,failure='Baseline incomplete;fixed tape unavailable',complete=False))
            records.append(rec);write(ART/f'{task}.json',records)
            del x,t,out,teacher,rr;guard.clear()
        summaries[task]=summarize(records);write(ART/'evaluation-summary.json',summaries)
        print('eval',task,len(records),round(summaries[task]['full50_component_macro'],5),flush=True)
    draft=read(str((ART/'draft.json').relative_to(ROOT)));s=summaries['draft'];g=plan['research_gate']
    ratio=s['pred']/max(s['true'],1);positive=sum(c['onset10']>c['null10'] for c in s['components'].values());type_checks={}
    for kind in ('hold','flick','drag'):
        null_f1=[]
        for j in range(len(plan['shifts_us'])):
            tp=sum(r['output_null'][j]['types'][kind]['tp'] for r in draft);n=sum(r['output_null'][j]['types'][kind]['pred'] for r in draft);nt=sum(r['output_null'][j]['types'][kind]['true'] for r in draft)
            null_f1.append(2*tp/max(n+nt,1))
        binary=[b for r in draft for b in r['teacher_scores']['type_binary'][kind]]
        gain=float(np.mean([b['prior_nll']-b['nll'] for b in binary])) if binary else None
        tail=[a-b for r in draft for a,b in zip(r['teacher_scores']['endpoint_prior_nll'],r['teacher_scores']['endpoint_nll'])]
        endpoint_gain=float(np.mean(tail)) if tail else None
        typ=s['types'][kind]
        evidence=(gain is not None and gain>=g['proper_gain']) or (kind=='hold' and endpoint_gain is not None and endpoint_gain>=g['proper_gain'])
        type_checks[kind]=dict(**typ,null_micro_f1_mean=float(np.mean(null_f1)),binary_proper_gain=gain,binary_decisions=len(binary),
            positive_decisions=sum(b['positive'] for b in binary),endpoint_path_gain=endpoint_gain if kind=='hold' else None,
            passed=bool(typ['f1']>=g['one_non_tap_f1'] and typ['tp']>=g['non_tap_tp'] and typ['matched_components']>=g['non_tap_components'] and typ['f1']-np.mean(null_f1)>=g['non_tap_null_gain'] and evidence))
    checks=dict(complete=s['complete']==32,empty=summaries['empty']['pred']==0 and summaries['empty']['complete']==2,count=g['count_ratio'][0]<=ratio<=g['count_ratio'][1],
        full=s['full50_component_macro']>=g['full50'],null=s['onset10_component_macro']-s['null10_component_macro']>=g['null10_gain'],
        positive_components=positive>=g['positive_components'],shift=s['onset10_component_macro']-summaries['audio_shift']['onset10_component_macro']>=g['shifted10_gain'],
        foreign=s['onset10_component_macro']-summaries['audio_foreign']['onset10_component_macro']>=g['foreign10_gain'],
        non_tap=any(v['passed'] for v in type_checks.values()),p2=len(p2)==8 and all('failure' not in p for p in p2))
    report.update(status='terminal_pending_mechanism_review',evaluation=summaries,p2=p2,research_checks={k:bool(v) for k,v in checks.items()},
        research_qualified=bool(all(checks.values())),non_tap_evidence=type_checks,count_ratio=ratio,
        confirmation_scored=False,scaling_authorized=False)


def worker(stage):
    plan=read(str(PLAN.relative_to(ROOT)));start=time.monotonic();model=opt=guard=None
    report=dict(id=ID,stage=stage,status='running',updates=0,curve=[],requests=[],cloud_compute_calls=0,cloud_spend_usd=0)
    path=ART/f'{stage}.json'
    def stop(*_):raise TimeoutError('Stage wall/interrupt')
    cap=plan['preflight_wall_s'] if stage=='preflight' else plan[stage+'_wall_s']
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(cap)
    try:
        if torch.__version__!='2.14.0' or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('Runtime/fallback identity')
        torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5)
        report['worker_pid']=os.getpid();guard=ManagedMPS(ROOT,ART,cap);panel,row=load_data()
        if stage=='preflight':report.update(status='preflight_completed',data=preflight(plan,panel,row));return
        model=fresh(plan,'mps')
        report['initial_weight_sha256']=weight_digest(model);report['runtime']=torch.__version__;report['worker_pid']=os.getpid()
        report['parameters']=sum(p.numel() for p in model.parameters())
        if report['parameters']>plan['max_parameters']:raise ValueError('Parameter cap')
        if stage=='evaluation':
            checkpoint=ART/'quality-1000.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            if not state['complete'] or state['updates']!=1000 or state['plan_sha256']!=sha(PLAN):raise ValueError('Final checkpoint identity')
            model.load_state_dict(state['model']);del state;model.eval();report['checkpoint_sha256']=sha(checkpoint);evaluate(model,plan,panel,row,guard,report)
        else:
            opt=torch.optim.AdamW(model.parameters(),lr=.002 if stage=='health' else plan['learning_rate'],weight_decay=.01,betas=(.9,.999),eps=1e-8)
            if stage=='health':health(model,opt,plan,panel,row,guard,report)
            else:train(model,opt,plan,panel,row,guard,report)
    except BaseException as exc:
        report['status']='execution_failed_pending_mechanism_review';report['failure']=dict(type=type(exc).__name__,error=str(exc),phase=guard.phase if guard else stage)
    finally:
        signal.alarm(0)
        if guard:write(ART/f'{stage}-telemetry.json',guard.close())
        success=report['status'] in ('health_passed','quality_completed','terminal_pending_mechanism_review','preflight_completed')
        if model is not None and (not success or stage=='quality'):
            checkpoint=ART/('quality-1000.pt' if report['status']=='quality_completed' else f'{stage}-diagnostic-state.pt')
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict() if opt else None,updates=report['updates'],complete=report['status']=='quality_completed',
                accumulation_phase=report.get('accumulation_phase'),optimizer_step_may_be_partial=report.get('accumulation_phase')=='optimizer_in_progress',
                rng_cpu=torch.get_rng_state(),rng_mps=torch.mps.get_rng_state(),rng_python=random.getstate(),plan_sha256=sha(PLAN),resume_authorized=False),checkpoint)
            report['checkpoint']=str(checkpoint.relative_to(ROOT));report['checkpoint_sha256']=sha(checkpoint)
        report['wall_s']=time.monotonic()-start;write(path,report)


def run():
    if REPORT.exists():raise FileExistsError('No automatic rerun')
    plan=read(str(PLAN.relative_to(ROOT)))
    for source,expected in {**plan['source_sha256'],**plan['runtime_records']}.items():
        if sha(ROOT/source)!=expected:raise ValueError('Registered drift '+source)
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/plan['runtime_target'])+os.pathsep+str(ROOT)
    started=time.monotonic();report=dict(id=ID,status='running',cloud_compute_calls=0,cloud_spend_usd=0,stages={})
    for stage in ('preflight','health','quality','evaluation'):
        cap=plan['preflight_wall_s'] if stage=='preflight' else plan[stage+'_wall_s']
        if time.monotonic()-started+cap>plan['process_wall_s']:raise TimeoutError('Insufficient complete stage budget')
        if admission(ROOT,GIB,15*GIB)['status']!='admitted':raise MemoryError('Disk admission')
        report['status']=stage+'_running';report['active_stage']=stage;publish(report)
        with (ART/f'{stage}.log').open('w') as log:
            subprocess.run([str(ROOT/'.venv/bin/python'),'-u','-m','scripts.run_native_cell_recovery','worker',stage],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=cap+30,check=True)
        result=read(str((ART/f'{stage}.json').relative_to(ROOT)));report['stages'][stage]=dict(status=result['status'],updates=result['updates'],wall_s=result['wall_s'])
        expected={'preflight':'preflight_completed','health':'health_passed','quality':'quality_completed','evaluation':'terminal_pending_mechanism_review'}[stage]
        if result['status']!=expected:
            report['status']=result['status'];report['failed_stage']=stage;publish(report);return
    report['status']='terminal_pending_mechanism_review';report['wall_s']=time.monotonic()-started;publish(report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','worker']);parser.add_argument('stage',nargs='?');args=parser.parse_args()
    if args.action=='worker':worker(args.stage)
    else:globals()[args.action]()
