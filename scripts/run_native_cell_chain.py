"""Registered fresh cell/member candidate; serial local execution and no cloud."""
from collections import Counter
from dataclasses import asdict
import argparse
import ast
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_bundle import read_grid
from beatfcn.model.native_cell_chain import NativeCellChain
from beatfcn.model.sample import match_f1
from beatfcn.train.native_cell_chain import cell_row,cell_batch,cell_loss,decode_cells,run_members,SUPPORT
from scripts.check_experiment_storage import admission,GIB
from scripts.run_c07_count_ancestral import specs
from scripts.run_local_evolution import load_data
from scripts.run_local_portfolio_scale import sha,write,weight_digest
from scripts.run_local_typed_canvas import score
from scripts.review_frontend_canvas_comparison import metrics
from scripts.run_joint_native_set import configure,audit_tensors


ROOT=Path(__file__).resolve().parents[1];ID='c11-native-cell-chain-v1'
PLAN=ROOT/f'bench/rhythm-v2/{ID}.json';ART=ROOT/f'artifacts/{ID}'
REPORT=ROOT/'reports/local-integration-v1/native-cell-chain.json'


def read(name):return json.loads((ROOT/name).read_text())


def save(report):
    write(REPORT,report);p=ROOT/'reports/local-integration-v1/state.json';s=json.loads(p.read_text())
    s.update(status=report['status'],candidate=ID,report=str(REPORT.relative_to(ROOT)),health_updates=report['health_updates'],
        quality_updates=report['quality_updates'],new_neural_updates=report['quality_updates'],training_process_running=report['status'] in ('health_running','quality_running'))
    write(p,s)


def journal(message):
    with (ROOT/'EXPERIMENT_LOG.md').open('a') as f:f.write('\n\n'+ID+' '+message+'\n')


def closure(entry):
    pending=[Path(entry)];seen=set()
    while pending:
        path=pending.pop()
        if str(path) in seen:continue
        seen.add(str(path));parts=list(path.with_suffix('').parts);package=parts[:-1]
        for node in ast.walk(ast.parse((ROOT/path).read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                prefix=package[:len(package)-node.level+1] if node.level else []
                base='.'.join(prefix+([node.module] if node.module else []));modules=[base]+[base+'.'+a.name for a in node.names]
            for module in modules:
                if not module.startswith(('beatfcn.','scripts.')):continue
                source=Path(*module.split('.')).with_suffix('.py')
                if not (ROOT/source).is_file():source=Path(*module.split('.'))/'__init__.py'
                if (ROOT/source).is_file():pending.append(source)
    return sorted(seen)


def fresh(plan,device='mps'):
    torch.manual_seed(plan['seed'])
    if device=='mps':torch.mps.manual_seed(plan['seed'])
    return NativeCellChain(FullEventConfig(**plan['model']),density_reference=plan['density_reference']).to(device)


def prepare():
    if PLAN.exists() or REPORT.exists():raise FileExistsError('Preserve registered trial')
    old=read('bench/rhythm-v2/joint-native-set-v1a.json')
    keys=('seed','model','density_reference','indices','sample_stream','health_indices','empty_health_index','health_steps',
          'quality_steps','effective_batch','learning_rate','minimum_learning_rate','warmup','weight_decay','clip',
          'health_wall_s','quality_wall_s','evaluation_wall_s','process_wall_s','input_payload_sha256','shifts_us')
    p={k:old[k] for k in keys}
    p.update(id=ID,preflight_wall_s=400,max_parameters=12000000,new_disk_peak_bytes=GIB,reserve_bytes=15*GIB,
        architecture='SSD cell context + explicit song read/cross-attention + GRU192 canonical members + head-conditioned types + exact endpoint tree',
        count_support='0..64 and OVERFLOW;core256;request-only addresses',objective='Natural joint count/member NLL divided by public valid cell count;no balanced/focal/auxiliary losses',
        decision='Count marginal mode then conditional canonical head/type/tree modes;not jointMAP',
        optimizer='AdamW all trainable parameters wd=.01,betas(.9,.999),eps1e-8;whole-recipe comparison,not same param-group decay as old concat',
        serialization='Exact-us field tape v1;old token grammar additionally checked on its native-tail subdomain;fractional tails require the explicit extended tape',
        health_gate=dict(loss_fraction=.15,mean_full=.90,minimum_full=.75,onset10=.80,empty_events=0,parity=1e-4),
        quality_gate=dict(full50=.3398040839133824,two_non_tap_gains=.03,third_non_tap_max_drop=.02,count_ratio=[.5,1.75],
            null_gain=.05,positive_components=10,input_shift_gain=.03,foreign_gain=.03),
        research_retention=dict(full_floor=.3003642151482393,one_non_tap_gain=.03,proper_score_subfinding_allowed=True),
        failure_weight_policy='Save one failed-health state_dict for necessary frozen intervention;no optimizer or routine intermediate weights',
        audio_controls='Content shifted+733ms on all local/mel/song clocks;timestamps/request/reference/controls fixed. Foreign donor first next different component;affine relative-core-time mapping for all3streams;coverage recorded.',
        cloud_compute_calls=0,cloud_spend_usd=0,scaling_authorized=False,confirmation_scored=False,
        execution_ready=False,pretraining_probe_receipts=[],predecessors=['joint-native-set-v1a','evolution-ssd-full-v1','c08-no-mark-history-v1'],
        shared_failure_modes=['global_query_head_crowding','fractional_tail_support','count_decision_calibration','teacher_free_interface'],
        mechanism_changes={
            'global_query_head_crowding':dict(change='Cell total count and canonical member chain replace global interchangeable query heads',prediction='No global query identity migration;local free coverage can fit',discriminating_control='Actual five-window free reconstruction,multiplicity and prefix diagnostics;full32same targets after admission'),
            'fractional_tail_support':dict(change='Exact native-or-authorized-fractional endpoint tree and explicit field tape',prediction='1usCOPYholds and same-head distinct tails roundtrip and normalize',discriminating_control='Constructed endpoint enumeration/translation and4000actualtargetSHA replay'),
            'count_decision_calibration':dict(change='One natural count likelihood;no positive/zero balancing,expected rounding or tail rank',prediction='Learned mode is directly inspectable with natural negative evidence',discriminating_control='All-cell count probabilities/CE,p0/expected/hard count and unseen empty windows'),
            'teacher_free_interface':dict(change='Selected-head acoustic type and conditional endpoint acoustic tree;only cell-local history',prediction='Teacher/free gap is localized rather than silently fixed upstream',discriminating_control='Truecount/trueprefix,machinecount/prefix diagnostics and actual audio controls')})
    files=closure('scripts/run_native_cell_chain.py')+['tests/test_exact_cell_support.py','tests/test_native_cell_chain.py','beatfcn/train/consistent_assignment.py']
    p['source_sha256']={name:sha(ROOT/name) for name in sorted(set(files))}
    write(PLAN,p)
    for name in p['source_sha256']:
        target=ART/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    shutil.copy2(PLAN,ART/'source/registered-plan.json')
    journal('冻结完整C11后继，实际静态点数10,240,592；源码依赖闭包、精确COPY新field tape和自然NLL。准备真实探针，尚无健康/质量通过。')


def resources(plan,started,cap):
    torch.mps.synchronize()
    if time.monotonic()-started>cap:raise TimeoutError('Stage wall cap')
    disk=admission(ROOT,plan['new_disk_peak_bytes'],plan['reserve_bytes'])
    if disk['status']!='admitted':raise ValueError('Disk reserve')
    used=sum(p.stat().st_size for p in ART.rglob('*') if p.is_file())
    if used>plan['new_disk_peak_bytes']:raise ValueError('Actual candidate disk cap')
    driver=torch.mps.driver_allocated_memory();recommended=torch.mps.recommended_max_memory()
    if driver>min(.8*recommended,10*GIB):raise ValueError('Driver memory cap')
    return dict(wall_s=time.monotonic()-started,driver_bytes=driver,allocated_bytes=torch.mps.current_allocated_memory(),
        recommended_bytes=recommended,free_disk_bytes=shutil.disk_usage(ROOT).free,new_artifact_bytes=used)


def preflight(plan,panel,row):
    started=time.monotonic()
    for name,expected in plan['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Source drift '+name)
    for name,expected in plan['input_payload_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Cache/target payload drift '+name)
    old=read('artifacts/joint-native-set-v1a/data-identity.json');types=Counter();checks=[]
    for batch in plan['sample_stream']:
        for s in batch:
            rr=cell_row(panel,row(s['index'],s['view'],s['field_index']));counts,members=SUPPORT.encode(rr['targets'],rr['request'])
            if Counter(SUPPORT.decode(counts,members,rr['request']))!=Counter(rr['targets']):raise ValueError('Exact multiset mismatch')
            digest=hashlib.sha256(json.dumps([asdict(e) for e in sorted(rr['targets'])],sort_keys=True).encode()).hexdigest()
            if digest!=old['checks'][len(checks)]['target_sha256']:raise ValueError('Original exposure identity changed')
            types.update(e.type for e in rr['targets']);checks.append(dict(**s,events=len(members),target_sha256=digest))
            if time.monotonic()-started>plan['preflight_wall_s']:raise TimeoutError('Preflight wall')
    receipt=dict(status='completed',requests=len(checks),types=dict(types),events=sum(types.values()),checks=checks,wall_s=time.monotonic()-started)
    write(ART/'data-identity.json',receipt)
    return {k:v for k,v in receipt.items() if k!='checks'}


@torch.no_grad()
def diagnostics(model,batches,save_prefix,deadline):
    model.eval();records=[]
    for rr,x,t in batches:
        output=model(x);layout={k:t[k] for k in ('heads','allowed')}
        pred=decode_cells(model,x,rr['request'],layout,trace=True,output=output,deadline=deadline)
        p=[RhythmEvent(**e) for e in pred['events']]
        variants={}
        for name,count in [('true_count_true_prefix',t['counts']),('machine_count_available_true_prefix',output['count'][0].argmax(-1).cpu().numpy())]:
            try:
                events,_,_,_=run_members(model,output,x,rr['request'],t,count,teacher=False,teacher_prefix=True,deadline=deadline)
                variants[name]=dict(scores=score(events,rr['targets']),events=[asdict(e) for e in events],oracle=True)
            except (ValueError,TimeoutError) as exc:variants[name]=dict(failure=str(exc),oracle=True)
        _,parts,teacher_steps,_=run_members(model,output,x,rr['request'],t,t['counts'],teacher=True,trace=True,deadline=deadline)
        records.append(dict(index=rr['index'],view=rr['view'],target=[asdict(e) for e in rr['targets']],prediction=pred,
            scores=score(p,rr['targets']),count_probability=output['count'][0].softmax(-1).cpu().tolist(),
            count_target=t['counts'].tolist(),teacher_nll={k:float(v) for k,v in parts.items()},teacher_steps=teacher_steps,oracle_variants=variants))
    write(ART/(save_prefix+'.json'),records)
    return records


def health(plan,panel,row,report):
    model=fresh(plan);report['parameters']=sum(p.numel() for p in model.parameters())
    if report['parameters']>plan['max_parameters']:raise ValueError('Parameter cap')
    report['initial_weight_sha256']=weight_digest(model)
    batches=[]
    for index,view in list(zip(plan['health_indices'],('draft','single_condition','random_groups','middle')))+[(plan['empty_health_index'],'draft')]:
        rr=cell_row(panel,row(index,view));x,t=cell_batch(rr,'mps');batches.append((rr,x,t))
    report['inputs']=[dict(index=rr['index'],tensors=audit_tensors(x)) for rr,x,t in batches]
    rr,x,t=batches[0];cpu=fresh(plan,'cpu');cx={k:v.cpu() for k,v in x.items()}
    co=cpu(cx);cl,_=cell_loss(cpu,co,cx,rr['request'],t);cl.backward()
    mo=model(x);ml,_=cell_loss(model,mo,x,rr['request'],t);ml.backward();torch.mps.synchronize()
    differences={k:float((co[k].detach()-mo[k].detach().cpu()).abs().max()) for k in ('count','x','song','reference')}
    differences['loss']=abs(float(cl.detach())-float(ml.detach()));gradient=0.;missing=[]
    for (name,p),(mn,mp) in zip(cpu.named_parameters(),model.named_parameters()):
        if name!=mn:raise ValueError('Parameter order')
        if (p.grad is None)!=(mp.grad is None):raise ValueError('CPU/MPS missing gradient '+name)
        if p.grad is not None:gradient=max(gradient,float((p.grad-mp.grad.cpu()).abs().max()))
        else:missing.append(name)
    differences['gradient']=gradient;report['cpu_mps']=differences;report['missing_gradient_first_request']=missing
    if max(differences.values())>plan['health_gate']['parity']:raise ValueError('CPU/MPS parity')
    del cpu,co,mo,cl,ml,cx;model.zero_grad(set_to_none=True);gc.collect();torch.mps.empty_cache()
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01,betas=(.9,.999),eps=1e-8)
    started=time.monotonic();first=None
    for step in range(1,101):
        model.train();opt.zero_grad(set_to_none=True);values=[];parts=[]
        for rr,x,t in batches:
            loss,detail=cell_loss(model,model(x),x,rr['request'],t)
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite health loss')
            (loss/len(batches)).backward();values.append(float(loss.detach()));parts.append({k:float(v.detach()) if isinstance(v,torch.Tensor) else v for k,v in detail.items()})
        grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        if not math.isfinite(grad):raise ValueError('Nonfinite health gradient')
        opt.step();first=np.mean(values) if first is None else first;report['health_updates']=step
        if step==1 or step%10==0:
            resource=resources(plan,started,plan['health_wall_s']);report['health_curve'].append(dict(step=step,loss=float(np.mean(values)),parts=parts,gradient=grad,**resource));save(report)
            print('health',step,float(np.mean(values)),round(resource['wall_s'],2),flush=True)
    records=diagnostics(model,batches,'health-diagnostics',time.monotonic()+min(300,plan['evaluation_wall_s']))
    nonempty=[r for r in records if r['target']];empty=[r for r in records if not r['target']];gate=plan['health_gate']
    checks=dict(loss=report['health_curve'][-1]['loss']<=first*gate['loss_fraction'],complete=all(r['prediction']['complete'] for r in records),
        mean_full=np.mean([r['scores']['full_note_f1'] for r in nonempty])>=gate['mean_full'],
        minimum_full=min(r['scores']['full_note_f1'] for r in nonempty)>=gate['minimum_full'],
        onset10=np.mean([r['scores']['onset_10ms']['f1'] or 0 for r in nonempty])>=gate['onset10'],empty=all(not r['prediction']['events'] for r in empty))
    report['health_checks']={k:bool(v) for k,v in checks.items()};report['health_passed']=all(checks.values())
    if not report['health_passed']:
        checkpoint=ART/'health-failed-0100.pt';torch.save(dict(model=model.state_dict(),updates=100,quality_updates=0,plan_sha256=sha(PLAN)),checkpoint)
        report['failure_checkpoint']=str(checkpoint.relative_to(ROOT));report['failure_checkpoint_sha256']=sha(checkpoint)
    report['health_wall_s']=time.monotonic()-started;save(report)
    del model,opt,batches;gc.collect();torch.mps.empty_cache()
    return report['health_passed']


def rate(plan,step):
    if step<=100:return plan['learning_rate']*step/100
    fraction=(step-100)/900
    return plan['minimum_learning_rate']+(plan['learning_rate']-plan['minimum_learning_rate'])*(1+math.cos(math.pi*fraction))/2


def train(plan,panel,row,report):
    model=fresh(plan)
    if weight_digest(model)!=report['initial_weight_sha256']:raise ValueError('Fresh initialization mismatch')
    report['health_weights_used']=False;report['status']='quality_running';save(report)
    opt=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=.01,betas=(.9,.999),eps=1e-8)
    started=time.monotonic();types=Counter();views=Counter();completed=False
    try:
        for step,batch in enumerate(plan['sample_stream'],1):
            model.train();opt.zero_grad(set_to_none=True);losses=[]
            for s in batch:
                rr=cell_row(panel,row(s['index'],s['view'],s['field_index']));x,t=cell_batch(rr,'mps');loss,_=cell_loss(model,model(x),x,rr['request'],t)
                if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite quality loss')
                (loss/len(batch)).backward();losses.append(float(loss.detach()));types.update(e.type for e in rr['targets']);views.update([s['view']])
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
            if not math.isfinite(grad):raise ValueError('Nonfinite quality gradient')
            for group in opt.param_groups:group['lr']=rate(plan,step)
            opt.step();report['quality_updates']=step
            if step==1 or step%25==0:
                resource=resources(plan,started,plan['quality_wall_s']);report['quality_curve'].append(dict(step=step,loss=float(np.mean(losses)),gradient=grad,**resource));save(report)
                print('quality',step,float(np.mean(losses)),round(resource['wall_s'],2),flush=True)
        completed=True
    finally:
        path=ART/('quality-1000.pt' if completed else 'quality-interrupted.pt')
        torch.save(dict(model=model.state_dict(),updates=report['quality_updates'],complete=completed,plan_sha256=sha(PLAN)),path)
        report['checkpoint']=str(path.relative_to(ROOT));report['checkpoint_sha256']=sha(path);report['quality_wall_s']=time.monotonic()-started
        report['exposure']=dict(types=dict(types),views=dict(views),windows=sum(views.values()),events=sum(types.values()));save(report)
    return model


def audio_variant(inputs,source,*,shift=None,scale=1.):
    result=dict(inputs);coverage={}
    for name,spacing in (('local',.04),('mel',.01),('song',.5)):
        values=source[name];times=source[name+'_times'];valid=source[name+'_valid'];query=inputs[name+'_times']*scale
        if shift is not None:query=query+shift
        if name=='local':
            b,t,n,f,d=values.shape;flat=values.permute(0,2,1,3,4).reshape(b,n,t*f*d)
            changed=read_grid(flat,times,valid,query,spacing).reshape(b,query.shape[1],t,f,d).permute(0,2,1,3,4)
        else:changed=read_grid(values,times,valid,query,spacing)
        result[name]=changed
        lo=times[:,:1]-spacing/2;hi=times.masked_fill(~valid,-1e9).amax(1,keepdim=True)+spacing/2
        coverage[name]=float((((query>=lo)&(query<=hi))&inputs[name+'_valid']).sum()/inputs[name+'_valid'].sum().clamp_min(1))
    return result,coverage


@torch.no_grad()
def evaluate(model,plan,panel,row,report):
    model.eval();started=time.monotonic();deadline=started+plan['evaluation_wall_s'];tasks=specs(plan)
    tasks.update(audio_shift=[(i,'draft',None,False) for i in plan['indices']['dev']],audio_foreign=[(i,'draft',None,False) for i in plan['indices']['dev']])
    summary={}
    for task,items in tasks.items():
        records=[]
        for index,view,bands,masked in items:
            if time.monotonic()>deadline:raise TimeoutError('Evaluation cap')
            rr=cell_row(panel,row(index,view,0,bands),mask_reference=masked);x,t=cell_batch(rr,'mps');control={}
            if task=='audio_shift':x,coverage=audio_variant(x,x,shift=.733);control=dict(shift_s=.733,coverage=coverage)
            elif task=='audio_foreign':
                dev=plan['indices']['dev'];start=dev.index(index);donor=next(dev[(start+j)%len(dev)] for j in range(1,len(dev)) if panel.records[dev[(start+j)%len(dev)]]['component']!=panel.records[index]['component'])
                dr=cell_row(panel,row(donor,'draft'));dx,_=cell_batch(dr,'mps');scale=(dr['request'].end_us-dr['request'].start_us)/(rr['request'].end_us-rr['request'].start_us)
                x,coverage=audio_variant(x,dx,scale=scale);control=dict(donor=donor,scale=scale,coverage=coverage)
            output=model(x);prediction=decode_cells(model,x,rr['request'],{k:t[k] for k in ('heads','allowed')},trace=True,output=output,deadline=deadline)
            events=[RhythmEvent(**e) for e in prediction['events']];scores=score(events,rr['targets']);start=rr['request'].start_us;width=rr['request'].end_us-start
            shifts=[match_f1([start+(e.onset_us-start+d)%width for e in events],[e.onset_us for e in rr['targets']],10000)[3] for d in plan['shifts_us']]
            _,_,teacher,holds=run_members(model,output,x,rr['request'],t,t['counts'],teacher=True,trace=True,deadline=deadline)
            records.append(dict(index=index,component=panel.records[index]['component'],view=view,request=asdict(rr['request']),prediction=prediction,
                target=[asdict(e) for e in rr['targets']],scores=scores,shifted_onset10=shifts,control=control,
                count_probability=output['count'][0].softmax(-1).cpu().tolist(),count_target=t['counts'].tolist(),teacher_steps=teacher,teacher_conditioned=True))
            write(ART/f'{task}.json',records)
        summary[task]=metrics(records)
    report['evaluation']=summary;report['evaluation_wall_s']=time.monotonic()-started;write(ART/'evaluation-summary.json',summary)


def run():
    if REPORT.exists():raise FileExistsError('No automatic experiment rerun')
    configure();plan=read(str(PLAN.relative_to(ROOT)));disk=admission(ROOT,plan['new_disk_peak_bytes'],plan['reserve_bytes'])
    if disk['status']!='admitted':raise ValueError('Disk admission')
    report=dict(id=ID,status='preflight_running',health_updates=0,quality_updates=0,health_curve=[],quality_curve=[],failures=[],cloud_compute_calls=0,cloud_spend_usd=0,disk=disk)
    save(report);started=time.monotonic()
    def stop(*_):raise TimeoutError('Process cap or external interruption')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(plan['process_wall_s'])
    try:
        panel,row=load_data();report['data_probe']=preflight(plan,panel,row);report['status']='health_running';save(report)
        if not health(plan,panel,row,report):report['status']='health_failed_pending_mechanism_review';return
        receipt=ART/'pretraining-probe.json';write(receipt,dict(status='completed',data_probe=report['data_probe'],parameters=report['parameters'],cpu_mps=report['cpu_mps'],health_checks=report['health_checks']))
        plan.update(execution_ready=True,pretraining_probe_receipts=[dict(status='completed',evidence=str(receipt.relative_to(ROOT)),scope='new_real_data_model_health')]);write(PLAN,plan);shutil.copy2(PLAN,ART/'source/frozen-plan.json')
        subprocess.run([str(ROOT/'.venv/bin/python'),'-m','scripts.validate_mechanism_reviews','--all','--successor',str(PLAN)],cwd=ROOT,check=True)
        report['plan_sha256']=sha(PLAN);model=train(plan,panel,row,report);report['status']='evaluation_running';save(report)
        evaluate(model,plan,panel,row,report);report['status']='terminal_pending_mechanism_review'
    except BaseException as exc:
        report['status']='execution_failed_pending_mechanism_review';report['failures'].append(dict(type=type(exc).__name__,error=str(exc)));raise
    finally:
        signal.alarm(0);report['process_wall_s']=time.monotonic()-started;save(report);journal(f"终态/阶段{report['status']}，健康{report['health_updates']}，质量{report['quality_updates']}，无云。")


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run'));args=p.parse_args();{'prepare':prepare,'run':run}[args.action]()
