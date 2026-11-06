"""C17 bounded whole-event masked resampling: fresh A/C, fixed denominator."""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.masked_native_events import MaskedNativeEvents
from beatfcn.train.masked_native_events import (
    inventory_row,masked_batch,empty_state,target_state,state_events,corruption,stable_rng,
    tuple_loss,decode_masked,proper_record,visible_history,KINDS,SLOTS,
)
from beatfcn.train.shared_total_inventory import apply_update
from beatfcn.train.managed_mps import ManagedMPS
from beatfcn.train.native_quality_metrics import summarize
from beatfcn.train.native_trial_evaluation import metadata_outcomes,case_inputs,score_outcome,acoustic_pairs
from scripts import run_native_cell_recovery as original
from scripts.check_experiment_storage import admission,GIB
from scripts.run_joint_native_set import audit_tensors
from scripts.run_local_evolution import load_data
from scripts.run_local_portfolio_scale import sha,write,weight_digest
from scripts.run_native_cell_chain import closure


ROOT=Path(__file__).resolve().parents[1]
ID='c17-masked-native-events-v1'
ART=ROOT/f'artifacts/{ID}';PLAN=ROOT/f'bench/rhythm-v2/{ID}.json'
REPORT=ROOT/f'reports/local-integration-v1/{ID}.json'
DECISION='reports/council-shared-total-c17/parent-final-decision.md'
ENGINE=SimpleNamespace(cell_row=inventory_row,cell_batch=masked_batch,specs=original.specs,
    score=original.score,output_null=original.output_null,zero_audio=original.zero_audio,audio_variant=original.audio_variant)


def read(path):return json.loads(Path(path).read_text())


def runtime():
    if torch.__version__!='2.14.0' or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('Isolated torch2.14/no fallback')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5)


def publish(report):
    write(REPORT,report);p=ROOT/'reports/local-integration-v1/state.json';s=read(p)
    s.update(status=report['status'],candidate=ID,report=str(REPORT.relative_to(ROOT)),
        progress_file=f'artifacts/{ID}/{report.get("active_stage","preparation")}.json',
        training_process_running=report['status'].endswith('_running'),confirmation_scored=False)
    write(p,s)


def mask_for(plan,step,micro,health,clean,rate=None):
    return corruption(clean,stable_rng(plan['seed'],'health' if health else 'quality',step,micro,'mask'),rate)


def _prepare(guard):
    if PLAN.exists() or (ART/'preparation-registration.json').exists():raise FileExistsError('No automatic preparation retry')
    ART.mkdir(parents=True,exist_ok=True);started=time.monotonic();runtime()
    write(ART/'preparation-registration.json',dict(id=ID,wall_s=600,seed=20261070,decision_sha256=sha(ROOT/DECISION),optimizer_updates=0,cloud_calls=0))
    if admission(ROOT,GIB,15*GIB)['status']!='admitted':raise MemoryError('Storage reserve')
    parent_path=ROOT/'bench/rhythm-v2/c16-shared-total-inventory-v1.json';parent=read(parent_path)
    for name,h in {**parent['source_sha256'],**parent['runtime_records'],**parent['input_payload_sha256']}.items():
        if sha(ROOT/name)!=h:raise ValueError('Source/input/runtime drift '+name)
    panel,row=load_data();global_counts=np.full(4,.5);slot_positive=np.full(256,.5);unique_rows=set();seen_events=set();seen_history=set();history={};audits=[];mask_records=[]
    skeleton=dict(seed=20261070)
    for step,batch in enumerate(parent['sample_stream'][:400],1):
        batch_masks=[]
        for micro,s in enumerate(batch):
            guard.check('preparation_targets_and_priors')
            if time.monotonic()-started>600:raise TimeoutError('Preparation600s')
            rr=inventory_row(panel,row(s['index'],s['view'],s.get('field_index',0)));x,t=masked_batch(rr,'cpu')
            state,selected,rate=mask_for(skeleton,step,micro,False,t['state'])
            batch_masks.append(dict(rate=rate,mask_sha256=hashlib.sha256(selected.tobytes()).hexdigest()))
            chart=panel.records[s['index']]['chart_id'];occurrence=Counter();keys={}
            events=state_events(t['state'])
            ordered=sorted(events,key=lambda e:(e.onset_us,KINDS.index(e.type),e.end_us))
            for slot,e in enumerate(ordered):
                sig=(e.onset_us,e.end_us,e.type,e.lane);ordinal=occurrence[sig];occurrence[sig]+=1
                key=(chart,*sig,ordinal);keys[slot]=key
                if key not in seen_events:global_counts[KINDS.index(e.type)]+=1;seen_events.add(key)
                if selected[slot]:
                    prev=visible_history(state,slot);hkey=(key,slot//16,prev,rate)
                    if hkey not in seen_history:
                        h=history.setdefault(f'{slot//16}:{prev}',[0]*4);h[KINDS.index(e.type)]+=1;seen_history.add(hkey)
            sequence=tuple((e.onset_us,e.end_us,e.type,e.lane) for e in ordered)
            identity=(chart,rr['request'].start_us,rr['request'].end_us,sequence)
            if identity not in unique_rows:slot_positive[:len(events)]+=1;unique_rows.add(identity)
            audits.append(dict(step=step,micro=micro,index=s['index'],view=s['view'],field_index=s.get('field_index',0),events=len(events),
                target_sha256=hashlib.sha256(repr(sequence).encode()).hexdigest(),public_input_sha256=original.input_digest(x),heads=len(t['layout']['heads'])))
            del rr,x,t,state
        mask_records.append(batch_masks)
    priors=dict(global_type_probability=(global_counts/global_counts.sum()).tolist(),global_type_counts=global_counts.tolist(),
        history_counts=history,slot_active_probability=(slot_positive/(len(unique_rows)+1)).tolist(),
        duration=read(ROOT/'artifacts/c14-training-prior-audit-v1a/prior.json'),
        identities=dict(unique_absolute_events=len(seen_events),unique_request_target_sequences=len(unique_rows),unique_event_history_contexts=len(seen_history)),
        counting_policy='Global type:chart/absolute-event/duplicateordinal;NULL:unique request target sequence;history:event+slotbucket+visible-history+rate. Different units explicit.')
    write(ART/'priors.json',priors);write(ART/'training-view-support.json',dict(status='passed',requests=audits,optimizer_updates=0))
    write(ART/'mask-schedule.json',dict(seed=20261070,quality=mask_records,policy='IPW1/r conditional on chosen rate;states are corruption,not public refs'))
    random.seed(20261070);np.random.seed(20261070);torch.manual_seed(20261070)
    model=MaskedNativeEvents(FullEventConfig(**parent['model']))
    with torch.no_grad():
        mean_active=float(np.mean(priors['slot_active_probability']));model.active_read.bias.fill_(math.log(mean_active/(1-mean_active)))
        model.type_read[-1].bias.copy_(torch.tensor(np.log(priors['global_type_probability']),dtype=torch.float32))
    parameters=sum(p.numel() for p in model.parameters())
    if parameters>12_000_000:raise ValueError('12M cap')
    initial=ART/'common-initial-state.pt';torch.save(model.state_dict(),initial);digest=weight_digest(model);del model
    p=dict(id=ID,seed=20261070,model=parent['model'],architecture=dict(event_layers=4,heads=8,ffn=768,slots=256,dropout=0,ssd=False,gru=False),parameters=parameters,
        parent_plan=str(parent_path.relative_to(ROOT)),parent_plan_sha256=sha(parent_path),
        initial_state_path=str(initial.relative_to(ROOT)),initial_state_sha256=sha(initial),initial_weight_digest=digest,
        indices=parent['indices'],sample_stream=parent['sample_stream'][:400],fit_stream=parent['fit_stream'],shifts_us=parent['shifts_us'],
        input_payload_sha256=parent['input_payload_sha256'],runtime_records=parent['runtime_records'],runtime_target=parent['runtime_target'],
        priors_path=str((ART/'priors.json').relative_to(ROOT)),priors_sha256=sha(ART/'priors.json'),mask_schedule_sha256=sha(ART/'mask-schedule.json'),
        fit_updates=100,quality_updates=400,effective_batch=4,max_optimizer_updates=1002,
        wall_s=dict(preparation=600,admission_health_total=1200,quality_per_arm=3000,evaluation=1200,total=9000),
        source_decision=DECISION,decision_sha256=sha(ROOT/DECISION),
        objective='Masked complete tuple natural conditional NLL,IPW1/r,divide256;not chart exact NLL or ELBO',
        decision='2sweeps*8blocks32 positions,T1 common-uniform sampling;NULL any position;no label count',
        initial_prior_bias='Train-onlyglobalactiveodds/typefrequencybias;not per-requesttrueN',
        priors_counting_policy=priors['counting_policy'],confirmation_scored=False,scaling_authorized=False,cloud_calls=0,
        predecessors=['c16-shared-total-inventory-v1'],shared_failure_modes=['conditional_type_learning','member_endpoint','sparse_occurrence','canonical_prefix'],
        mechanism_changes=dict(
            conditional_type_learning=dict(change='Headconditionedmarkwithwhole-eventselfattentionandpublicintents',prediction='PropertypebeatsC/frequency/historyandfreenon-tapacrosscomponents',discriminating_control='MatchedrawaudiozeroA/C,allmaskproperplusactualfree'),
            member_endpoint=dict(change='Tupleheadnativefullsupport,endexactprior+zeroresidual;nohistoryexclusion',prediction='Tailproperbeatspriorandheadmatchedfreetailimproves',discriminating_control='Exactpriortreeinvariance,proper/freetailseparate'),
            sparse_occurrence=dict(change='NULLversuscompleteeventslotsinwholestate,T1fixedchain',prediction='Actualfreecount/locationwithoutteacherN',discriminating_control='Empty/null/audiozero/shift/foreign,all340freebeforeproper'),
            canonical_prefix=dict(change='Noother-eventhead/type/endexclusions;all-masktrainingandresampleownstate',prediction='Allnativelegalmultisetsremainreachable',discriminating_control='COPY/duplicates/shortgap/supporttests;transferfailureexplicit')),
        pretraining_probe_receipts=[dict(status='completed',evidence='reports/council-shared-total-c17/masked-native-tests.log',scope='New full tuple codec,exactprior residual,CPU/MPS loss-gradient and forcedNULL regression;not d256admission')],
        execution_ready=True,execution_ready_scope='Admissionentryonly;actualhealthrequiredforfreshquality')
    metadata=metadata_outcomes(ENGINE,p,panel,row);old=read(ROOT/parent['expected_free_metadata'])
    for task,records in metadata.items():
        reference={v['index']:v for v in old[task]}
        for r in records:
            if r['target']!=reference[r['index']]['target']:raise ValueError('Original target denominator drift')
    write(ART/'expected-free-metadata.json',metadata);p.update(expected_free_metadata=str((ART/'expected-free-metadata.json').relative_to(ROOT)),expected_free_metadata_sha256=sha(ART/'expected-free-metadata.json'))
    p['source_sha256']={name:sha(ROOT/name) for name in set(closure('scripts/run_masked_native_events.py'))|{'tests/test_masked_native_events.py'}}
    write(PLAN,p)
    for name in p['source_sha256']:
        target=ART/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    write(ART/'source/frozen-plan.json',p);write(ART/'preparation.json',dict(status='completed',wall_s=time.monotonic()-started,parameters=parameters,requests=1600,prior_identities=priors['identities'],optimizer_updates=0,cloud_calls=0))
    print(dict(status='prepared',parameters=parameters,wall_s=time.monotonic()-started))


def prepare():
    if PLAN.exists() or (ART/'preparation-registration.json').exists():raise FileExistsError('No automatic preparation retry')
    ART.mkdir(parents=True,exist_ok=True);runtime();guard=ManagedMPS(ROOT,ART,600)
    def stop(*_):raise TimeoutError('Preparation600s')
    old=signal.getsignal(signal.SIGALRM);signal.signal(signal.SIGALRM,stop);signal.alarm(598)
    try:_prepare(guard)
    except BaseException as exc:
        write(ART/'preparation-failure.json',dict(status='preparation_failed_pending_mechanism_review',failure=dict(type=type(exc).__name__,error=str(exc)),optimizer_updates=0,cloud_calls=0))
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old)
        write(ART/'preparation-runtime.json',dict(telemetry=guard.close(),wall_s=time.monotonic()-guard.start))


def fresh(plan,device='mps'):
    if sha(ROOT/plan['initial_state_path'])!=plan['initial_state_sha256']:raise ValueError('Initializer drift')
    random.seed(plan['seed']);np.random.seed(plan['seed']);torch.manual_seed(plan['seed'])
    model=MaskedNativeEvents(FullEventConfig(**plan['model']));model.load_state_dict(torch.load(ROOT/plan['initial_state_path'],map_location='cpu',weights_only=True))
    if weight_digest(model)!=plan['initial_weight_digest']:raise ValueError('Initial tensor digest')
    return model.to(device)


def row_inputs(panel,row,s,arm,device='mps'):
    rr=inventory_row(panel,row(s['index'],s['view'],s.get('field_index',0)));x,t=masked_batch(rr,device)
    return rr,original.zero_audio(x) if arm=='C' else x,t


def optimizer(model):return torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)


def admission_stage(model,opt,plan,panel,row,guard,report,priors):
    cpu=fresh(plan,'cpu');cpu_opt=optimizer(cpu);values=[];gradients={};diffs=dict(states=0.,active=0.,head=0.,kind=0.,loss=0.,gradient=0.)
    cases=[s for batch in plan['fit_stream'] for s in batch];report['requests']=[]
    for device,net,op in [('cpu',cpu,cpu_opt),('mps',model,opt)]:
        net.eval();op.zero_grad(set_to_none=True)
        for i,s in enumerate(cases):
            guard.check('admission_'+device);rr,x,t=row_inputs(panel,row,s,'A',device)
            blind,bt=masked_batch({**rr,'targets':[]},device)
            if any(not torch.equal(x[k],blind[k]) for k in x) or not np.array_equal(t['layout']['heads'],bt['layout']['heads']):raise ValueError('Hidden targets alter public input')
            del blind,bt
            memory=net.encode(x,t['layout'],rr['request'].start_us);out=net(memory,empty_state());selected=np.ones(256,bool)
            loss,_=tuple_loss(net,out,rr['request'],t,selected,1.,priors['duration'])
            ids=np.flatnonzero(t['state']['status']==2);lookup={int(h):j for j,h in enumerate(t['layout']['heads'])};addresses=np.array([lookup[int(t['state']['head'][j])] for j in ids],np.int64)
            current={k:out[k].detach().cpu() for k in ('states','active')}
            if len(ids):
                current.update(head=net.head_logits(out,ids).detach().cpu(),kind=net.type_logits(out,ids,addresses).detach().cpu())
            else:current.update(head=torch.empty(0),kind=torch.empty(0))
            current['loss']=float(loss.detach().cpu())
            if device=='cpu':
                values.append(current);report['requests'].append(dict(index=s['index'],view=s['view'],field_index=s.get('field_index',0),inputs=audit_tensors(x),events=len(rr['targets']),holds=sum(e.type=='hold' for e in rr['targets']),cross_window_holds=sum(e.type=='hold' and e.end_us>rr['request'].end_us for e in rr['targets']),hidden_input_unchanged=True))
            else:
                for k in ('states','active','head','kind'):
                    if current[k].numel():diffs[k]=max(diffs[k],float((current[k]-values[i][k]).abs().max()))
                diffs['loss']=max(diffs['loss'],abs(current['loss']-values[i]['loss']))
            (loss/8).backward();del rr,x,t,memory,out,loss,current;guard.clear()
        branches={name:float(p.grad.norm().cpu()) if p.grad is not None else None for name,p in net.named_parameters() if name in ('active_read.weight','head_query.weight','type_read.2.weight','tail_residual.weight')}
        report.setdefault('branches',{})[device]=branches
        if any(v is None or not math.isfinite(v) or v<=0 for v in branches.values()):raise FloatingPointError('Missing supervised tuple branch')
        if device=='cpu':gradients={n:p.grad.clone() if p.grad is not None else None for n,p in net.named_parameters()}
        else:
            for n,p in net.named_parameters():
                if (p.grad is None)!=(gradients[n] is None):raise ValueError('Gradient support parity')
                if p.grad is not None:diffs['gradient']=max(diffs['gradient'],float((p.grad.cpu()-gradients[n]).abs().max()))
            if any(not math.isfinite(v) or v>1e-4 for v in diffs.values()):raise ValueError('d256 CPU/MPS parity '+str(diffs))
        norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.))
        if not math.isfinite(norm):raise FloatingPointError('Admission gradient')
        apply_update(op,report);original.finite_optimizer(net,op)
    report.update(status='admission_passed',cpu_mps=diffs,admission_weights_discarded=True)
    del cpu,cpu_opt,values,gradients;guard.clear()
    # Actual full-width finite-chain cost on a training request, never on dev.
    rr,x,t=row_inputs(panel,row,cases[0],'A');model.eval();torch.mps.synchronize();start=time.monotonic()
    model.query_coverage={}
    with torch.no_grad():
        probe=decode_masked(model,x,rr['request'],t['layout'],priors['duration'],seed=plan['seed'],public_id=(cases[0]['index'],rr['request'].start_us,rr['request'].end_us),deadline=guard.start+guard.wall_s,clear_callback=guard.clear)
        torch.mps.synchronize();free_s=time.monotonic()-start
        start=time.monotonic();memory=model.encode(x,t['layout'],rr['request'].start_us)
        for rate in (1.,.5,.25):
            state,selected,_=corruption(t['state'],stable_rng(plan['seed'],'admission',rate),rate)
            output=model(memory,state);proper_record(model,output,rr['request'],t,state,selected,rate,priors)
            del output,state;guard.clear()
        torch.mps.synchronize();proper_s=time.monotonic()-start
    estimate=340*(free_s+proper_s)
    report['actual_cost_probe']=dict(index=cases[0]['index'],free_s=free_s,proper_three_rates_s=proper_s,
        full_panel_linear_estimate_s=estimate,scope='Single fixed train request,rough feasibility only;not a speed guarantee',prediction=probe)
    if not probe['complete']:raise ValueError('Actual full-width sampler incomplete')
    if estimate>1200:raise TimeoutError('Fixed panel runtime estimate exceeds registered1200s')
    del rr,x,t,probe,memory;guard.clear()


@torch.no_grad()
def fit_measure(model,plan,panel,row,arm,guard,priors,decode=False):
    result=dict(loss=0.,null_sum=0.,null_count=0,types={k:[] for k in KINDS},tail=[],requests=[])
    model.eval()
    for batch in plan['fit_stream']:
        for s in batch:
            guard.check('health_measure');rr,x,t=row_inputs(panel,row,s,arm);memory=model.encode(x,t['layout'],rr['request'].start_us);out=model(memory,empty_state())
            detail=proper_record(model,out,rr['request'],t,empty_state(),np.ones(256,bool),1.,priors);result['loss']+=detail['loss']
            for v in detail['occupancy']:
                if not v['positive']:result['null_sum']+=v['nll'];result['null_count']+=1
            for kind,values in detail['types'].items():result['types'][kind].extend(v['model_nll'] for v in values if v['positive'])
            result['tail'].extend(v['nll'] for v in detail['tail'])
            item=dict(index=s['index'],view=s['view'],targets=len(rr['targets']))
            del out,memory
            if decode:item['prediction']=decode_masked(model,x,rr['request'],t['layout'],priors['duration'],seed=plan['seed'],public_id=(s['index'],rr['request'].start_us,rr['request'].end_us),deadline=guard.start+guard.wall_s,clear_callback=guard.clear)
            result['requests'].append(item);del rr,x,t,detail;guard.clear()
    result['loss']/=8;result['null_nll']=result['null_sum']/max(result['null_count'],1)
    result['type_nll']={k:float(np.mean(v)) if v else None for k,v in result['types'].items()};result['type_support']={k:len(v) for k,v in result.pop('types').items()}
    result['tail_nll']=float(np.mean(result['tail'])) if result['tail'] else None;result['tail_support']=len(result.pop('tail'))
    return result


def train_stage(model,opt,plan,panel,row,guard,report,arm,health,priors):
    if read(ART/'admission.json')['status']!='admission_passed':raise ValueError('Actual admission required')
    if health:report['before']=fit_measure(model,plan,panel,row,arm,guard,priors)
    stream=plan['fit_stream'] if health else plan['sample_stream'];steps=100 if health else 400;exposure=Counter();samples=[]
    for step in range(1,steps+1):
        model.train();opt.zero_grad(set_to_none=True);losses=[];factors=Counter()
        for micro,s in enumerate(stream[(step-1)%len(stream)]):
            report['accumulation_phase']=micro;guard.check('training_forward');rr,x,t=row_inputs(panel,row,s,arm)
            memory=model.encode(x,t['layout'],rr['request'].start_us);state,selected,rate=mask_for(plan,step,micro,health,t['state']);out=model(memory,state)
            loss,detail=tuple_loss(model,out,rr['request'],t,selected,rate,priors['duration'])
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Tuple denoising loss')
            (loss/4).backward();losses.append(float(loss.detach().cpu()));exposure.update(e.type for e in rr['targets'])
            factors.update({k:float(v.detach().cpu())/4 for k,v in detail.items()})
            del rr,x,t,memory,state,selected,out,loss,detail;guard.clear()
        gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        if not math.isfinite(gradient):raise FloatingPointError('Tuple gradient')
        rate=3e-4 if health else 3e-4*step/40 if step<=40 else 3e-5+.5*(3e-4-3e-5)*(1+math.cos(math.pi*(step-40)/360))
        for group in opt.param_groups:group['lr']=rate
        guard.check('optimizer');apply_update(opt,report);original.finite_optimizer(model,opt);sample=guard.clear();samples.append(sample)
        report['exposure']=dict(exposure)
        if step==1 or step%10==0:
            report.setdefault('curve',[]).append(dict(step=step,loss=float(np.mean(losses)),factors=dict(factors),gradient=gradient,lr=rate,**sample));write(ART/f'{report["stage"]}.json',report)
            print(report['stage'],step,round(float(np.mean(losses)),6),round(sample['wall_s'],2),flush=True)
        if not health and step<=64 and time.monotonic()-guard.start>900:raise TimeoutError('First64 wall')
        if not health and step==64:
            growth=float(np.median([v['physical_footprint_bytes'] for v in samples[32:]])-np.median([v['physical_footprint_bytes'] for v in samples[:32]]));median=float(np.median([v['physical_footprint_bytes'] for v in samples[32:]]))
            report['prefix64']=dict(growth_bytes=growth,median_bytes=median,passed=growth<=256*2**20 and median<=5*GIB)
            if not report['prefix64']['passed']:raise MemoryError('First64 physical stability')
    if health:
        a=fit_measure(model,plan,panel,row,arm,guard,priors,decode=True);b=report['before'];report['after']=a
        checks=dict(finite=math.isfinite(a['loss']) and math.isfinite(a['null_nll']),free_structural=all(v['prediction']['complete'] for v in a['requests']))
        if arm=='A':checks.update(loss=a['loss']<=.5*b['loss'],null=a['null_nll']<=b['null_nll'],types=all(a['type_nll'][k] is not None and a['type_nll'][k]<b['type_nll'][k] for k in KINDS),tail=a['tail_nll'] is not None and a['tail_nll']<b['tail_nll'])
        report.update(status='health_passed' if all(checks.values()) else 'health_failed_pending_mechanism_review',checks=checks,health_weights_discarded=True)
    else:
        path=ART/f'quality-{arm}-0400.pt';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),updates=400,plan_sha256=sha(PLAN),seed=plan['seed']),path)
        report.update(status='quality_completed',checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=sha(path),weight_digest=weight_digest(model))


def load_quality(plan,arm):
    receipt=read(ART/f'quality-{arm}.json');path=ROOT/receipt['checkpoint']
    if sha(path)!=receipt['checkpoint_sha256']:raise ValueError('Quality weight identity')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['updates']!=400 or saved['plan_sha256']!=sha(PLAN):raise ValueError('Quality metadata')
    model=fresh(plan);model.load_state_dict(saved['model']);return model.eval()


@torch.no_grad()
def evaluate(plan,panel,row,guard,report,priors):
    expected=read(ROOT/plan['expected_free_metadata']);groups={arm:deepcopy(expected) for arm in ('A','C')};deadline=guard.start+guard.wall_s
    for arm,tasks in groups.items():
        for task,records in tasks.items():write(ART/f'{arm}-{task}.json',records)
    for arm in ('A','C'):
        model=load_quality(plan,arm);before=weight_digest(model)
        for task,records in groups[arm].items():
            for rec in records:
                guard.check('free_'+arm+'_'+task);rec['status']='attempted';rr,x,t,control=case_inputs(ENGINE,plan,panel,row,task,rec['index'],rec['view'],tuple(rec['bands']) if rec['bands'] else None,rec['masked'])
                if arm=='C':x=original.zero_audio(x)
                model.query_coverage={};prediction=decode_masked(model,x,rr['request'],t['layout'],priors['duration'],seed=plan['seed'],public_id=(rec['index'],rr['request'].start_us,rr['request'].end_us),deadline=deadline,clear_callback=guard.clear)
                events=[RhythmEvent(**e) for e in prediction['events']];scores,partial=score_outcome(ENGINE,events,rr['targets'],prediction['complete']);null=original.output_null(events,rr['targets'],rr['request'],plan['shifts_us'])
                if not prediction['complete']:
                    for v in null:v.update(onset10=0.,full50=0.)
                rec.update(status='completed' if prediction['complete'] else 'decode_failed',prediction=prediction,scores=scores,raw_partial_scores=partial,control=control,output_null=null,output_null_onset10=[v['onset10'] for v in null],input_sha256=original.input_digest(x),query_coverage_free=deepcopy(model.query_coverage))
                write(ART/f'{arm}-{task}.json',records);del rr,x,t,prediction,events;guard.clear()
            print('free',arm,task,len(records),flush=True)
        if weight_digest(model)!=before:raise ValueError('Evaluation weight drift')
        del model;guard.clear()
    report['free_completed']=340
    for arm in ('A','C'):
        model=load_quality(plan,arm)
        for task,records in groups[arm].items():
            for rec in records:
                guard.check('proper_'+arm+'_'+task);rr,x,t,_=case_inputs(ENGINE,plan,panel,row,task,rec['index'],rec['view'],tuple(rec['bands']) if rec['bands'] else None,rec['masked'])
                if arm=='C':x=original.zero_audio(x)
                model.query_coverage={};memory=model.encode(x,t['layout'],rr['request'].start_us);panels=[]
                for rate in (1.,.5,.25):
                    state,selected,_=corruption(t['state'],stable_rng(plan['seed'],rec['index'],rr['request'].start_us,rr['request'].end_us,rate,'proper'),rate)
                    out=model(memory,state);panels.append(proper_record(model,out,rr['request'],t,state,selected,rate,priors));del out,state;guard.clear()
                rec.update(teacher_status='completed',proper_panels=panels,query_coverage_teacher=deepcopy(model.query_coverage));write(ART/f'{arm}-{task}.json',records)
                del rr,x,t,memory,panels;guard.clear()
            print('proper',arm,task,len(records)*3,flush=True)
        del model;guard.clear()
    report['summary']={a:{task:summarize(records) for task,records in tasks.items()} for a,tasks in groups.items()}
    report['research']=research_decision(plan,groups,report['summary'])
    report.update(status='terminal_pending_mechanism_review',free_completed=340,conditional_completed=1020,confirmation_scored=False,scaling_authorized=False)


def research_decision(plan,groups,summary):
    def mean(v):return float(np.mean(v)) if v else None
    proper={};type_checks={}
    for arm in ('A','C'):
        data={k:dict(positive=[],zero=[],frequency_positive=[],history_positive=[],frequency_zero=[],history_zero=[],components=defaultdict(list)) for k in KINDS};heads=[];uniform=[];tail=[];tail_prior=[]
        for rec in groups[arm]['draft']:
            panel=rec['proper_panels'][0]
            for kind,values in panel['types'].items():
                for v in values:
                    label='positive' if v['positive'] else 'zero';data[kind][label].append(v['model_nll']);data[kind]['frequency_'+label].append(v['frequency_nll']);data[kind]['history_'+label].append(v['history_nll'])
                    if v['positive']:data[kind]['components'][rec['component']].append(v)
            heads.extend(panel['head_nll']);uniform.extend([panel['head_uniform_nll']]*len(panel['head_nll']));tail.extend(v['nll'] for v in panel['tail']);tail_prior.extend(v['prior_nll'] for v in panel['tail'])
        proper[arm]=dict(types={k:{**{name:mean(v) for name,v in values.items() if name!='components'},'positive_support':len(values['positive']),'zero_support':len(values['zero']),'components':{c:{field:mean([v[field] for v in vv]) for field in ('model_nll','frequency_nll','history_nll')} for c,vv in values['components'].items()}} for k,values in data.items()},head=mean(heads),head_uniform=mean(uniform),tail=mean(tail),tail_prior=mean(tail_prior))
    for kind in KINDS:
        a=proper['A']['types'][kind];c=proper['C']['types'][kind]
        improved=sum(c['components'].get(comp,{}).get('model_nll',-math.inf)>v['model_nll'] and v['frequency_nll']>v['model_nll'] and v['history_nll']>v['model_nll'] for comp,v in a['components'].items())
        valid=all(v is not None for v in [a['positive'],c['positive'],a['frequency_positive'],a['history_positive'],a['zero'],c['zero'],a['frequency_zero'],a['history_zero']])
        type_checks[kind]=valid and min(c['positive']-a['positive'],a['frequency_positive']-a['positive'],a['history_positive']-a['positive'])>=.02 and max(a['zero']-c['zero'],a['zero']-a['frequency_zero'],a['zero']-a['history_zero'])<=.005 and improved>=3
    a=summary['A']['draft'];pairs={k:acoustic_pairs(groups['A']['draft'],groups['A'][k]) for k in ('audio_shift','audio_foreign')}
    density=list(zip(groups['A']['density_low'],groups['A']['density_high']))
    checks=dict(complete=all(r['prediction']['complete'] for tasks in groups.values() for records in tasks.values() for r in records),
        proper_types=all(type_checks.values()),head=proper['A']['head']<proper['A']['head_uniform'],tail=proper['A']['tail_prior']-proper['A']['tail']>=.05,
        full=a['full50_component_macro']>=.27,onset=a['onset10_component_macro']>=.25,count=.6<=a['pred']/max(a['true'],1)<=1.4,
        types=all(a['types'][k]['tp']>=n for k,n in dict(tap=225,hold=5,flick=5,drag=16).items()) and all(a['types'][k]['matched_components']>=n for k,n in dict(hold=2,flick=3,drag=6).items()),
        empty=summary['A']['empty']['pred']<=13,acoustic=all(v['positive_covered_components']>=10 and v['common_covered_components']>=10 and v['covered_delta_onset10'] is not None and v['covered_delta_onset10']>=.10 for v in pairs.values()),
        reference=summary['A']['completion']['full50_component_macro']-summary['A']['reference_masked']['full50_component_macro']>=.01,
        density=sum(len(hi['prediction']['events'])>len(lo['prediction']['events']) for lo,hi in density)>=3 and all(r['prediction']['complete'] for pair in density for r in pair))
    return dict(checks={k:bool(v) for k,v in checks.items()},supported=all(checks.values()),proper=proper,type_checks={k:bool(v) for k,v in type_checks.items()},acoustic=pairs,
        scope='Complete new recipe exploratory only;conditional denoising not chartNLL;A/C whole audio effect not component causality')


def worker(stage,cap):
    runtime();plan=read(PLAN)
    for name,h in plan['source_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Frozen source drift '+name)
    if sha(ROOT/plan['priors_path'])!=plan['priors_sha256']:raise ValueError('Prior drift')
    priors=read(ROOT/plan['priors_path']);guard=ManagedMPS(ROOT,ART,cap);report=dict(id=ID,stage=stage,status='running',updates=0,optimizer_updates=0,cloud_calls=0);model=opt=None
    def stop(*_):raise TimeoutError('Registered stage cap/interruption')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(max(1,int(cap)-2))
    try:
        panel,row=load_data()
        if stage=='evaluation':evaluate(plan,panel,row,guard,report,priors)
        else:
            model=fresh(plan);opt=optimizer(model);report['initial_digest']=weight_digest(model)
            if stage=='admission':admission_stage(model,opt,plan,panel,row,guard,report,priors)
            else:
                health=stage.startswith('health');arm=stage[-1]
                if not health and any(read(ART/f'health-{a}.json')['status']!='health_passed' for a in ('A','C')):raise ValueError('Both actual health receipts required')
                train_stage(model,opt,plan,panel,row,guard,report,arm,health,priors)
    except BaseException as exc:
        report.update(status='execution_failed_pending_mechanism_review',failure=dict(type=type(exc).__name__,error=str(exc),phase=guard.phase))
        if model is not None:
            path=ART/f'{stage}-failure-state.pt';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict() if opt else None,updates=report['updates'],phase=report.get('accumulation_phase'),plan_sha256=sha(PLAN)),path);report.update(failure_checkpoint=str(path.relative_to(ROOT)),failure_checkpoint_sha256=sha(path))
    finally:
        signal.alarm(0);report.update(telemetry=guard.close(),wall_s=time.monotonic()-guard.start);write(ART/f'{stage}.json',report)
    print({k:report.get(k) for k in ('stage','status','updates','wall_s','failure')},flush=True)


def run():
    if REPORT.exists():raise FileExistsError('No automatic experiment retry')
    plan=read(PLAN);prep=read(ART/'preparation.json');started=time.monotonic();admission_health=0
    env={**os.environ,'PYTHONPATH':str(ROOT/plan['runtime_target'])+os.pathsep+str(ROOT)}
    report=dict(id=ID,status='registered',stages={},preparation_wall_s=prep['wall_s'],optimizer_updates=0,cloud_calls=0,cloud_spend_usd=0,confirmation_scored=False,scaling_authorized=False)
    try:
        for stage in ('admission','health-A','health-C','quality-A','quality-C','evaluation'):
            remaining=9000-prep['wall_s']-(time.monotonic()-started)
            cap=min(1200-admission_health if stage=='admission' or stage.startswith('health') else 1200 if stage=='evaluation' else 3000,remaining)
            if cap<=2:raise TimeoutError('Registered wall exhausted')
            if admission(ROOT,GIB,15*GIB)['status']!='admitted':raise MemoryError('Disk reserve')
            report.update(status=stage+'_running',active_stage=stage);publish(report)
            with (ART/f'{stage}.log').open('w') as log:subprocess.run([str(ROOT/'.venv/bin/python'),'-u','-m','scripts.run_masked_native_events','worker',stage,str(int(cap))],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=cap,check=True)
            r=read(ART/f'{stage}.json');report['stages'][stage]={k:r.get(k) for k in ('status','updates','wall_s','failure')};report['optimizer_updates']+=r.get('updates',0)
            if stage=='admission' or stage.startswith('health'):admission_health+=r['wall_s']
            expected='admission_passed' if stage=='admission' else 'health_passed' if stage.startswith('health') else 'terminal_pending_mechanism_review' if stage=='evaluation' else 'quality_completed'
            if r['status']!=expected:report.update(status=r['status'],failed_stage=stage);break
        else:report.update(status='terminal_pending_mechanism_review',research=read(ART/'evaluation.json')['research'])
    except BaseException as exc:report.update(status='supervisor_failed_pending_mechanism_review',failure=dict(type=type(exc).__name__,error=str(exc)))
    finally:report['wall_s']=time.monotonic()-started+prep['wall_s'];publish(report)
    print({k:report.get(k) for k in ('status','failed_stage','optimizer_updates','wall_s','failure')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','worker']);parser.add_argument('stage',nargs='?');parser.add_argument('cap',nargs='?',type=int);args=parser.parse_args()
    if args.action=='worker':worker(args.stage,args.cap)
    else:globals()[args.action]()
