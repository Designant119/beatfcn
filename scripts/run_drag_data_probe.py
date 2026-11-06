"""Small matched data-only intervention; complete C17 tuple law, no cloud."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import random
from pathlib import Path
import signal
import time

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.masked_native_events import MaskedNativeEvents
from beatfcn.train.drag_density_filter import DragFilteredPanel, RULE, filtered_request_row
from beatfcn.train.masked_native_events import (masked_batch, empty_state, target_state, corruption,
    tuple_loss, decode_masked, stable_rng, KINDS)
from beatfcn.train.native_diagnostic_priors import duration_probability
from beatfcn.train.managed_mps import ManagedMPS
from beatfcn.train.typed_inventory import inventory_row
from scripts.run_local_evolution import load_data
from scripts.run_native_cell_chain import closure
from scripts.run_local_portfolio_scale import weight_digest
from scripts.run_local_typed_canvas import score
from scripts import run_native_cell_recovery as original


ROOT=Path(__file__).resolve().parents[1]
ID='data-drag-density-probe-v1'
ART=ROOT/'artifacts'/ID
PLAN=ROOT/'bench/rhythm-v2'/(ID+'.json')
REPORT=ROOT/'reports/local-integration-v1'/(ID+'.json')


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def runtime():
    if torch.__version__!='2.14.0' or __import__('os').environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('Requires isolated2.14/no fallback')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5)


def get_row(panel,row,filtered,index,arm):
    rr=row(index,'draft',0)
    return inventory_row(panel,rr) if arm=='raw' else filtered_request_row(filtered,rr)


def batch(panel,row,filtered,index,arm,device='mps'):
    rr=get_row(panel,row,filtered,index,arm);x,t=masked_batch(rr,device)
    return rr,x,t


def fresh(plan,device='mps'):
    if sha(ROOT/plan['initial_path'])!=plan['initial_sha256']:raise ValueError('Initializer drift')
    net=MaskedNativeEvents(FullEventConfig(**plan['model']))
    net.load_state_dict(torch.load(ROOT/plan['initial_path'],map_location='cpu',weights_only=True))
    if weight_digest(net)!=plan['initial_tensor_digest']:raise ValueError('Tensor init drift')
    return net.to(device)


def prepare():
    if PLAN.exists():raise FileExistsError('Preserve frozen probe')
    ART.mkdir(parents=True,exist_ok=True);start=time.monotonic();runtime()
    panel,row=load_data();filtered=DragFilteredPanel(panel)
    parent=read(ROOT/'bench/rhythm-v2/c17-masked-native-events-v1.json')
    audit=read(ROOT/'reports/local-integration-v1/data-drag-density-audit-v1.json')
    affected=[r['index'] for r in audit['per_request']['train64'] if r['removed']]
    others=sorted(set(parent['indices']['train64'])-set(affected))
    rng=np.random.default_rng(20261081);rng.shuffle(others)
    train=sorted(affected+others[:128-len(affected)])
    stream=[]
    for epoch in range(2):
        order=np.array(train);np.random.default_rng(20261081+epoch).shuffle(order)
        stream.extend([list(map(int,order[j:j+2])) for j in range(0,len(order),2)])
    holds=sorted(train,key=lambda i:(-sum(e.type=='hold' for e in panel.targets(i)),i))
    flicks=sorted(train,key=lambda i:(-sum(e.type=='flick' for e in panel.targets(i)),i))
    empties=[i for i in parent['indices']['train64'] if not panel.targets(i)]
    fit=list(dict.fromkeys([min(affected),holds[0],flicks[0],min(empties)]))
    if len(fit)!=4:raise ValueError('Fixed four health requests must be distinct')
    priors={};input_ids={};target_ids={};used=sorted(set(train+fit+parent['indices']['dev']+parent['indices']['empty_dev']))
    for arm in ('raw','filtered'):
        counts=np.full(4,.5);seen=set();events=Counter()
        for i in train:
            occurrence=Counter()
            for e in (panel.targets(i) if arm=='raw' else filtered.targets(i)):
                sig=(panel.records[i]['chart_id'],e.onset_us,e.end_us,e.type,e.lane)
                ordinal=occurrence[sig];occurrence[sig]+=1;key=(*sig,ordinal)
                if key not in seen:counts[KINDS.index(e.type)]+=1;seen.add(key)
                events[e.type]+=1
        priors[arm]=dict(probability=(counts/counts.sum()).tolist(),counts=counts.tolist(),events=dict(events),unique_events=len(seen))
    for i in used:
        if time.monotonic()-start>300:raise TimeoutError('Prepare300s')
        rr,x,t=batch(panel,row,filtered,i,'raw','cpu')
        fr,fx,ft=batch(panel,row,filtered,i,'filtered','cpu')
        if any(not torch.equal(x[k],fx[k]) for k in x):raise ValueError('Draft public inputs changed under drag label filtering')
        if not np.array_equal(t['layout']['heads'],ft['layout']['heads']):raise ValueError('Public head support changed')
        if Counter(e for e in rr['targets'] if e.type!='drag')!=Counter(e for e in fr['targets'] if e.type!='drag'):raise ValueError('Non-drag changed')
        input_ids[str(i)]=original.input_digest(x)
        target_ids[str(i)]={a:hashlib.sha256(repr([(e.onset_us,e.end_us,e.type,e.lane) for e in r['targets']]).encode()).hexdigest() for a,r in [('raw',rr),('filtered',fr)]}
        del rr,x,t,fr,fx,ft
    # Actual completion partitions remain original, including random-group views.
    views=[]
    for spec in [s for group in parent['sample_stream'][:16] for s in group]:
        rr=row(spec['index'],spec['view'],spec['field_index'])
        fr=filtered_request_row(filtered,rr)
        if Counter(e for e in rr['targets'] if e.type!='drag')!=Counter(e for e in fr['targets'] if e.type!='drag'):raise ValueError('Partition changed non-drag')
        views.append(dict(index=spec['index'],view=spec['view'],request_equal=rr['request']==fr['request'],before=len(rr['targets']),after=len(fr['targets'])))
    write(ART/'pipeline-preflight.json',dict(status='completed',draft_inputs_exact=len(used),non_drag_exact=True,public_support_exact=True,views=views,
        source_audit='reports/local-integration-v1/data-drag-density-audit-v1.json',original_partition_before_removal=True,cloud_calls=0))
    config=dict(parent['model']);config.update(d=64,heads=4,ffn=192)
    random.seed(20261081);np.random.seed(20261081);torch.manual_seed(20261081)
    net=MaskedNativeEvents(FullEventConfig(**config))
    raw_prior=read(ROOT/parent['priors_path'])
    with torch.no_grad():
        p=float(np.mean(raw_prior['slot_active_probability']));net.active_read.bias.fill_(math.log(p/(1-p)))
        net.type_read[-1].bias.copy_(torch.tensor(np.log(priors['raw']['probability']),dtype=torch.float32))
    initial=ART/'common-initial-state.pt';torch.save(net.state_dict(),initial)
    plan=dict(id=ID,seed=20261081,model=config,parameters=sum(p.numel() for p in net.parameters()),
        initial_path=str(initial.relative_to(ROOT)),initial_sha256=sha(initial),initial_tensor_digest=weight_digest(net),
        train_indices=train,fit_indices=fit,stream=stream,eval_indices=parent['indices']['dev'],empty_indices=parent['indices']['empty_dev'],
        priors=priors,duration_prior=raw_prior['duration'],rule=RULE,input_digest=input_ids,target_digest=target_ids,
        intervention='Only original-chart high-density drag target removal;draft inputs bitwise same;common initial biases and fixed endpoint prior',
        sampling='Train enriched128windows include all32affected;other96seeded;2passes/128updates each. Not original416 distribution or C17 performance comparison.',
        objective='Unchanged C17 natural masked whole tuple NLL/IPW;all/.5/.25 mixture same random stream',
        decode='Unchanged complete2sweeps*8blocks*T1;no output filtering;allmask marginal NULL mode count diagnostic only,not joint chartMAP',
        health_updates=50,quality_updates=128,effective_batch=2,learning_rate=.0003,health_lr=.001,
        max_optimizer_updates=356,wall_s=dict(preparation=300,health_per_arm=300,quality_per_arm=900,evaluation_per_arm=450,total=3600),
        cost_policy='Small data-hypothesis probe;bounded actual stages,not original340panel extrapolation. No retry/extension/seed/threshold search.',
        research_screen=dict(type_NLL_gain=.03,class_positive_gain=.05,max_other_class_NLL_drop=.10,
            non_tap_tp_gain=3,min_components_with_gain=2,hold_joint_gain=2,tail_median_relative_reduction=.15,
            max_additional_empty_windows=2,max_additional_empty_events=3,
            interpretation='Mechanism evidence for a larger DATA test only;not product/scaling gate. Report every chain even on failure.'),
        controls='Both weights scored on BOTH identical raw and filtered dev targets;current dev removal0. Common raw-arm machine context,even slots masked;all-mask/first-member/hold prior separately.',
        predecessors=['c17-masked-native-events-v1','c16-shared-total-inventory-v1'],shared_failure_modes=['type_prior_gap','readout_empty','event_context_tail'],
        mechanism_changes={k:dict(change='Remove only dense drag training supervision;model/law/optimizer/RNG fixed across arms',prediction='Data-only change reduces '+k,
            discriminating_control='Paired same initialization;unchanged dev targets;allmask/first/self-context proper and full native outputs') for k in ['type_prior_gap','readout_empty','event_context_tail']},
        pretraining_probe_receipts=[dict(status='completed',evidence='artifacts/data-drag-density-tests.log',scope='New half-open/duplicate/non-drag/full-chart filter regression'),
            dict(status='completed',evidence=str((ART/'pipeline-preflight.json').relative_to(ROOT)),scope='Actual new data pipeline,draft inputs/support and original-partition labels')],
        execution_ready=True,execution_ready_scope='Small probe CPU/MPS parity then actual discarded fit required;not scaling',
        confirmation_scored=False,cloud_calls=0,cloud_spend_usd=0)
    if plan['parameters']>1_000_000:raise ValueError('Small probe must stay under1M')
    plan['source_sha256']={name:sha(ROOT/name) for name in set(closure('scripts/run_drag_data_probe.py'))|{'tests/test_drag_density_filter.py'}}
    write(PLAN,plan)
    for name in plan['source_sha256']:
        target=ART/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    write(ART/'preparation.json',dict(status='completed',wall_s=time.monotonic()-start,parameters=plan['parameters'],train_requests=len(train),optimizer_updates=0))
    print(dict(status='prepared',parameters=plan['parameters'],wall_s=time.monotonic()-start),flush=True)


def health(net,opt,plan,panel,row,filtered,arm,guard):
    def measure():
        vals=[]
        with torch.no_grad():
            for i in plan['fit_indices']:
                rr,x,t=batch(panel,row,filtered,i,arm);out=net(net.encode(x,t['layout'],rr['request'].start_us),empty_state())
                loss,_=tuple_loss(net,out,rr['request'],t,np.ones(256,bool),1.,plan['duration_prior']);vals.append(float(loss.cpu()))
                del rr,x,t,out,loss;guard.clear()
        return float(np.mean(vals))
    before=measure();curve=[]
    for step in range(1,51):
        net.train();opt.zero_grad(set_to_none=True);total=0.
        for i in plan['fit_indices']:
            guard.check('health_'+arm);rr,x,t=batch(panel,row,filtered,i,arm)
            out=net(net.encode(x,t['layout'],rr['request'].start_us),empty_state())
            loss,_=tuple_loss(net,out,rr['request'],t,np.ones(256,bool),1.,plan['duration_prior'])
            total+=float(loss.detach().cpu());(loss/4).backward();del rr,x,t,out,loss;guard.clear()
        norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True));opt.step()
        curve.append(dict(step=step,loss=total/4,gradient_norm=norm))
        if step%10==0:print(dict(stage='health',arm=arm,step=step,loss=total/4),flush=True)
    after=measure();return dict(before=before,after=after,ratio=after/before,passed=after<.9*before,updates=50,curve=curve)


def type_stats(detail,priors):
    records=[]
    for j,slot in enumerate(detail['active_slots']):
        records.append(dict(slot=slot,truth=detail['type_labels'][j],probability=detail['type_probability'][j],
            support=detail['support'][j],head_nll=detail['head_nll'][j]))
    return records


@torch.no_grad()
def proper(net,x,rr,t,plan,state,selected):
    out=net(net.encode(x,t['layout'],rr['request'].start_us),state)
    _,detail=tuple_loss(net,out,rr['request'],t,selected,1.,plan['duration_prior'],details=True)
    probability=out['active'][0].sigmoid().cpu().numpy();n=sum(t['state']['status']==2)
    tails=[]
    for v in detail['tail_nll']:
        p=duration_probability(plan['duration_prior'],v['head'],v['end'],rr['request'].duration_us,rr['request'].copy_times)
        tails.append(dict(**v,prior_nll=-math.log(p)))
    return dict(types=type_stats(detail,plan['priors']),tails=tails,
        occupancy=dict(expected=float(probability.sum()),marginal_mode_active=int((probability>=.5).sum()),
            true_events=int(n),expected_on_NULL_target_slots=float(probability[n:].sum()),
            positive_mean_probability=float(probability[:n].mean()) if n else None,
            true_slots_below_half=int((probability[:n]<.5).sum()),binary_nll=detail['binary_nll'],
            probability=probability.tolist()),support_failures=0,
        scope='True-anchor conditional proper,not free generation. No prefix support exclusions in C17 law.')


def hold_errors(pred,truth):
    pp=[e for e in pred if e.type=='hold'];tt=[e for e in truth if e.type=='hold'];usedp=set();usedt=set();errors=[]
    pairs=sorted((abs(p.onset_us-t.onset_us),i,j) for i,p in enumerate(pp) for j,t in enumerate(tt) if abs(p.onset_us-t.onset_us)<=50_000)
    for _,i,j in pairs:
        if i not in usedp and j not in usedt:
            usedp.add(i);usedt.add(j);errors.append(abs(pp[i].end_us-tt[j].end_us)/1000)
    return dict(predicted=len(pp),truth=len(tt),head_matches=len(errors),tail_abs_ms=errors,
        median_tail_ms=float(np.median(errors)) if errors else None,joint_50ms=sum(v<=50 for v in errors))


def evaluate(net,plan,panel,row,filtered,arm,guard):
    records=[];raw_path=ART/'raw-outcomes.json'
    raw_records={r['index']:r for r in read(raw_path)} if arm=='filtered' else {}
    indices=plan['eval_indices']+plan['empty_indices']
    # Persist ALL free outcomes before consulting any target-conditioned head.
    for i in indices:
        guard.check('free_'+arm);rr,x,t=batch(panel,row,filtered,i,arm)
        prediction=decode_masked(net,x,rr['request'],t['layout'],plan['duration_prior'],seed=plan['seed'],
            public_id=(i,rr['request'].start_us,rr['request'].end_us),deadline=guard.start+guard.wall_s,clear_callback=guard.clear)
        events=[RhythmEvent(**e) for e in prediction['events']]
        truth=panel.targets(i);clean=filtered.targets(i)
        if not prediction['complete']:raise ValueError('Complete native sampler failed '+str(prediction['failure']))
        records.append(dict(index=i,component=panel.records[i]['component'],empty=i in plan['empty_indices'],prediction=prediction,
            target_raw=[asdict(e) for e in truth],target_filtered=[asdict(e) for e in clean],
            scores_raw=score(events,truth),scores_filtered=score(events,clean),hold_raw=hold_errors(events,truth),hold_filtered=hold_errors(events,clean)))
        write(ART/f'{arm}-outcomes.json',records);del rr,x,t,events;guard.clear()
    for record in records:
        i=record['index'];guard.check('proper_'+arm);rr,x,t=batch(panel,row,filtered,i,arm)
        record['all_mask']=proper(net,x,rr,t,plan,empty_state(),np.ones(256,bool));guard.clear()
        baseline=record if arm=='raw' else raw_records[i]
        machine=[RhythmEvent(**e) for e in baseline['prediction']['events']]
        context=target_state(machine,rr['request'],t['layout']);selected=np.arange(256)%2==0
        context['status'][selected]=0
        for key in ('head','kind','end'):context[key][selected]=0
        record['common_machine_context']=proper(net,x,rr,t,plan,context,selected)
        record['common_context_source']='raw control free output;mask all even slots;never true prefix'
        write(ART/f'{arm}-outcomes.json',records);del rr,x,t;guard.clear()
    return records


def run():
    runtime();plan=read(PLAN)
    for name,expected in plan['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen source drift '+name)
    if REPORT.exists():raise FileExistsError('No repeat probe')
    panel,row=load_data();filtered=DragFilteredPanel(panel);started=time.monotonic()
    report=dict(id=ID,status='running',optimizer_updates=0,stages={},cloud_calls=0,cloud_spend_usd=0,confirmation_scored=False)
    write(REPORT,report)
    def timeout(*_):raise TimeoutError('Global probe3600s')
    old=signal.getsignal(signal.SIGALRM);signal.signal(signal.SIGALRM,timeout);signal.alarm(3595)
    net=None;opt=None;guard=None
    try:
        # Actual small-model parity: no optimizer, weights remain initial.
        net=fresh(plan,'cpu');gpu=fresh(plan);net.eval();gpu.eval();i=plan['fit_indices'][0]
        parity={}
        for arm in ('raw','filtered'):
            vals=[];grads=[]
            for device,m in [('cpu',net),('mps',gpu)]:
                m.zero_grad(set_to_none=True);rr,x,t=batch(panel,row,filtered,i,arm,device)
                out=m(m.encode(x,t['layout'],rr['request'].start_us),empty_state())
                loss,_=tuple_loss(m,out,rr['request'],t,np.ones(256,bool),1.,plan['duration_prior']);loss.backward()
                vals.append(float(loss.detach().cpu()));grads.append({n:p.grad.detach().cpu().clone() for n,p in m.named_parameters() if p.grad is not None})
            error=max(float((grads[0][n]-grads[1][n]).abs().max()) for n in grads[0]);difference=abs(vals[0]-vals[1])
            if max(error,difference)>1e-4:raise ValueError('Small data CPU/MPS parity')
            parity[arm]=dict(loss_cpu=vals[0],loss_mps=vals[1],loss_difference=difference,gradient_max_difference=error)
        del net,gpu,opt,rr,x,t,out,loss,grads;net=None;opt=None
        torch.mps.empty_cache();write(ART/'numerical-preflight.json',dict(status='passed',parity=parity,optimizer_updates=0))
        for arm in ('raw','filtered'):
            stage='health-'+arm;report['active_stage']=stage;write(REPORT,report)
            guard=ManagedMPS(ROOT,ART,300);net=fresh(plan);opt=torch.optim.AdamW(net.parameters(),lr=plan['health_lr'],weight_decay=.01)
            value=health(net,opt,plan,panel,row,filtered,arm,guard);value.update(runtime=guard.close(),wall_s=time.monotonic()-guard.start);guard=None
            report['optimizer_updates']+=value['updates'];report['stages'][stage]=value;write(REPORT,report)
            del net,opt;net=None;opt=None;torch.mps.empty_cache()
            if not value['passed']:raise ValueError('Actual fixed batch fit failed;do not extend')
        for arm in ('raw','filtered'):
            stage='quality-'+arm;report['active_stage']=stage;write(REPORT,report)
            guard=ManagedMPS(ROOT,ART,900);net=fresh(plan);opt=torch.optim.AdamW(net.parameters(),lr=plan['learning_rate'],weight_decay=.01)
            curve=[];exposure=Counter();affected=0
            for step,indices in enumerate(plan['stream'],1):
                net.train();opt.zero_grad(set_to_none=True);total=0.
                for micro,i in enumerate(indices):
                    guard.check(stage);rr,x,t=batch(panel,row,filtered,i,arm)
                    state,selected,rate=corruption(t['state'],stable_rng(plan['seed'],step,micro,'mask'))
                    out=net(net.encode(x,t['layout'],rr['request'].start_us),state)
                    loss,_=tuple_loss(net,out,rr['request'],t,selected,rate,plan['duration_prior'])
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite tuple loss')
                    total+=float(loss.detach().cpu());exposure.update(e.type for e in rr['targets'])
                    affected+=len(panel.targets(i))!=len(filtered.targets(i));(loss/len(indices)).backward()
                    del rr,x,t,state,out,loss;guard.clear()
                norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True));opt.step();report['optimizer_updates']+=1
                curve.append(dict(step=step,loss=total/len(indices),gradient_norm=norm))
                if step%16==0:print(dict(stage=stage,step=step,loss=total/len(indices),wall_s=time.monotonic()-guard.start),flush=True)
            path=ART/(arm+'-quality-0128.pt');torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),updates=128,plan_sha256=sha(PLAN)),path)
            report['stages'][stage]=dict(updates=128,initial_tensor_digest=plan['initial_tensor_digest'],checkpoint_sha256=sha(path),
                curve=curve,exposure=dict(exposure),affected_exposures=affected,wall_s=time.monotonic()-guard.start,runtime=guard.close());guard=None
            write(REPORT,report);del net,opt;net=None;opt=None;torch.mps.empty_cache()
        for arm in ('raw','filtered'):
            stage='evaluation-'+arm;report['active_stage']=stage;write(REPORT,report)
            net=fresh(plan);net.load_state_dict(torch.load(ART/(arm+'-quality-0128.pt'),map_location='cpu',weights_only=False)['model']);net.eval()
            guard=ManagedMPS(ROOT,ART,450);records=evaluate(net,plan,panel,row,filtered,arm,guard)
            report['stages'][stage]=dict(free_completed=len(records),allmask_proper=len(records),common_machine_context_proper=len(records),
                wall_s=time.monotonic()-guard.start,runtime=guard.close());guard=None
            write(REPORT,report);del net,records;net=None;torch.mps.empty_cache()
        report['status']='completed_pending_data_mechanism_review'
    except BaseException as exc:
        report.update(status='execution_failed_pending_data_mechanism_review',failure=dict(type=type(exc).__name__,error=str(exc)))
        if net is not None:
            torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict() if opt else None,updates=report['optimizer_updates'],stage=report.get('active_stage')),ART/'failure-state.pt')
        raise
    finally:
        if guard is not None:report['failure_runtime']=guard.close()
        report['wall_s']=time.monotonic()-started;write(REPORT,report)
        signal.alarm(0);signal.signal(signal.SIGALRM,old)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run'));args=parser.parse_args()
    prepare() if args.action=='prepare' else run()
