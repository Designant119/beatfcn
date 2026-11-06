"""Opt-in, bounded local engineering screen; never launches cloud compute."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from copy import deepcopy
import json
import math
import signal
import time

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_typed_canvas import LocalTypedCanvas, canvas_loss
from beatfcn.model.local_song_rollout import canonical, independent_legality
from beatfcn.model.evaluation import score_events
from beatfcn.train.local_request_panel import event_request
from beatfcn.train.local_bundle import make_row, prefix_row
from beatfcn.train.local_typed_canvas import encode_canvas, decode_target, canvas_batch, decode_canvas
from beatfcn.train.reference_completion import partition
from scripts.run_local_bundle import ROOT, configure, panel_for, sha, write, log


PLAN=ROOT/'bench/rhythm-v2/local-typed-canvas-v1.json'
REPORT=ROOT/'reports/local_typed_canvas_v1.json'
ART=ROOT/'artifacts/local-typed-canvas-v1'
VIEWS=('draft','single_condition','all_conditions','random_groups','middle','continuation')
LIVE={}


def journal(text):
    with (ROOT/'EXPERIMENT_LOG.md').open('a') as f:f.write('\n\nlocal-typed-canvas-v1 '+text+'\n')


def ledger(report):
    path=ROOT/'reports/rhythm_v2_budget.json'; data=json.loads(path.read_text())
    data['local_cpu_diagnostics']['local-typed-canvas-v1']={
        'status':report['status'],'report':str(REPORT.relative_to(ROOT)),'plan':str(PLAN.relative_to(ROOT)),
        'device':'mps','updates':report['updates'],'health_updates':report['health_updates'],
        'wall_s':report.get('wall_s',{}),'new_cloud_spend_usd':0,'new_cloud_compute_calls':0,
        'purpose':'Engineering resource-screen, not product release acceptance; legacy counters unchanged'}
    write(path,data)


def save(report):
    write(REPORT,report);ledger(report)


def fresh(plan):
    torch.manual_seed(plan['seed'])
    return LocalTypedCanvas(FullEventConfig(**plan['model']),density_reference=plan['density_reference'])


def check_sources(plan):
    for path,expected in plan['source_sha256'].items():
        if sha(ROOT/path)!=expected:raise ValueError('Frozen source changed: '+path)


def prepare():
    if PLAN.exists() or REPORT.exists():raise FileExistsError('Preserve existing experiment')
    prior=json.loads((ROOT/'bench/rhythm-v2/local-bundle-v1.json').read_text())
    plan={k:deepcopy(prior[k]) for k in ('seed','tape','features','indices','chart_ids','data_summary','health_indices','replay_indices','schedule')}
    plan.update(id='local-typed-canvas-v1',steps=900,health_steps=100,effective_batch=2,
        model=asdict(FullEventConfig(d=128,heads=4,kv_heads=2,ffn=384,encoder_layers=1,decoder_layers=0,dropout=0.,mel_bins=128)),
        health_wall_s=600,train_wall_s=2400,evaluation_wall_s=600,learning_rate=.001,minimum_lr=.0001,warmup=50,
        cloud_compute_limit=0,comparison='Immutable local-bundle-v1 bundle1000; whole-recipe engineering comparison, fewer new quality updates; no causal/equal-compute claim',
        authorization='User: implement canvas first; local purpose is identifying candidates worth bounded resource escalation, not release readiness',
        gate=dict(full_gain=.03,onset_gain=.05,minimum_improved_songs=5,maximum_full_drop_if_timing_route=.01,
                  density_ratio=[.5,1.75],minimum_legal_nonempty_drafts=14),
        hypotheses='Parallel physical addresses and per-type counts remove recursive time/type feedback and improve held-out events enough to justify a bounded scale pilot',
        inference='No token sampling loop, no NMS/min-gap/type quotas/count clipping; explicit count/copy/event/serialization resource failures',
        local_resource_contract=dict(frames=200,cell_us=40000,per_type_slots=16,per_frame_authorized_copies=16),
        release_acceptance=False)
    panel=panel_for(plan); audit={}; total=0;seconds=0
    for split,indices in plan['indices'].items():
        n=0;max_count=0;copies=0
        for index in indices:
            original=panel.targets(index);req=event_request(panel.records[index]['request'])
            if split=='train':total+=len(original);seconds+=(req.end_us-req.start_us)/1e6
            for view in VIEWS:
                request,events=req,original
                if view in ('random_groups','middle','continuation'):
                    request,_,events,_=partition(original,req,view,f"{panel.records[index]['chart_id']}:{req.start_us}",plan['seed'])
                encoded=encode_canvas(events,request)
                if decode_target(encoded,request)!=sorted(events):raise ValueError('Lossless multiset roundtrip failed')
                max_count=max(max_count,int(encoded['count'].max()));copies=max(copies,len(request.copy_times));n+=1
        audit[split]=dict(request_views=n,lossless=True,maximum_per_cell_per_type=max_count,maximum_request_copy_times=copies)
    plan['density_reference']=total/seconds
    plan['empty_health_index']=next(i for i in plan['indices']['train'] if panel.records[i]['event_count']==0)
    model=fresh(plan);params=sum(p.numel() for p in model.parameters())
    if params>3_000_000:raise ValueError('Parameter resource cap')
    paths=set(prior['source_sha256'])|set(prior['target_sha256'])|{
        'beatfcn/model/local_typed_canvas.py','beatfcn/train/local_typed_canvas.py',
        'scripts/run_local_typed_canvas.py','design/LOCAL_TYPED_CANVAS_v1.md'}
    plan['source_sha256']={p:sha(ROOT/p) for p in paths};plan['frozen_at_unix']=time.time()
    write(PLAN,plan)
    report=dict(id=plan['id'],status='prepared',complete=False,promising_for_bounded_scale_pilot=False,
        cloud_compute_calls=0,updates=0,health_updates=0,parameters=params,representation_audit=audit,
        curve=[],evaluations={},failures=[],wall_s={},data_summary=plan['data_summary'])
    save(report);journal(f'准备完成：{params}可训练参数，{sum(a["request_views"] for a in audit.values())}真实请求视图无损往返；密度参照只取训练。100健康+900新训，旧bundle仅冻结工程对照；按用户纠正使用扩算筛选门。')
    log(dict(stage='prepared',parameters=params,audit=audit))


def checkpoint(model,opt,plan,stage,step):
    path=ART/f'{stage}-{step:04d}.pt';path.parent.mkdir(exist_ok=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),config=plan['model'],
                    stage=stage,updates=step,plan_sha256=sha(PLAN),cpu_rng=torch.get_rng_state(),
                    mps_rng=torch.mps.get_rng_state()),path)


def score(pred,target):
    pp=canonical(pred,'phira');tt=canonical(target,'phira');s=score_events(pp,tt,'phira')
    for ms in (10,50):
        tp=s[f'note_{ms}ms']['tp']-s[f'hold_head_{ms}ms']['tp']+s[f'hold_joint_{ms}ms']['tp']
        s[f'full_note_f1_{ms}ms']=2*tp/max(len(pred)+len(target),1)
    s['full_note_f1']=s['full_note_f1_50ms'];s['onset_f1']=s['onset_50ms']['f1'] or 0.
    s['per_type']={}
    for k in ('tap','hold','flick','drag'):
        v=score_events([e for e in pp if e.type==k],[e for e in tt if e.type==k],'phira')
        s['per_type'][k]=v['hold_joint_50ms' if k=='hold' else 'note_50ms']
    return s


def generate(model,panel,indices,*,view='draft',bands=None,mask_reference=False,deadline=None):
    records=[]
    for index in indices:
        if deadline is not None and time.monotonic()>=deadline:
            records.append(dict(index=index,attempted=False,reason='evaluation_wall_limit'));continue
        row=make_row(panel,index,view,manual_bands=bands)
        inputs,_=canvas_batch([row],'mps',mask_reference=mask_reference)
        result=decode_canvas(model,inputs,[row['request']],deadline=min(deadline or float('inf'),time.monotonic()+45))[0]
        pred=[RhythmEvent(**e) for e in result['events']];gaps=np.diff(sorted(set(e.onset_us for e in pred)))
        records.append(dict(index=index,attempted=True,chart_id=panel.records[index]['chart_id'],view=view,
            request=asdict(row['request']),prediction=result,target=[asdict(e) for e in row['targets']],
            visible=[asdict(e) for e in row['visible']],mask_reference=mask_reference,scores=score(pred,row['targets']),
            legality=independent_legality(pred,'phira',row['request'].duration_us),
            types=dict(Counter(e.type for e in pred)),target_types=dict(Counter(e.type for e in row['targets'])),
            positive_gap_median_ms=float(np.median(gaps)/1000) if len(gaps) else None,
            positive_gaps_under_10ms=int((gaps<10000).sum()),positive_gaps=len(gaps),
            visible_preservation='External verbatim multiset merge; not learned copying'))
    return records


def tensor_stats(values):
    out={}
    for k,v in values.items():
        v=v.float();out[k]=dict(shape=list(v.shape),finite=bool(torch.isfinite(v).all()),
            minimum=float(v.min()),maximum=float(v.max()),mean=float(v.mean()),std=float(v.std(unbiased=False)))
        if not out[k]['finite']:raise ValueError('Nonfinite tensor '+k)
    return out


def health(plan,report):
    if report['status']!='prepared':raise ValueError('Health requires unused prepared run')
    started=time.monotonic();panel=panel_for(plan)
    rows=[make_row(panel,i,v) for i,v in zip(plan['health_indices'],('draft','single_condition','random_groups','middle'))]
    rows.append(make_row(panel,plan['empty_health_index']))
    batches=[canvas_batch([r]) for r in rows];model=fresh(plan).eval()
    audit=dict(inputs=tensor_stats(batches[0][0]),targets=tensor_stats(batches[0][1]))
    with torch.no_grad():cpu={k:v.cpu() for k,v in model(batches[0][0]).items()}
    model.to('mps');batches=[({k:v.to('mps') for k,v in x.items()},{k:v.to('mps') for k,v in y.items()}) for x,y in batches]
    with torch.no_grad():gpu=model(batches[0][0])
    parity={k:float((v.cpu()-cpu[k]).abs().max()) for k,v in gpu.items()}
    if max(parity.values())>1e-4:raise ValueError('CPU/MPS parity failed')
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=0)
    LIVE.update(model=model,opt=opt,stage='health',step=0)
    report.update(status='health_running',health_tensor_audit=audit,cpu_mps_parity=parity);save(report)
    for step in range(1,plan['health_steps']+1):
        if time.monotonic()-started>plan['health_wall_s']:raise TimeoutError('Health wall limit')
        model.train();opt.zero_grad(set_to_none=True);total=0.;parts=Counter()
        for x,y in batches:
            loss,terms=canvas_loss(model(x),y,x);(loss/len(batches)).backward();total+=float(loss.detach())/len(batches)
            parts.update({k:float(v.detach())/len(batches) for k,v in terms.items()})
        grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step()
        LIVE['step']=step
        report['health_updates']=step;report['wall_s']['health']=time.monotonic()-started
        if step==1 or step%25==0:
            point=dict(step=step,loss=total,parts=dict(parts),grad=grad,wall_s=report['wall_s']['health'])
            report.setdefault('health_curve',[]).append(point);save(report);log(dict(stage='health',**point))
    checkpoint(model,opt,plan,'health',step)
    final=[];raw=[]
    for row,(x,y) in zip(rows,batches):
        prediction=decode_canvas(model,x,[row['request']],deadline=started+plan['health_wall_s'])[0]
        pred=[RhythmEvent(**e) for e in prediction['events']];s=score(pred,row['targets'])
        legal=independent_legality(pred,'phira',row['request'].duration_us)
        passed=prediction['complete'] and not legal
        if row['targets']:
            passed=passed and s['full_note_f1']>=.95 and (s['onset_10ms']['f1'] or 0)>=.95
            passed=passed and all(v['f1']>=.90 for v in s['per_type'].values() if v['true']>0)
        else:passed=passed and len(pred)==0
        final.append(dict(index=row['index'],view=row['view'],passed=bool(passed),predicted=len(pred),
                          target=len(row['targets']),full_note_f1=s['full_note_f1'],onset_10ms=s['onset_10ms'],per_type=s['per_type']))
        raw.append(dict(index=row['index'],prediction=prediction,target=[asdict(e) for e in row['targets']],scores=s,legality=legal))
    write(ART/'health-predictions.json',raw)
    report['health']=dict(passed=all(x['passed'] for x in final),requests=final,quality_weights_reused=False)
    report['status']='health_passed' if report['health']['passed'] else 'health_failed'
    report['complete']=not report['health']['passed'];report['wall_s']['health']=time.monotonic()-started;save(report)
    journal(f'真实100步健康终态{report["status"]}，逐请求{[(x["index"],x["full_note_f1"],x["passed"]) for x in final]}；wall={report["wall_s"]["health"]:.3f}s。健康权重不用于质量。')
    log(dict(stage=report['status'],health=report['health']))


def train(plan,report):
    if report['status']!='health_passed' or report['updates']:raise ValueError('Fresh quality requires actual passed health')
    panel=panel_for(plan);model=fresh(plan).to('mps');opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.)
    LIVE.update(model=model,opt=opt,stage='quality',step=0)
    started=time.monotonic();deadline=started+plan['train_wall_s'];buffer={}
    report['status']='training';save(report)
    for step in range(1,plan['steps']+1):
        if time.monotonic()>=deadline:raise TimeoutError('Quality wall limit')
        lr=.001*step/50 if step<=50 else .0001+.0009*(1+math.cos(math.pi*(step-50)/(plan['steps']-50)))/2
        for group in opt.param_groups:group['lr']=lr
        schedule=plan['schedule'][step-1];rows=[]
        for j,spec in enumerate(schedule):
            kwargs={};view=spec['view']
            if j==0 and step>200 and step%10==0 and buffer:
                index=plan['replay_indices'][(step//10)%len(plan['replay_indices'])]
                if index in buffer:
                    spec={**spec,'index':index};view='continuation';kwargs['generated_history']=buffer[index]
            rows.append(make_row(panel,spec['index'],view,field_index=spec['field_index'],**kwargs))
        model.train();opt.zero_grad(set_to_none=True);total=0.;terms=Counter()
        for row in rows:
            x,y=canvas_batch([row],'mps');loss,pieces=canvas_loss(model(x),y,x)
            (loss/len(rows)).backward();total+=float(loss.detach())/len(rows)
            terms.update({k:float(v.detach())/len(rows) for k,v in pieces.items()})
        grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step()
        LIVE['step']=step
        report['updates']=step;report['wall_s']['training']=time.monotonic()-started
        if step==1 or step%25==0:
            point=dict(step=step,loss=total,parts=dict(terms),lr=lr,grad=grad,wall_s=report['wall_s']['training'])
            report['curve'].append(point);save(report);log(dict(stage='train',**point))
        if step in (200,500,800):
            history=[]
            for i in plan['replay_indices']:
                row=prefix_row(panel,i);row['targets']=[]
                x,_=canvas_batch([row],'mps');p=decode_canvas(model,x,[row['request']],deadline=deadline)[0]
                history.append(dict(index=i,prediction=p))
                if p['complete']:buffer[i]=p['events']
            write(ART/f'replay-{step:04d}.json',history)
        if step in (250,500,900):
            checkpoint(model,opt,plan,'quality',step)
            if step!=900:
                generated=generate(model,panel,plan['indices']['val'][:4],deadline=deadline)
                write(ART/f'draft-{step:04d}.json',generated)
                report['evaluations'][str(step)]={'draft_full_f1':float(np.mean([x['scores']['full_note_f1'] for x in generated if x['attempted']]))}
                save(report)
    report['status']='trained';report['wall_s']['training']=time.monotonic()-started;save(report)
    journal(f'新鲜初始化正式训练{report["updates"]}更新完成，wall={report["wall_s"]["training"]:.3f}s；未用健康权重，云0。')


def evaluate(plan,report):
    if report['status']!='trained':raise ValueError('Final evaluation requires completed quality')
    saved=torch.load(ART/'quality-0900.pt',map_location='cpu',weights_only=False)
    if saved['plan_sha256']!=sha(PLAN) or saved['updates']!=900:raise ValueError('Checkpoint identity mismatch')
    model=fresh(plan);model.load_state_dict(saved['model']);model.to('mps').eval();panel=panel_for(plan)
    started=time.monotonic();deadline=started+plan['evaluation_wall_s'];report['status']='evaluating';save(report)
    cases={'draft':(plan['indices']['val'],'draft',None,False),
           'completion':(plan['indices']['val'][::2],'middle',None,False),
           'density_low':(plan['indices']['val'][::4],'draft',{'density_notes_per_second':[2.,3.]},False),
           'density_high':(plan['indices']['val'][::4],'draft',{'density_notes_per_second':[6.,8.]},False),
           'reference_masked':(plan['indices']['val'][::2],'middle',None,True)}
    for task,(ids,view,bands,mask) in cases.items():
        path=ART/f'{task}-final.json'
        if path.exists():raise FileExistsError('Preserve previous evaluation')
        rows=generate(model,panel,ids,view=view,bands=bands,mask_reference=mask,deadline=deadline);write(path,rows)
        report['evaluations'][task]={'attempted':sum(x['attempted'] for x in rows),
            'complete':sum(x.get('prediction',{}).get('complete',False) for x in rows),
            'full_note_f1':float(np.mean([x['scores']['full_note_f1'] for x in rows if x['attempted']]))}
        save(report);log(dict(stage='evaluate',task=task,**report['evaluations'][task]))
    report['status']='evaluated';report['wall_s']['evaluation']=time.monotonic()-started;save(report)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','health','train','evaluate']);args=parser.parse_args()
    if args.stage=='prepare':prepare();return
    plan=json.loads(PLAN.read_text());report=json.loads(REPORT.read_text());check_sources(plan);configure()
    def stop(signum,_):raise InterruptedError(f'Signal {signum}')
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:{'health':health,'train':train,'evaluate':evaluate}[args.stage](plan,report)
    except BaseException as exc:
        if LIVE:
            checkpoint(LIVE['model'],LIVE['opt'],plan,LIVE['stage']+'-stopped',LIVE['step'])
        report['status']=args.stage+'_stopped';report['failures'].append(f'{type(exc).__name__}: {exc}');save(report)
        journal(f'{report["status"]}，health_updates={report["health_updates"]}，quality_updates={report["updates"]}：{exc}。不自动续训。')
        raise


if __name__=='__main__':main()
