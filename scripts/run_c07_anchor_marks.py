"""C07 event-conditioned mark learning; fixed machine anchors remain external."""
from collections import Counter
from dataclasses import asdict
from functools import lru_cache
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from beatfcn.model.anchor_mark_chain import AnchorMarkChain,KINDS,mark_loss,sampled_marks
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import encode_events
from beatfcn.model.sample import match_f1
from beatfcn.train.anchor_marks import anchor_inputs,teacher_example
from scripts.run_c07_count_ancestral import specs,VIEWS
from scripts.run_local_evolution import load_data
from scripts.run_local_portfolio_scale import sha,write,weight_digest
from scripts.run_local_typed_canvas import score
from scripts.review_frontend_canvas_comparison import metrics
from scripts.validate_mechanism_reviews import validate_successor,load_registry


ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'bench/rhythm-v2/c07-anchor-conditioned-mark-chain-v1.json'
ART=ROOT/'artifacts/c07-anchor-conditioned-mark-chain-v1'
FOLDER=ROOT/'reports/local-evolution-v3'
REPORT=FOLDER/'anchor-marks.json'


def read(path):return json.loads((ROOT/path).read_text())


def state(**values):
    p=FOLDER/'state.json';old=read(str(p.relative_to(ROOT))) if p.exists() else {}
    write(p,{**old,**values})


def prepare():
    if PLAN.exists():raise FileExistsError('Do not replace a frozen candidate')
    c=read('bench/rhythm-v2/c06-reset-unified-count-law-v1.json')
    files=set(c['sources'])|{'scripts/run_c07_anchor_marks.py','beatfcn/model/anchor_mark_chain.py',
        'beatfcn/train/anchor_marks.py','tests/test_anchor_mark_chain.py','beatfcn/model/canvas_diffusion.py',
        'scripts/run_local_evolution.py','scripts/run_local_typed_canvas.py','beatfcn/train/local_bundle.py',
        'beatfcn/train/local_request_panel.py','beatfcn/train/mert_canvas_features.py','scripts/run_c07_count_ancestral.py'}
    gate=dict(typed50=.363897,full50=.336075,non_tap_pooled=.066753,components=9,
        oracle_prior_nll_gain=.10,audio_nll_gain=.05,reference_nll_gain=.05)
    p=dict(id='c07-anchor-conditioned-mark-chain-v1',experiment_class='local_mechanism',seed=20261032,
        predecessors=['c06-reset-unified-count-law-v1','c07-count-ancestral-audit-v1'],single_intervention='Replace independent sparse typed-cell counts with an event-conditioned mark/type-and-tail chain; machine onset multiset fixed externally. Primary dimension DataRepresentation; complete factorization recipe, not isolated GRU/width causality.',
        shared_failure_modes=['sparse_positive_type_supervision','diffuse_independent_count_law','absolute_tail_clock_alias'],
        mechanism_changes=dict(sparse_positive_type_supervision=dict(change='Four mutually exclusive type classes only at supplied event nodes, generated type prefix and legitimate reference',prediction='Types beat training prior and improve on fixed machine anchors',discriminating_control='Oracle and machine tracks; local400ms/song-only/full-audio foreign input and reference-mask/mismatch'),
            diffuse_independent_count_law=dict(change='No cellwise count/existence law or global selection; every supplied node has one exclusive learned mark',prediction='Avoid trading diffuse independent count mass for low-precision off-anchor objects',discriminating_control='Exactly fixed machine onset multiset, true-input versus foreign-audio/reference and training prior'),
            absolute_tail_clock_alias=dict(change='Retain head-relative native-ms endpoint clock, not proposed unconditioned absolute tail labels',prediction='Translated clock has identical delta target/input and emitted endpoint shifts exactly',discriminating_control='COPY/multiset/cross-window roundtrip and brute-force legal partition; no clipping')),
        parent_deviation='Sol B absolute-tail proposal repaired before training: d=end_ms-floor(head_us/1000); end=(floor(head_us/1000)+d)*1000. Full legal support, same601/50/40 fields.',
        indices=c['indices'],sample_stream=c['sample_stream'][:1000],health_indices=c['health_indices'],empty_health_index=c['empty_health_index'],
        seeds=[2026102901,2026102902,2026102903,2026102904],steps=1000,health_steps=100,
        optimizer=dict(lr=.001,minimum_lr=.0001,warmup=50,schedule_steps=1000,weight_decay=0,clip=1.,effective_batch=2),
        model=dict(d=128,max_parameters=2000000),gate=gate,
        health_gate=dict(type_nll_ratio=.1,tail_nll_ratio=.1,finite_complete=True,parity_max=1e-4),
        wall_s=dict(preflight=300,health=600,quality=1800,evaluation=900,process=3600),
        memory=dict(driver_gib=8,recommended_fraction=.7),sources={f:sha(ROOT/f) for f in sorted(files)},
        cloud_compute_calls=0,execution_ready=False,pretraining_probe_receipts=[],
        baseline='artifacts/c06-reset-unified-count-law-v1/unified',baseline_report='reports/local-evolution-v2/unified-count.json',
        baseline_plan_sha256=sha(ROOT/'bench/rhythm-v2/c06-reset-unified-count-law-v1.json'),
        failure_policy='No seed/change of gate/retry/resume/longer training. Any stop preserves checkpoint and requires mechanism review.',
        normalization='Sum joint type+legal relative-tail NLL over events, divide by total events of the two original requests; empty pairs have no effective learning and are reported separately.',
        control_policy='Sorted-component cyclic audio donor, fixed anchors and random uniforms. Wrong reference requires exact visible-content row count and another component; unavailable controls retained as failures and block reference claim.')
    write(PLAN,p)
    for f in files:
        dst=ART/'source'/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dst)
    shutil.copy2(PLAN,ART/'source/initial-plan.json')
    state(status='prepared',active_candidate=p['id'],plan=str(PLAN.relative_to(ROOT)),report=str(REPORT.relative_to(ROOT)),cloud_compute_calls=0)


def guard(plan,started,stage_start,stage):
    torch.mps.synchronize();a=torch.mps.driver_allocated_memory();r=torch.mps.recommended_max_memory()
    if a>min(plan['memory']['driver_gib']*2**30,plan['memory']['recommended_fraction']*r):raise MemoryError('MPS process cap')
    if time.monotonic()-started>plan['wall_s']['process'] or time.monotonic()-stage_start>plan['wall_s'][stage]:raise TimeoutError(stage+' wall cap')
    return dict(allocated=torch.mps.current_allocated_memory(),driver=a,recommended=r)


def device(x,name='mps'):return {k:v.to(name) for k,v in x.items()}


def fresh(plan,name='mps'):
    torch.manual_seed(plan['seed']);return AnchorMarkChain(plan['model']['d']).to(name)


def digest_example(x,y,heads):
    out={k:dict(shape=list(v.shape),sha256=hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()) for k,v in {**x,**{'target_'+k:v for k,v in y.items()}}.items()}
    return dict(tensors=out,heads_sha256=hashlib.sha256(np.asarray(heads,np.int64).tobytes()).hexdigest(),events=len(heads))


@torch.no_grad()
def diagnostic(model,batches):
    model.eval();types=tails=0.;n=h=0
    for x,y,_ in batches:
        _,p=mark_loss(model,device(x),device(y));types+=float(p['type_sum']);tails+=float(p['tail_sum']);n+=p['events'];h+=p['holds']
    return dict(type_nll=types/max(n,1),tail_nll=tails/max(h,1),events=n,holds=h)


@torch.no_grad()
def teacher_stats(model,x,y,prior):
    x,y=device(x),device(y);out=model.teacher(x,y['kind']);labels=y['kind'][0]
    if not len(labels):return dict(events=0,type_nll=0.,prior_nll=0.,per_type={},confusion={})
    p=out['kind'][0].softmax(-1);nll=F.cross_entropy(out['kind'][0],labels,reduction='none');brier=((p-F.one_hot(labels,4))**2).sum(-1)
    return dict(events=len(labels),type_nll=float(nll.mean()),prior_nll=float(-np.log(prior[labels.cpu().numpy()]).mean()),
        per_type={k:dict(n=int((labels==j).sum()),nll=float(nll[labels==j].mean()) if bool((labels==j).any()) else None,
            brier=float(brier[labels==j].mean()) if bool((labels==j).any()) else None) for j,k in enumerate(KINDS)},
        confusion=Counter((int(a),int(b)) for a,b in zip(labels.cpu(),p.argmax(-1).cpu())))


def record(pred,rr,index,component,view,shifts):
    events=[RhythmEvent(**v) for v in pred];events.sort();failure=None;tokens=[]
    try:tokens=encode_events(events,rr['request']);assert len(tokens)<=rr['request'].max_tokens
    except (ValueError,AssertionError) as exc:failure=str(exc) or 'Serialization token limit'
    scores=score(events,rr['targets']);s=rr['request'].start_us;w=rr['request'].end_us-s
    if failure:scores['full_note_f1']=scores['onset_f1']=0.;scores['onset_10ms']['f1']=0.
    null=[match_f1([s+(e.onset_us-s+d)%w for e in events],[e.onset_us for e in rr['targets']],10000)[3] for d in shifts]
    return dict(index=index,component=component,view=view,request=asdict(rr['request']),prediction=dict(complete=failure is None,failure=failure,events=[asdict(e) for e in events],tokens=tokens),target=[asdict(e) for e in rr['targets']],scores=scores,shifted_onset10=null)


def evaluate(model,plan,panel,row,prior,deadline):
    tasks=specs(plan);baseline={task:{r['index']:r for r in read(plan['baseline']+f'/quality-1000-{task}.json')} for task in VIEWS}
    components=sorted({panel.records[i]['component'] for i in plan['indices']['dev']})
    donor_ids={c:next(i for i in plan['indices']['dev'] if panel.records[i]['component']==components[(j+1)%len(components)]) for j,c in enumerate(components)}
    summaries={};oracle=[];control_failures=[]
    saved={};base_plan=read('bench/rhythm-v2/c06-reset-unified-count-law-v1.json')
    for condition in ('real','local_roll','song_foreign','audio_foreign','reference_foreign'):
        summaries[condition]={}
        for task,items in tasks.items():
            if condition!='real' and task not in (('completion',) if condition=='reference_foreign' else ('draft',)):continue
            records=[[] for _ in plan['seeds']]
            for index,view,bands,masked in items:
                if time.monotonic()>deadline:raise TimeoutError('Mark evaluation')
                rr=row(index,view,0,bands);component=panel.records[index]['component'];btask='completion' if task=='reference_masked' else task
                heads=sorted(v['onset_us'] for v in baseline[btask][index]['prediction']['events'])
                audio=reference=None;roll=condition=='local_roll'
                if condition in ('song_foreign','audio_foreign'):
                    donor=row(donor_ids[component],'draft')['inputs']
                    audio=donor if condition=='audio_foreign' else {**rr['inputs'],'song':donor['song'],'song_valid':donor['song_valid']}
                if condition=='reference_foreign':
                    own=rr['inputs'];n=int((own['reference_valid'] & (own['reference'][:,10]>0)).sum());candidates=[]
                    for j in plan['indices']['dev'][::2]:
                        if panel.records[j]['component']==component:continue
                        dr=row(j,'middle')['inputs'];m=int((dr['reference_valid'] & (dr['reference'][:,10]>0)).sum())
                        content=dr['reference'][dr['reference_valid'] & (dr['reference'][:,10]>0)]
                        starts=content[:,0]*1.2e9+rr['request'].start_us
                        ends=content[:,1]*1.2e9+rr['request'].start_us
                        legal=bool(((starts>=0)&(starts<rr['request'].duration_us)&
                            ((starts<rr['request'].start_us)|(starts>=rr['request'].end_us))&
                            (ends<=rr['request'].duration_us)).all())
                        if m==n and legal:candidates.append((panel.records[j]['component'],j,dr))
                    if not candidates:
                        control_failures.append(dict(task=task,index=index,condition=condition,reason='No other component with exact visible-content count'));continue
                    reference=sorted(candidates,key=lambda v:(v[0],v[1]))[0][2]
                x=anchor_inputs(rr,heads,audio=audio,reference=reference,mask_reference=masked,roll_audio=roll)
                generated=sampled_marks(model,device(x),heads,rr['request'].duration_us,index=index,view_enum=VIEWS.index(task),seeds=plan['seeds'],allowed_types=rr['request'].allowed_types,deadline=deadline)
                for draw,values in enumerate(generated):
                    if sorted(v['onset_us'] for v in values)!=heads:raise AssertionError('Machine anchor multiset changed')
                    records[draw].append(record(values,rr,index,component,view,base_plan['shifts_us']))
                if task in ('draft','completion','reference_masked'):
                    ox,oy,oh=teacher_example(rr)
                    ox=anchor_inputs(rr,oh,audio=audio,reference=reference,mask_reference=masked,roll_audio=roll)
                    stats=teacher_stats(model,ox,oy,prior);stats['confusion']={f'{a},{b}':v for (a,b),v in stats['confusion'].items()}
                    oracle.append(dict(index=index,component=component,task=task,condition=condition,**stats))
                    if condition=='real' and task=='draft':
                        og=sampled_marks(model,device(ox),oh,rr['request'].duration_us,index=index,view_enum=0,seeds=plan['seeds'],allowed_types=rr['request'].allowed_types,deadline=deadline)
                        for draw,values in enumerate(og):saved.setdefault(draw,[]).append(record(values,rr,index,component,'oracle',base_plan['shifts_us']))
            summaries[condition][task]=[]
            for draw,rows in enumerate(records):
                write(ART/'evaluation'/condition/f'draw{draw}-{task}.json',rows)
                summaries[condition][task].append(metrics(rows) if rows else None)
        write(ART/'oracle-probability.json',oracle)
    for draw,rows in saved.items():write(ART/'evaluation/oracle'/f'draw{draw}-draft.json',rows)
    return dict(machine=summaries,oracle_probability=oracle,oracle_free={str(k):metrics(v) for k,v in saved.items()},control_failures=control_failures)


def run():
    if REPORT.exists():raise FileExistsError('No automatic restart')
    started=time.monotonic();plan=read(str(PLAN.relative_to(ROOT)));report=dict(id=plan['id'],status='preflight',health_updates=0,quality_updates=0,effective_quality_updates=0,health_weights_used=False,curves=[],failures=[],cloud_compute_calls=0)
    write(REPORT,report);state(status='running',pid=os.getpid())
    for f,h in plan['sources'].items():
        if sha(ROOT/f)!=h:raise ValueError('Source drift '+f)
    panel,row=load_data();base=read('bench/rhythm-v2/local-frontend-comparison-v2.json')
    for f,h in base['mert_cache_sha256'].items():
        if sha(ROOT/f)!=h:raise ValueError('Admitted MERT cache drift '+f)
    identities={};prior_count=np.zeros(4,np.int64);empty_pairs=0
    @lru_cache(maxsize=512)
    def example(index,view='draft',field=0):
        value=teacher_example(row(index,view,field));key=str((index,view,field))
        if key in identities:
            actual=digest_example(*value)
            if any(actual[k]!=identities[key][k] for k in actual):raise ValueError('Frozen teacher/input drift '+key)
        return value
    for step,batch in enumerate(plan['sample_stream']):
        nn=0
        for spec in batch:
            key=(spec['index'],spec['view'],spec['field_index']);x,y,heads=example(*key);nn+=len(heads)
            prior_count+=np.bincount(y['kind'].flatten().numpy(),minlength=4)
            identities[str(key)]=dict(index=key[0],component=panel.records[key[0]]['component'],chart_id=panel.records[key[0]]['chart_id'],audio_sha256=panel.records[key[0]]['audio_sha256'],**digest_example(x,y,heads))
        empty_pairs+=nn==0
        guard(plan,started,started,'preflight')
    prior=(prior_count+.5)/(prior_count.sum()+2)
    write(ART/'data-identity.json',dict(rows=identities,exposure_type_counts=prior_count.tolist(),prior=prior.tolist(),empty_update_pairs=empty_pairs,
        unique_request_specs=len(identities),components=len({v['component'] for v in identities.values()}),audios=len({v['audio_sha256'] for v in identities.values()}),charts=len({v['chart_id'] for v in identities.values()}),
        source_data={f:sha(ROOT/base[f]) for f in ('tape','features','mert_cache_report')},cache_sha256=base['mert_cache_sha256']))
    health_specs=list(zip(plan['health_indices'],('draft','single_condition','random_groups','middle')))+[(plan['empty_health_index'],'draft')]
    health=[example(i,v) for i,v in health_specs]
    cpu=fresh(plan,'cpu');gpu=fresh(plan);initial=weight_digest(cpu);parameters=sum(v.numel() for v in cpu.parameters())
    if parameters>plan['model']['max_parameters']:raise ValueError('Mark model parameter cap')
    x,y,_=health[0];cl,_=mark_loss(cpu,x,y);ml,_=mark_loss(gpu,device(x),device(y));cl.backward();ml.backward();torch.mps.synchronize()
    for a,b in zip(cpu.parameters(),gpu.parameters()):
        if (a.grad is None)!=(b.grad is None):raise ValueError('Gradient presence mismatch')
        if a.grad is not None and not bool(torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()):raise ValueError('Nonfinite actual gradients')
    parity=max(abs(float(cl.detach())-float(ml.detach())),max(float((a.grad-b.grad.cpu()).abs().max()) for a,b in zip(cpu.parameters(),gpu.parameters()) if a.grad is not None))
    if not math.isfinite(parity) or parity>1e-4:raise ValueError('Actual CPU/MPS loss/gradient parity')
    receipt=dict(status='completed',scope='new_implementation',parameters=parameters,initial_weight_sha256=initial,parity_max=parity,optimizer_updates=0,
        input_shapes={k:list(v.shape) for k,v in x.items()},input_stats={k:dict(min=float(v.min()),max=float(v.max()),mean=float(v.float().mean()),std=float(v.float().std()) if v.numel()>1 else 0.) for k,v in x.items() if v.numel()},data_identity_sha256=sha(ART/'data-identity.json'))
    write(FOLDER/'anchor-preflight.json',receipt)
    del cpu,gpu,cl,ml;gc.collect();torch.mps.empty_cache()
    plan.update(execution_ready=True,pretraining_probe_receipts=[dict(status='completed',scope='new_implementation',evidence=str((FOLDER/'anchor-preflight.json').relative_to(ROOT)))])
    if validate_successor(plan,{r['id'] for r in load_registry()}):raise ValueError('Predecessor mechanism contract invalid')
    write(PLAN,plan);shutil.copy2(PLAN,ART/'source/frozen-plan.json');report.update(plan_sha256=sha(PLAN),initial_weight_sha256=initial,parameters=parameters)
    stage='health';model=None;opt=None
    try:
        for stage,steps in [('health',100),('quality',1000)]:
            model=fresh(plan);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0)
            if weight_digest(model)!=initial:raise ValueError('Fresh initialization mismatch')
            stage_start=time.monotonic();report['status']=stage+'_running'
            if stage=='health':report['initial_health']=diagnostic(model,health)
            for step in range(1,steps+1):
                memory=guard(plan,started,stage_start,stage);model.train();opt.zero_grad(set_to_none=True)
                lr=.001*step/50 if step<=50 else .0001+.0009*(1+math.cos(math.pi*(step-50)/950))/2
                for group in opt.param_groups:group['lr']=lr
                batch=health if stage=='health' else [example(s['index'],s['view'],s['field_index']) for s in plan['sample_stream'][step-1]]
                total_events=sum(len(v[2]) for v in batch);loss_value=0.
                if total_events:
                    for x,y,heads in batch:
                        if not heads:continue
                        loss,parts=mark_loss(model,device(x),device(y));weighted=loss*len(heads)/total_events;weighted.backward();loss_value+=float(weighted.detach())
                    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step()
                    if stage=='quality':report['effective_quality_updates']+=1
                else:norm=0.
                torch.mps.synchronize();report[stage+'_updates']=step
                if step==1 or step%10==0:
                    report['curves'].append(dict(stage=stage,step=step,loss=loss_value,gradient=norm,lr=lr,wall_s=time.monotonic()-stage_start,memory=memory));write(REPORT,report);print(json.dumps(report['curves'][-1]),flush=True)
                if step in (100,250,500,1000):torch.save(dict(model=model.state_dict(),stage=stage,updates=step,plan_sha256=sha(PLAN)),ART/f'{stage}-{step:04d}.pt')
            report[stage+'_wall_s']=time.monotonic()-stage_start
            if stage=='health':
                final=diagnostic(model,health);report['final_health']=final;first=report['initial_health'];complete=True;health_predictions=[]
                for (i,v),(x,y,heads) in zip(health_specs,health):
                    out=sampled_marks(model,device(x),heads,row(i,v)['request'].duration_us,index=i,view_enum=0,seeds=[plan['seeds'][0]])
                    complete=complete and len(out[0])==len(heads) and all(e['end_us']>=e['onset_us'] for e in out[0])
                    health_predictions.append(dict(index=i,view=v,heads=heads,prediction=out[0],targets=[asdict(e) for e in row(i,v)['targets']]))
                machine_empty=[]
                for saved in read(plan['baseline']+'/quality-1000-empty.json'):
                    rr=row(saved['index'],'draft');heads=sorted(v['onset_us'] for v in saved['prediction']['events']);x=anchor_inputs(rr,heads)
                    out=sampled_marks(model,device(x),heads,rr['request'].duration_us,index=saved['index'],view_enum=1,seeds=[plan['seeds'][0]])
                    if sorted(v['onset_us'] for v in out[0])!=heads:raise ValueError('Machine-empty anchor interface')
                    machine_empty.append(dict(index=saved['index'],heads=heads,prediction=out[0]))
                write(ART/'health-predictions.json',dict(health=health_predictions,machine_empty_interface_only=machine_empty))
                report['health_gate']=dict(type=final['type_nll']<=first['type_nll']*.1,tail=final['tail_nll']<=first['tail_nll']*.1,complete=complete)
                report['health_passed']=all(report['health_gate'].values());write(REPORT,report)
                if not report['health_passed']:
                    report['status']='health_failed_pending_mechanism_review';write(REPORT,report);state(status=report['status'],pid=None);return
                del model,opt;gc.collect();torch.mps.empty_cache()
        report['final_weight_sha256']=sha(ART/'quality-1000.pt');report['status']='evaluating';write(REPORT,report)
        moment=time.monotonic();report['evaluation']=evaluate(model,plan,panel,row,prior,min(started+3600,moment+900))
        report.update(status='terminal_pending_mechanism_review',evaluation_wall_s=time.monotonic()-moment,process_wall_s=time.monotonic()-started);write(REPORT,report);state(status=report['status'],pid=None)
    except BaseException as exc:
        report.update(status='stopped_pending_mechanism_review',error=f'{type(exc).__name__}: {exc}',process_wall_s=time.monotonic()-started)
        if model is not None:torch.save(dict(model=model.state_dict(),stage=stage,updates=report[stage+'_updates']),ART/'interrupted.pt')
        write(REPORT,report);state(status=report['status'],pid=None);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run'));args=parser.parse_args()
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('Silent MPS fallback prohibited')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5)
    def stop(*_):raise KeyboardInterrupt('External stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        {'prepare':prepare,'run':run}[args.action]()
    except BaseException as exc:
        if args.action=='run' and REPORT.exists():
            saved=read(str(REPORT.relative_to(ROOT)))
            if saved['status'] not in ('stopped_pending_mechanism_review','health_failed_pending_mechanism_review'):
                saved.update(status='preflight_failed_pending_mechanism_review',error=f'{type(exc).__name__}: {exc}')
                write(REPORT,saved);state(status=saved['status'],pid=None)
        raise
