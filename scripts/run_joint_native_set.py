"""One bounded, fresh complete-set experiment. No cloud or automatic retries."""
from collections import Counter
from dataclasses import asdict
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.joint_native_set import JointNativeSet
from beatfcn.model.sample import match_f1
from beatfcn.train.joint_native_set import adapted_row, set_batch, set_loss, decode_set, SUPPORT
from scripts.check_experiment_storage import admission, GIB
from scripts.run_local_evolution import load_data
from scripts.run_local_portfolio_scale import sha, write, weight_digest
from scripts.run_local_group_diffusion_health import optimizer_for, rate
from scripts.run_c07_count_ancestral import specs
from scripts.run_local_typed_canvas import score
from scripts.review_frontend_canvas_comparison import metrics


ROOT = Path(__file__).resolve().parents[1]
ID = 'joint-native-set-v1a'
PLAN = ROOT / f'bench/rhythm-v2/{ID}.json'
ART = ROOT / f'artifacts/{ID}'
FOLDER = ROOT / 'reports/local-integration-v1'
REPORT = FOLDER / 'joint-native-set-v1a.json'


def read(path):
    return json.loads((ROOT / path).read_text())


def state(**values):
    p = FOLDER / 'state.json'
    old = json.loads(p.read_text())
    old.update(values)
    write(p, old)


def journal(text):
    with (ROOT / 'EXPERIMENT_LOG.md').open('a') as stream:
        stream.write('\n\nC10 joint-native-set-v1 ' + text + '\n')


def save(report):
    write(REPORT, report)
    state(status=report['status'], candidate=ID, training_process_running=report['status'] in ('health_running', 'quality_running'),
          health_updates=report['health_updates'], new_neural_updates=report['quality_updates'], report=str(REPORT.relative_to(ROOT)))


def fresh(plan, device='mps'):
    torch.manual_seed(plan['seed'])
    if device == 'mps':
        torch.mps.manual_seed(plan['seed'])
    return JointNativeSet(FullEventConfig(**plan['model']), density_reference=plan['density_reference'],
        queries=plan['queries'], set_layers=plan['set_layers']).to(device)


def configure():
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise ValueError('No silent CPU fallback')
    if not torch.backends.mps.is_available():
        raise ValueError('Registered backend unavailable')
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.5)


def prepare():
    if PLAN.exists() or REPORT.exists():
        raise FileExistsError('Preserve registered experiment')
    parent_path = 'bench/rhythm-v2/local-concat-control-quality-v1.json'
    parent = read(parent_path)
    for name, expected in parent['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Historical concat dependency changed: ' + name)
    files = tuple(parent['source_sha256']) + (
        'beatfcn/model/joint_native_set.py', 'beatfcn/train/joint_native_set.py',
        'beatfcn/train/native_joint_support.py', 'beatfcn/train/joint_reference.py',
        'scripts/run_joint_native_set.py', 'tests/test_joint_native_set.py',
        'scripts/check_experiment_storage.py')
    p = {k:parent[k] for k in ('seed', 'model', 'density_reference', 'indices', 'health_indices', 'empty_health_index',
        'sample_stream', 'quality_steps', 'health_steps', 'effective_batch', 'learning_rate', 'minimum_learning_rate',
        'warmup', 'weight_decay', 'clip', 'health_wall_s', 'quality_wall_s', 'evaluation_wall_s', 'shifts_us', 'input_payload_sha256')}
    p.update(id=ID, parent_plan=parent_path, parent_sha256=sha(ROOT / parent_path),
        queries=256, set_layers=2, max_parameters=12000000, new_disk_peak_bytes=GIB, reserve_bytes=15*GIB,
        preflight_wall_s=400, process_wall_s=9000, execution_ready=False,
        hypothesis='A complete native event set with matched attributes removes count-tail-rank and fixed-machine-head bottlenecks; continuous SSD audio plus directly readable song/reference and head-conditioned relative tails can improve the whole recipe.',
        probability='Per-query5way class and normalized native/COPY head; head-conditioned legally normalized relative tail. Balanced matching surrogate is NOT a set likelihood or a calibrated occupancy score.',
        objective='Hungarian full type+head+hold NLL cost; .5 mean of per-present-type positive means + .5 unmatched absent mean. Empty request: absent mean only. No focal, count, existence, total or interval loss.',
        readout='Class argmax, native-address argmax, conditional legal relative-tail argmax; no count rounding, tail rank, quota, NMS, deduplication or min-gap.',
        component_changes=['concat SSD continuous encoder and original frame fusion retained', 'full song and complete reference rows directly cross-attended',
            'finite event-set queries jointly supervised; repeated identical targets are separate columns',
            'native integer-ms aliases unique; fractional authorized COPY kept', 'relative tail conditioned on head at train and predicted head at decode'],
        health_gate=dict(loss_fraction=.15, positive_mean_full_f1=.90, positive_min_full_f1=.75, mean_onset10=.80, empty_events=0,
            cpu_mps_max_abs=1e-4, finite=True),
        research_gate=dict(full_gain_over_concat=.02, alternative_non_tap_gain=.03, full_retention_floor=.3003642151482393,
            count_ratio=[.5,1.75], max_empty_mean=3, acoustic_gain=.05, positive_components=10),
        default_gate=dict(full50=.3398040839133824, hold_f1=.05, flick_f1=.18565400843881857, drag_f1=.18430034129692832, empty_events=0),
        scaling_authorized=False, release_acceptance=False, cloud_compute_calls=0, confirmation_scored=False,
        pretraining_probe_receipts=[], predecessors=['ssd_full', 'no-mark-history', 'local-concat-control-quality-v1'],
        shared_failure_modes=['fixed_machine_heads', 'count_readout_mismatch', 'tail_origin_and_conditioning', 'lost_known_empty_reference'],
        mechanism_changes={
            'fixed_machine_heads':dict(change='Queries generate their own heads and kinds',prediction='Whole free-event score can improve beyond mark-only chain',discriminating_control='Same32native targets versus concat and strongestSSD; no oracle replaced as main result'),
            'count_readout_mismatch':dict(change='One5way decision per query; no global expected count/ranking',prediction='Mode outputs reflect trained presence decisions',discriminating_control='Empty inputs, per-query marginals versus hard emitted count, prior/no-audio controls'),
            'tail_origin_and_conditioning':dict(change='Relative legal delta and explicit head-conditioned tail features',prediction='Matched-head tails transfer across window origins',discriminating_control='Exact fractionalCOPY/translation regression and emitted matched-hold errors'),
            'lost_known_empty_reference':dict(change='Known complete regions persist even with zero visible events',prediction='Protected silence is visible to encoder',discriminating_control='Actual known-empty input rows; full reference removal with requests unchanged')},
        source_sha256={f:sha(ROOT/f) for f in files})
    # Correct registry ids, do not invent predecessor identity.
    p['predecessors']=['evolution-ssd-full-v1', 'c08-no-mark-history-v1', 'local-concat-control-quality-v1']
    p['implementation_fix']='v1 failed before optimizer update0: unsupported MPS index_copy. v1a uses an algebraically identical incidence matrix insertion; independent gradient regression required, no fallback.'
    write(PLAN,p)
    for f in files:
        dest=ART/'source'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
    shutil.copy2(PLAN,ART/'source/registered-plan.json')
    journal('冻结完整联合候选：保留concat的4000请求/1000更新及优化器；10.68M静态实例化，正式参数/设备与健康仍待真实探针。无旧质量权重复用，无云。匹配代理不作集合边际似然声明。')


def check_sources(plan):
    for name, expected in plan['source_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Registered source drift: '+name)


def resource(plan, started, wall):
    torch.mps.synchronize()
    if time.monotonic()-started > wall:
        raise TimeoutError('Registered stage wall cap')
    disk=admission(ROOT,plan['new_disk_peak_bytes'],plan['reserve_bytes'])
    if disk['status']!='admitted':
        raise ValueError('Disk reserve exhausted')
    driver=torch.mps.driver_allocated_memory();recommended=torch.mps.recommended_max_memory()
    if driver > .8*recommended:
        raise ValueError('MPS driver memory cap')
    return dict(wall_s=time.monotonic()-started,allocated_bytes=torch.mps.current_allocated_memory(),driver_bytes=driver,
                recommended_bytes=recommended,free_disk_bytes=shutil.disk_usage(ROOT).free)


def preflight(plan,panel,row):
    started=time.monotonic();checks=[];types=Counter();max_n=max_cell=duplicates=cross=0
    for path,expected in plan['input_payload_sha256'].items():
        if sha(ROOT/path)!=expected:raise ValueError('Cache/target identity drift: '+path)
    for batch in plan['sample_stream']:
        for s in batch:
            rr=adapted_row(panel,row(s['index'],s['view'],s['field_index']))
            coded=SUPPORT.encode(rr['targets'],rr['request'])
            if Counter(SUPPORT.decode(coded,rr['request']))!=Counter(rr['targets']):raise ValueError('Multiset audit failed')
            n=len(coded.members);max_n=max(max_n,n);max_cell=max(max_cell,int(coded.counts.max(initial=0)))
            if n>plan['queries']:raise ValueError('Query capacity overflow')
            types.update(e.type for e in rr['targets']);duplicates+=n-len(set(rr['targets']))
            cross+=sum(e.type=='hold' and e.end_us>rr['request'].end_us for e in rr['targets'])
            checks.append(dict(**s,events=n,target_sha256=hashlib.sha256(json.dumps([asdict(e) for e in sorted(rr['targets'])],sort_keys=True).encode()).hexdigest()))
            if time.monotonic()-started>plan['preflight_wall_s']:raise TimeoutError('Preflight wall cap')
    data=dict(status='completed',requests=len(checks),exposure_type_counts=dict(types),max_events=max_n,max_cell=max_cell,
        exact_duplicate_exposures=duplicates,cross_core_hold_exposures=cross,checks=checks,wall_s=time.monotonic()-started)
    write(ART/'data-identity.json',data)
    return {k:v for k,v in data.items() if k!='checks'}


def audit_tensors(inputs):
    values={}
    for k,v in inputs.items():
        if not v.numel():continue
        f=v.float();values[k]=dict(shape=list(v.shape),finite=bool(torch.isfinite(f).all()),
            minimum=float(f.min()),maximum=float(f.max()),mean=float(f.mean()),std=float(f.std(unbiased=False)))
        if not values[k]['finite']:raise ValueError('Nonfinite input '+k)
    return values


@torch.no_grad()
def health_decode(model,batches):
    rows=[]
    for rr,x,t in batches:
        pred=decode_set(model,x,rr['request']);events=[RhythmEvent(**e) for e in pred['events']]
        rows.append(dict(index=rr['index'],view=rr['view'],target=[asdict(e) for e in rr['targets']],
            prediction=pred,scores=score(events,rr['targets'])))
    return rows


def health(plan,panel,row,report):
    model=fresh(plan);report['parameters']=sum(p.numel() for p in model.parameters())
    if report['parameters']>plan['max_parameters']:raise ValueError('Parameter cap')
    report['initial_weight_sha256']=weight_digest(model)
    specifications=list(zip(plan['health_indices'],('draft','single_condition','random_groups','middle')))+[(plan['empty_health_index'],'draft')]
    batches=[]
    for i,v in specifications:
        rr=adapted_row(panel,row(i,v));x,t=set_batch(rr,'mps');batches.append((rr,x,t))
    report['input_audit']=[dict(index=rr['index'],tensors=audit_tensors(x)) for rr,x,t in batches]
    rr,x,t=batches[0]
    cpu=fresh(plan,'cpu');cx={k:v.cpu() for k,v in x.items()}
    co=cpu(cx);cl,_=set_loss(cpu,co,t,rr['request'],cx);cl.backward()
    mo=model(x);ml,_=set_loss(model,mo,t,rr['request'],x);ml.backward();torch.mps.synchronize()
    differences={k:float((co[k].detach()-mo[k].detach().cpu()).abs().max()) for k in ('kind','head','query')}
    differences['loss']=abs(float(cl)-float(ml))
    differences['gradient']=max(float((p.grad.detach()-mp.grad.detach().cpu()).abs().max())
        for p,mp in zip(cpu.parameters(),model.parameters()) if p.grad is not None and mp.grad is not None)
    report['cpu_mps']=differences
    if max(differences.values())>plan['health_gate']['cpu_mps_max_abs']:raise ValueError('CPU/MPS parity')
    del cpu,co,mo,cl,ml,cx;model.zero_grad(set_to_none=True);gc.collect();torch.mps.empty_cache()
    opt=optimizer_for(model,plan);started=time.monotonic();initial=None
    for step in range(1,plan['health_steps']+1):
        model.train();opt.zero_grad(set_to_none=True);losses=[];parts=[]
        for rr,x,t in batches:
            loss,detail=set_loss(model,model(x),t,rr['request'],x)
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite health loss')
            (loss/len(batches)).backward();losses.append(float(loss.detach()))
            parts.append({k:float(v.detach()) for k,v in detail.items()})
        grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        if not np.isfinite(grad):raise ValueError('Nonfinite health gradient')
        for g in opt.param_groups:g['lr']=.002
        opt.step();report['health_updates']=step
        mean=float(np.mean(losses));initial=mean if initial is None else initial
        if step==1 or step%10==0:
            sampled=resource(plan,started,plan['health_wall_s'])
            report['health_curve'].append(dict(step=step,loss=mean,parts=parts,gradient=grad,**sampled));save(report)
            print('health',step,mean,round(sampled['wall_s'],2),flush=True)
    results=health_decode(model,batches);write(ART/'health-predictions.json',results)
    nonempty=[r for r in results if r['target']];empty=[r for r in results if not r['target']]
    gate=plan['health_gate'];last=report['health_curve'][-1]['loss']
    checks=dict(loss=last<=initial*gate['loss_fraction'],complete=all(r['prediction']['complete'] for r in results),
        mean_full=np.mean([r['scores']['full_note_f1'] for r in nonempty])>=gate['positive_mean_full_f1'],
        minimum_full=min(r['scores']['full_note_f1'] for r in nonempty)>=gate['positive_min_full_f1'],
        onset10=np.mean([r['scores']['onset_10ms']['f1'] or 0 for r in nonempty])>=gate['mean_onset10'],
        empty=all(len(r['prediction']['events'])==gate['empty_events'] for r in empty))
    checks={k:bool(v) for k,v in checks.items()};report['health_checks']=checks;report['health_passed']=all(checks.values())
    report['health_wall_s']=time.monotonic()-started;save(report)
    del model,opt,batches;gc.collect();torch.mps.empty_cache()
    return report['health_passed']


def train(plan,panel,row,report):
    model=fresh(plan)
    if weight_digest(model)!=report['initial_weight_sha256']:raise ValueError('Fresh initialization drift')
    report['health_weights_used']=False;opt=optimizer_for(model,plan);started=time.monotonic();exposure=Counter();views=Counter()
    report['status']='quality_running';save(report)
    for step,batch in enumerate(plan['sample_stream'],1):
        model.train();opt.zero_grad(set_to_none=True);values=[]
        for s in batch:
            rr=adapted_row(panel,row(s['index'],s['view'],s['field_index']));x,t=set_batch(rr,'mps')
            loss,_=set_loss(model,model(x),t,rr['request'],x)
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite quality loss')
            (loss/len(batch)).backward();values.append(float(loss.detach()));exposure.update(e.type for e in rr['targets']);views.update([s['view']])
        grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),plan['clip']))
        if not np.isfinite(grad):raise ValueError('Nonfinite quality gradient')
        for group in opt.param_groups:group['lr']=rate(plan,step)
        opt.step();report['quality_updates']=step
        if step==1 or step%25==0:
            sample=resource(plan,started,plan['quality_wall_s'])
            report['quality_curve'].append(dict(step=step,loss=float(np.mean(values)),gradient=grad,**sample));save(report)
            print('quality',step,float(np.mean(values)),round(sample['wall_s'],2),flush=True)
    torch.mps.synchronize();report['quality_wall_s']=time.monotonic()-started
    report['exposure']=dict(windows=sum(views.values()),events=sum(exposure.values()),types=dict(exposure),views=dict(views))
    checkpoint=ART/'quality-1000.pt'
    torch.save(dict(model=model.state_dict(),updates=report['quality_updates'],plan_sha256=sha(PLAN)),checkpoint)
    report['checkpoint_sha256']=sha(checkpoint);report['status']='evaluation_running';save(report)
    return model


@torch.no_grad()
def evaluate(model,plan,panel,row,report):
    started=time.monotonic();deadline=started+plan['evaluation_wall_s'];summary={}
    for task,items in specs(plan).items():
        records=[]
        for i,view,bands,masked in items:
            if time.monotonic()>deadline:raise TimeoutError('Evaluation wall cap')
            rr=adapted_row(panel,row(i,view,0,bands),mask_reference=masked);x,t=set_batch(rr,'mps')
            out=model(x);pred=decode_set(model,x,rr['request'],deadline=deadline,output=out)
            events=[RhythmEvent(**e) for e in pred['events']];scores=score(events,rr['targets'])
            start=rr['request'].start_us;width=rr['request'].end_us-start
            shifted=[match_f1([start+(e.onset_us-start+d)%width for e in events],[e.onset_us for e in rr['targets']],10000)[3] for d in plan['shifts_us']]
            kp=out['kind'][0].softmax(-1).cpu().numpy().astype(np.float64)
            hp=out['head'][0].softmax(-1).cpu().numpy().astype(np.float64)
            allowed=x['joint_head_valid'][0].cpu().numpy();hp=hp[:,allowed]
            expected=kp[:,1:].T@hp
            occupancy=np.stack([-np.expm1(np.log1p(-np.clip(kp[:,k,None]*hp,0,1-1e-12)).sum(0)) for k in range(1,5)])
            probpath=ART/'probabilities'/f'{task}-{i}.npz';probpath.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(probpath,kind=kp.astype(np.float32),expected=expected.astype(np.float32),occupancy=occupancy.astype(np.float32),
                heads=SUPPORT.addresses(rr['request'])[0].reshape(-1)[allowed])
            records.append(dict(index=i,component=panel.records[i]['component'],view=view,request=asdict(rr['request']),
                prediction=pred,target=[asdict(e) for e in rr['targets']],scores=scores,shifted_onset10=shifted,
                model_expected_events=float((1-kp[:,0]).sum()),probabilities=str(probpath.relative_to(ROOT))))
            write(ART/f'{task}.json',records)
        summary[task]=metrics(records)
    report['evaluation']=summary;report['evaluation_wall_s']=time.monotonic()-started
    write(ART/'evaluation-summary.json',summary)


def run():
    if REPORT.exists():raise FileExistsError('No automatic scientific rerun')
    configure();plan=read(str(PLAN.relative_to(ROOT)));check_sources(plan)
    disk=admission(ROOT,plan['new_disk_peak_bytes'],plan['reserve_bytes'])
    if disk['status']!='admitted':raise ValueError('Disk admission')
    report=dict(id=ID,status='preflight_running',health_updates=0,quality_updates=0,health_curve=[],quality_curve=[],failures=[],
        disk_admission=disk,health_weights_used=None,cloud_compute_calls=0,cloud_spend_usd=0)
    save(report);started=time.monotonic()
    def stop(*_):raise TimeoutError('Process wall or owner interrupt')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGALRM,stop)
    signal.alarm(plan['process_wall_s'])
    try:
        panel,row=load_data();report['data_probe']=preflight(plan,panel,row);report['status']='health_running';save(report)
        if not health(plan,panel,row,report):
            report['status']='health_failed_pending_mechanism_review';save(report);return
        receipt=FOLDER/'joint-native-set-pretraining.json'
        write(receipt,dict(status='completed',data_probe=report['data_probe'],parameters=report['parameters'],cpu_mps=report['cpu_mps'],health_checks=report['health_checks']))
        plan.update(execution_ready=True,pretraining_probe_receipts=[dict(status='completed',evidence=str(receipt.relative_to(ROOT)),scope='new_implementation_real_data_and_health')])
        write(PLAN,plan);shutil.copy2(PLAN,ART/'source/frozen-plan.json');report['plan_sha256']=sha(PLAN)
        subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'scripts.validate_mechanism_reviews', '--all', '--successor', str(PLAN)], cwd=ROOT, check=True)
        model=train(plan,panel,row,report);evaluate(model,plan,panel,row,report)
        report['status']='terminal_pending_mechanism_review'
    except BaseException as exc:
        report['status']='execution_failed_pending_mechanism_review'
        report['failures'].append(dict(type=type(exc).__name__,error=str(exc)))
        raise
    finally:
        signal.alarm(0);report['process_wall_s']=time.monotonic()-started;save(report)
        journal(f"状态{report['status']}，health={report['health_updates']}，quality={report['quality_updates']}，云0。终态需完整机制复盘后结案。")


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run'));args=parser.parse_args()
    {'prepare':prepare,'run':run}[args.action]()
