"""One complete warm-start noisy-proposal20ms refinement experiment."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.noisy_proposal_refiner import NoisyProposalRefiner,refiner_loss,generate_refined
from beatfcn.model.sample import match_f1
from beatfcn.train.canonical_probability import explicit_row
from beatfcn.train.noisy_proposals import proposal_cache_key,proposal_inputs,proposal_targets,jittered_training_proposals
from beatfcn.train.whole_song_mel import WholeMelStore
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_group_diffusion_health import optimizer_for,rate
from scripts.run_local_portfolio_scale import row_factory,weight_digest,sha,write
from scripts.run_local_typed_canvas import score
from scripts.review_frontend_canvas_comparison import metrics


ROOT=Path(__file__).resolve().parents[1];IDENT='local-noisy-proposal-refiner-v2'
PLAN=ROOT/f'bench/rhythm-v2/{IDENT}.json';REPORT=ROOT/'reports/local_noisy_proposal_refiner_v2.json';ART=ROOT/'artifacts/local-noisy-proposal-refiner-v2'
BODY=ROOT/'artifacts/local-whole-song-arrangement-v1/quality-1000.pt'


def fresh(plan,device='mps'):
    torch.manual_seed(plan['seed']);model=NoisyProposalRefiner(FullEventConfig(**plan['model']),density_reference=plan['density_reference'])
    saved=torch.load(BODY,map_location='cpu',weights_only=False)
    model.body.load_state_dict(saved['model']);return model.to(device)


def journal(text):
    with (ROOT/'EXPERIMENT_LOG.md').open('a') as f:f.write(f'\n\n{IDENT} {text}\n')


def checkpoint(model,opt,stage,step):
    path=ART/f'{stage}-{step:04d}.pt';path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),stage=stage,updates=step,plan_sha256=sha(PLAN),cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state()),path);return path


def cached(example,cache,masked=False):
    key=proposal_cache_key(example,masked)
    record=cache['files'].get(key)
    if record is None:raise ValueError('Exact frozen input proposal missing')
    path=ROOT/record['path']
    if sha(path)!=record['sha256']:raise ValueError('Proposal cache identity drift')
    value=json.loads(path.read_text())
    return value['prediction'],[int(e['onset_us']) for e in value['prediction']['events']]


def augmented_times(example,cache,seed,*,health=False):
    rng=np.random.default_rng(seed);source,times=cached(example,cache)
    if not health and rng.random()<.5:return jittered_training_proposals(example['targets'],example['request'],seed),dict(source='truth_jitter_and_background',original_complete=source['complete'])
    times=times if source['complete'] else []
    request=example['request'];width=(request.end_us-request.start_us)//1000
    n=int(rng.poisson(2*(request.end_us-request.start_us)/1e6))
    background=[request.start_us+int(v)*1000 for v in rng.integers(max(width,1),size=max(n,1))]
    return sorted(times+background),dict(source='frozenSSD_and_background',original_complete=source['complete'],background=len(background))


def batch(example,times,store,panel,device='mps',masked=False):
    inputs,_=store.batch(example,panel.records[example['index']]['audio_sha256'],device,mask_reference=masked)
    proposal=proposal_inputs(times,example['request'],device)
    target,matching=proposal_targets(times,example['targets'],example['request'],device=device)
    return inputs,proposal,target,matching


def diagnostics(model,batches):
    model.eval();rows=[]
    with torch.no_grad():
        for e,inp,proposal,target,matching in batches:
            loss,parts=refiner_loss(model,inp,proposal,target)
            rows.append(dict(index=e['index'],loss=float(loss),parts={k:float(v) for k,v in parts.items()},matching=matching))
    return rows


def evaluate(model,plan,row,panel,store,cache,deadline):
    dev=plan['indices']['dev'];sel=dev[::2]
    tasks={'draft':[(i,'draft',None,False) for i in dev],'empty':[(i,'draft',None,False) for i in plan['indices']['empty_dev']],
        'completion':[(i,'middle',None,False) for i in sel],'reference_masked':[(i,'middle',None,True) for i in sel],
        'density_low':[(i,'draft',(2.,3.),False) for i in dev[::8]],'density_high':[(i,'draft',(6.,8.),False) for i in dev[::8]]}
    summaries={}
    for task,specs in tasks.items():
        records=[]
        for index,view,band,masked in specs:
            if time.monotonic()>deadline:raise TimeoutError('Frozen refiner evaluation wall')
            e=explicit_row(row(index,view,0,band));source,times=cached(e,cache,masked)
            try:
                if not source['complete']:raise ValueError('Frozen proposal generator incomplete: '+str(source['failure']))
                if times:
                    # Inference does not construct matching targets or consult
                    # hidden events; only actual frozen proposal times enter.
                    inp,_=store.batch(e,panel.records[index]['audio_sha256'],mask_reference=masked)
                    proposal=proposal_inputs(times,e['request'],'mps');pred=generate_refined(model,inp,proposal,e['request'])
                else:pred=[]
                prediction=dict(complete=True,events=[asdict(v) for v in pred],source_proposals=len(times))
            except (ValueError,RuntimeError) as exc:pred=[];prediction=dict(complete=False,events=[],failure=str(exc),source_proposals=len(times))
            start=e['request'].start_us;width=e['request'].end_us-start
            shifts=[match_f1([start+(v.onset_us-start+d)%width for v in pred],[v.onset_us for v in e['targets']],10000)[3] for d in plan['shifts_us']]
            records.append(dict(index=index,component=panel.records[index]['component'],view=view,request=asdict(e['request']),prediction=prediction,target=[asdict(v) for v in e['targets']],scores=score(pred,e['targets']),shifted_onset10=shifts,
                frozen_proposal_scores=score([RhythmEvent(**v) for v in source['events']],e['targets']),proposal_generator_complete=source['complete']))
            write(ART/f'{task}-final.json',records)
        summaries[task]=metrics(records)
    return summaries


def zero_proposal_loss(model):
    # An empty proposal set has no member likelihood. Its request weight stays
    # in the fixed batch mean; no oracle/background event is invented.
    return sum(p.sum() * 0 for p in model.parameters() if p.requires_grad)


def main():
    if PLAN.exists() or REPORT.exists():raise FileExistsError('Preserve complete refiner run')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('No silent fallback')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5)
    cache_path=ROOT/'reports/local_noisy_proposal_cache_v1.json';cache=json.loads(cache_path.read_text())
    if cache['status']!='completed':raise ValueError('Complete fixed proposal cache required')
    parent=json.loads((ROOT/'bench/rhythm-v2/local-portfolio-scale-v1.json').read_text());body_plan_path=ROOT/'bench/rhythm-v2/local-whole-song-arrangement-v1.json';body_plan=json.loads(body_plan_path.read_text())
    body_report=json.loads((ROOT/'reports/local_whole_song_arrangement_v1.json').read_text())
    if body_report['status']!='completed' or body_report['final_weight_sha256']!=sha(BODY):raise ValueError('Exact full-song body warmstart identity')
    sources=('scripts/run_noisy_proposal_refiner_v2.py','beatfcn/model/noisy_proposal_refiner.py','beatfcn/train/noisy_proposals.py','beatfcn/model/whole_song_arrangement.py','beatfcn/train/whole_song_mel.py')
    plan=dict(id=IDENT,seed=20261060,model=body_plan['model'],density_reference=parent['density_reference'],indices=parent['indices'],sample_stream=parent['sample_stream']['train64'][:1000],
        health_indices=parent['health_indices'],empty_health_index=parent['empty_health_index'],health_steps=100,quality_steps=1000,effective_batch=2,
        learning_rate=.0003,minimum_learning_rate=.00003,warmup=50,weight_decay=.01,clip=1.,health_wall_s=900,quality_wall_s=3600,evaluation_wall_s=900,max_trainable_parameters=4_000_000,
        implementation_repair='v1 stopped after267updates on a legitimate zero-proposal training request. Preserve v1. Empty request contributes exact0 with fixed batch denominator; no invented event. Same original warmstart/seed, new100health and fresh1000; old267updates separately charged.',
        warmstart=dict(body_sha256=sha(BODY),body_plan_sha256=sha(body_plan_path),proposal_cache_sha256=sha(cache_path),mode='B audio/body warm start plus new random attribute heads. All health-updated weights discarded; quality reloads this exact initial warm start, not random audio pretraining.'),
        proposal_policy='Inference frozenSSD1000 expected-tail-rank only; training50%its actual proposals plus uniform background,50%truth jitter<=20ms plus uniform background2/s. Fixed train-only augmentation; no oracle inference, quotas/NMS/min-gap or type input torefiner.',
        matching='Typeblind1to1 maxcardinality/minabs distance<=50ms, multiset duplicates retained; unmatched proposals classREJECT, unproposed targets not invented.',
        head='20ms addressed features plus causal proposal conv; native/COPY conditional refinement within50ms, native targetGaussian sigma1.5ms, fullhead/tail copy authorization. Exact native clocks computed onCPU.',
        hypothesis='StrongSSD placement proposals and learned full-song type features can be combined through supervised noisy attributes/reject/refinement as a complete pipeline, addressing their observed complementary failures.',
        baseline=dict(full_f1=.3198040839133824,onset10=.3144195247712359),
        local_screen=dict(min_nonempty=24,count_ratio=[.5,1.75],onset10=.20,onset10_minus_shift=.05,positive_components=10,full_note_f1=.20),
        refinement_screen=dict(full_gain=.02,onset_noninferiority=.02,min_component_wins=10),
        health_gate='Loss decreases10%; classification andhead excessCE over targetentropy no worse5%, finite CPU/MPS<=1e-4, legal5realproposal+background outputs. No productF1/raretype presence gate.',
        shifts_us=parent['shifts_us'],cloud_compute_calls=0,confirmation_scored=False,source_sha256={n:sha(ROOT/n) for n in sources})
    write(PLAN,plan)
    for name in sources:
        dest=ART/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    shutil.copy2(PLAN,ART/'source/frozen-plan.json')
    panel,features=setup(json.loads((ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()));row=row_factory(panel,features);store=WholeMelStore(ROOT,panel.features.records,ROOT/'artifacts/local-whole-song-arrangement-v1/whole-mel')
    model=fresh(plan);torch.mps.manual_seed(plan['seed']);opt=optimizer_for(model,plan)
    n=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable>plan['max_trainable_parameters']:raise ValueError('Refiner trainable parameter cap')
    report=dict(id=IDENT,status='health_running',parameters=n,trainable=trainable,plan_sha256=sha(PLAN),initial_weight_sha256=weight_digest(model),health_updates=0,quality_updates=0,curves=[],failures=[],health_weights_used=False,body_is_warmstarted=True,augmentation_exposure={},cloud_compute_calls=0,confirmation_scored=False)
    write(REPORT,report);journal(f'完整有噪提议路线冻结，total{n}/trainable{trainable}，SSD提议+B音频body显式warmstart；100健康后重置相同warmstart新头再1000；云0。')
    def stopped(*_):raise KeyboardInterrupt('External interruption')
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped);stage='health';started=time.monotonic()
    try:
        specs=list(zip(plan['health_indices'],('draft','single_condition','random_groups','middle')))+[(plan['empty_health_index'],'draft')];batches=[]
        for index,view in specs:
            e=explicit_row(row(index,view));times,provenance=augmented_times(e,cache,plan['seed']+index,health=True)
            inp,proposal,target,matching=batch(e,times,store,panel);batches.append((e,inp,proposal,target,dict(matching,**provenance)))
        report['input_audit']=[]
        for e,inp,proposal,target,matching in batches:
            audited={}
            for section,values in (('audio_reference_controls',inp),('proposal_only',proposal)):
                audited[section]={}
                for key,value in values.items():
                    if section=='audio_reference_controls' and key not in ('mel','mel_flux','whole_song','whole_song_times','whole_song_valid','reference','controls'):continue
                    v=value.detach().cpu()
                    audited[section][key]=dict(shape=list(v.shape),finite=bool(torch.isfinite(v).all()),minimum=float(v.min()),maximum=float(v.max()),
                        mean=float(v.float().mean()),std=float(v.float().std()) if v.numel()>1 else 0.)
            report['input_audit'].append(dict(index=e['index'],matching=matching,tensors=audited))
        write(REPORT,report)
        report['initial_diagnostic']=diagnostics(model,batches)
        cpu=fresh(plan,'cpu').eval();e,inp,proposal,target,_=batches[0]
        ci,cp,ct=({k:v.cpu() for k,v in d.items()} for d in (inp,proposal,target))
        with torch.no_grad():a,b=cpu(ci,cp),model(inp,proposal)
        forward=max(float((a[k]-b[k].cpu()).abs().max()) for k in a)
        cl,_=refiner_loss(cpu,ci,cp,ct);ml,_=refiner_loss(model,inp,proposal,target);cl.backward();ml.backward();params=dict(cpu.named_parameters());gradient=0.
        for name,p in model.named_parameters():
            if (p.grad is None)!=(params[name].grad is None):raise ValueError('Gradient presence parity')
            if p.grad is not None:gradient=max(gradient,float((p.grad.cpu()-params[name].grad).abs().max()))
        report['parity']=dict(forward=forward,gradient=gradient)
        if max(forward,gradient)>1e-4:raise ValueError('CPU/MPS parity')
        del cpu,a,b,cl,ml,params
        for step in range(1,101):
            if time.monotonic()-started>plan['health_wall_s']:raise TimeoutError('Health wall')
            model.train();opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']=rate(plan,step)
            total=0.
            for e,inp,proposal,target,_ in batches:
                loss,_=refiner_loss(model,inp,proposal,target);(loss/5).backward();total+=float(loss.detach())/5
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();report['health_updates']=step
            if step%10==0 or step==1:
                point=dict(stage=stage,step=step,loss=total,gradient=grad,wall_s=time.monotonic()-started);report['curves'].append(point);write(REPORT,report);print(json.dumps(point),flush=True)
            if step%25==0:checkpoint(model,opt,stage,step)
        report['final_diagnostic']=diagnostics(model,batches);preds=[]
        for e,inp,proposal,target,matching in batches:
            pred=generate_refined(model,inp,proposal,e['request']);preds.append(dict(index=e['index'],prediction=dict(complete=True,events=[asdict(v) for v in pred]),target=[asdict(v) for v in e['targets']],scores=score(pred,e['targets']),matching=matching))
        write(ART/'health-predictions.json',preds);a,b=report['initial_diagnostic'],report['final_diagnostic']
        gates=dict(loss=bool(np.mean([v['loss'] for v in b])<=.9*np.mean([v['loss'] for v in a])),complete=len(preds)==5,
            classification=bool(np.mean([v['parts']['classification'] for v in b])<=1.05*np.mean([v['parts']['classification'] for v in a])),
            head_excess=bool(np.mean([v['parts']['head']-v['parts']['head_target_entropy'] for v in b])<=1.05*np.mean([v['parts']['head']-v['parts']['head_target_entropy'] for v in a])))
        report.update(health_gates=gates,health_passed=all(gates.values()),health_wall_s=time.monotonic()-started)
        if not report['health_passed']:
            report['status']='health_failed';write(REPORT,report);journal(f'健康失败结案{gates}。');return
        checkpoint(model,opt,'health',100);del model,opt,batches;torch.mps.empty_cache();model=fresh(plan);torch.mps.manual_seed(plan['seed']);opt=optimizer_for(model,plan)
        if weight_digest(model)!=report['initial_weight_sha256']:raise ValueError('Original warmstart+newhead identity mismatch')
        stage='quality';started=time.monotonic();report['status']='quality_running';write(REPORT,report);journal('健康通过；丢弃健康学习，恢复原Bbody和相同新头初态，1000质量开始。')
        for step in range(1,1001):
            if time.monotonic()-started>plan['quality_wall_s']:raise TimeoutError('Quality wall')
            model.train();opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']=rate(plan,step)
            total=0.
            for j,spec in enumerate(plan['sample_stream'][step-1]):
                e=explicit_row(row(spec['index'],spec['view'],spec['field_index']));times,provenance=augmented_times(e,cache,plan['seed']+17*step+j)
                report['augmentation_exposure'][provenance['source']]=report['augmentation_exposure'].get(provenance['source'],0)+1
                if times:
                    inp,proposal,target,_=batch(e,times,store,panel)
                    loss,_=refiner_loss(model,inp,proposal,target)
                else:
                    report['empty_training_requests']=report.get('empty_training_requests',0)+1
                    loss=zero_proposal_loss(model)
                (loss/2).backward();total+=float(loss.detach())/2
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();report['quality_updates']=step
            if step%25==0 or step==1:
                point=dict(stage=stage,step=step,loss=total,gradient=grad,wall_s=time.monotonic()-started);report['curves'].append(point);write(REPORT,report);print(json.dumps(point),flush=True)
            if step%250==0:checkpoint(model,opt,stage,step)
        path=checkpoint(model,opt,'quality',1000);report['final_weight_sha256']=sha(path);report['status']='evaluating';write(REPORT,report)
        moment=time.monotonic();result=evaluate(model,plan,row,panel,store,cache,min(started+plan['quality_wall_s'],moment+plan['evaluation_wall_s']))
        d=result['draft'];g=plan['local_screen'];gates=dict(complete=d['complete']==d['attempted'],nonempty=d['nonempty']>=g['min_nonempty'],count_ratio=g['count_ratio'][0]<=d['density_ratio']<=g['count_ratio'][1],onset10=d['onset10']>=g['onset10'],onset_null=d['onset10']-d['shift_null10']>=g['onset10_minus_shift'],components=sum(v['onset10']>v['shift_null10'] for v in d['components'].values())>=g['positive_components'],full_f1=d['full_note_f1']>=g['full_note_f1'])
        report.update(status='completed',evaluations=result,local_screen=gates,local_screen_passed=all(gates.values()),evaluation_wall_s=time.monotonic()-moment,total_quality_wall_s=time.monotonic()-started)
        write(REPORT,report);journal(f'完整1000和实际提议自由评价完成，本地筛选{gates}；不择优、不加步、不自动上云。')
    except BaseException as exc:
        report.update(status='stopped',stopped_stage=stage,stage_wall_s=time.monotonic()-started);report['failures'].append(f'{type(exc).__name__}: {exc}')
        checkpoint(model,opt,f'{stage}-interrupted',report[f'{stage}_updates']);write(REPORT,report);journal(f'停止保留现场：{report["failures"][-1]}。');raise


if __name__=='__main__':main()
