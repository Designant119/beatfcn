"""Complete10ms lattice/Focal/counterfactual route, separate from all others."""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from beatfcn.events.controls import BAND_EDGES
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_lattice import LatticeCanvas,lattice_loss,generate_lattice,counterfactual_hinge
from beatfcn.model.local_song_rollout import canonical
from beatfcn.model.sample import match_f1
from beatfcn.train.canonical_probability import explicit_row
from beatfcn.train.event_conditions import training_labels
from beatfcn.train.lattice_targets import lattice_batch,lattice_target
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_group_diffusion_health import optimizer_for,rate
from scripts.run_local_portfolio_scale import row_factory,weight_digest,sha,write
from scripts.run_local_typed_canvas import score
from scripts.review_frontend_canvas_comparison import metrics


ROOT=Path(__file__).resolve().parents[1];IDENT='local-lattice-complete-v1'
PLAN=ROOT/f'bench/rhythm-v2/{IDENT}.json';REPORT=ROOT/'reports/local_lattice_complete_v1.json';ART=ROOT/f'artifacts/{IDENT}'
FIELD_NAMES={0:'density_notes_per_second',2:'hold_head_ratio',6:'flick_ratio',7:'drag_ratio'}


def fresh(plan,device='mps'):
    torch.manual_seed(plan['seed'])
    model=LatticeCanvas(FullEventConfig(**plan['model']),density_reference=plan['density_reference'],type_prior=plan['type_prior'])
    prior=torch.tensor(plan['positive_count_prior'],dtype=torch.float32)
    with torch.no_grad():
        model.presence.weight.zero_();model.presence.bias.fill_(math.log(plan['occupied_fraction']/(1-plan['occupied_fraction'])))
        model.multiplicity.weight.zero_();model.multiplicity.bias.copy_(prior.log())
    return model.to(device)


def journal(text):
    with (ROOT/'EXPERIMENT_LOG.md').open('a') as f:f.write(f'\n\n{IDENT} {text}\n')


def checkpoint(model,opt,stage,step):
    path=ART/f'{stage}-{step:04d}.pt';path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),stage=stage,updates=step,plan_sha256=sha(PLAN),cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state()),path)
    return path


def diagnostics(model,batches):
    model.eval();out=[]
    with torch.no_grad():
        for e,inp,tgt in batches:
            loss,parts=lattice_loss(model,inp,tgt);out.append(dict(index=e['index'],loss=float(loss),parts={k:float(v) for k,v in parts.items()}))
    return out


def evaluate(model,plan,row,panel,deadline):
    dev=plan['indices']['dev'];sel=dev[::2]
    tasks={'draft':[(i,'draft',None,False) for i in dev],'empty':[(i,'draft',None,False) for i in plan['indices']['empty_dev']],
        'completion':[(i,'middle',None,False) for i in sel],'reference_masked':[(i,'middle',None,True) for i in sel],
        'density_low':[(i,'draft',(2.,3.),False) for i in sel[::4]],'density_high':[(i,'draft',(6.,8.),False) for i in sel[::4]]}
    summaries={}
    for task,specs in tasks.items():
        records=[]
        for index,view,bands,masked in specs:
            if time.monotonic()>deadline:raise TimeoutError('Frozen lattice evaluation wall')
            e=explicit_row(row(index,view,0,bands));inp,_=lattice_batch(e,mask_reference=masked)
            try:
                pred=generate_lattice(model,inp,e['request']);prediction=dict(complete=True,events=[asdict(v) for v in pred])
            except (ValueError,RuntimeError) as exc:pred=[];prediction=dict(complete=False,events=[],failure=str(exc))
            s=score(pred,e['targets']);start=e['request'].start_us;width=e['request'].end_us-start
            shifts=[match_f1([start+(v.onset_us-start+d)%width for v in pred],[v.onset_us for v in e['targets']],10000)[3] for d in plan['shifts_us']]
            _,raw=training_labels('phira',canonical(pred,'phira'),(e['request'].start_us,e['request'].end_us))
            records.append(dict(index=index,component=panel.records[index]['component'],view=view,request=asdict(e['request']),prediction=prediction,target=[asdict(v) for v in e['targets']],scores=s,shifted_onset10=shifts,realized_controls=raw))
            write(ART/f'{task}-final.json',records)
        summaries[task]=metrics(records)
    # Fixed external requests derive their bands from train-only coverage, never
    # from validation hidden attributes. Counterfactual original-chart F1 is not
    # treated as the uniquely correct target.
    conditions=[]
    for field in (2,6,7):
        name=FIELD_NAMES[field]
        for band in plan['ratio_probe_bands'][name]:
            for index in dev[::8]:
                if time.monotonic()>deadline:raise TimeoutError('Lattice conditions deadline')
                e=explicit_row(row(index,'draft'));inp,_=lattice_batch(e)
                inp['controls'][0,field,:3]=torch.tensor([band[0],band[1],1.],device='mps')
                try:pred=generate_lattice(model,inp,e['request']);complete=True;failure=None
                except (ValueError,RuntimeError) as exc:pred=[];complete=False;failure=str(exc)
                _,raw=training_labels('phira',canonical(pred,'phira'),(e['request'].start_us,e['request'].end_us))
                conditions.append(dict(index=index,field=name,requested=band,realized=raw,complete=complete,failure=failure,events=[asdict(v) for v in pred]))
                write(ART/'ratio-probes-final.json',conditions)
    summaries['ratio_probes']=dict(attempted=len(conditions),complete=sum(v['complete'] for v in conditions),role='Response only; no validation-label-derived condition or unique counterfactual fidelity target')
    return summaries


def main():
    if PLAN.exists() or REPORT.exists():raise FileExistsError('Preserve lattice attempt')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('No silent fallback')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5)
    parent=json.loads((ROOT/'bench/rhythm-v2/local-portfolio-scale-v1.json').read_text())
    panel,features=setup(json.loads((ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()));row=row_factory(panel,features)
    type_hist=np.zeros(4,np.int64);count_hist=np.zeros(13,np.int64);frames=occupied=0;audited=0
    for specs in parent['sample_stream']['train64'][:1000]:
        for spec in specs:
            e=row(spec['index'],spec['view'],spec['field_index']);target=lattice_target(e['targets'],e['request']);audited+=1
    for index in parent['indices']['train64']:
        e=row(index,'draft');t=lattice_target(e['targets'],e['request']);n=(e['request'].end_us-e['request'].start_us+9999)//10000
        count_hist+=np.bincount(t['count'][:n],minlength=13);frames+=n;occupied+=int((t['count'][:n]>0).sum())
        type_hist+=np.bincount(t['kind'][t['active']],minlength=4)
    support=json.loads((ROOT/'reports/local_portfolio_condition_support_v1.json').read_text())
    ratio_bands={}
    for field in (2,6,7):
        name=FIELD_NAMES[field];bands=sorted(json.loads(v) for v in support['fields'][name]['supported_bands'])
        if len(bands)<2:raise ValueError('No train-supported bilateral ratio probe: '+name)
        ratio_bands[name]=[bands[0],bands[-1]]
    config={**parent['models']['capacity64'],'d':160,'heads':4,'kv_heads':2,'ffn':512,'encoder_layers':1}
    sources=('scripts/run_local_lattice_experiment.py','beatfcn/model/local_lattice.py','beatfcn/train/lattice_targets.py','beatfcn/train/canonical_probability.py','beatfcn/model/local_typed_canvas.py')
    plan=dict(id=IDENT,seed=20261057,model=config,density_reference=parent['density_reference'],type_prior=((type_hist+.5)/(type_hist.sum()+2)).tolist(),
        occupied_fraction=occupied/frames,positive_count_prior=((count_hist[1:]+.5)/(count_hist[1:].sum()+6)).tolist(),
        count_histogram=count_hist.tolist(),actual_training_views_audited=audited,indices=parent['indices'],sample_stream=parent['sample_stream']['train64'][:1000],
        health_indices=parent['health_indices'],empty_health_index=parent['empty_health_index'],health_steps=100,quality_steps=1000,effective_batch=2,
        learning_rate=.0003,minimum_learning_rate=.00003,warmup=50,weight_decay=.01,clip=1.,health_wall_s=900,quality_wall_s=3600,evaluation_wall_s=900,max_parameters=3_000_000,
        counterfactual_probability=.5,counterfactual_weight=.3,ratio_probe_bands=ratio_bands,shifts_us=parent['shifts_us'],
        hypothesis='Complete10ms parallel lattice,12member mixed-type slots, native/COPY residue and relative legal tail, Focal+train-log-prior objectives and explicit density/type counterfactual count hinges jointly avoid AR attractors.',
        interpretation='Complete-candidate screen, not component causality. Focal/class-balanced action logits are not claimed to be calibrated natural probabilities. Health disables counterfactual regularization; quality enables it.',
        health_gate='100 real5request updates: loss decreases10%, type/residual do not worsen5%, finite CPU/MPS<=1e-4 and5 legal direct outputs. No productionF1 or type-presence threshold.',
        cloud_compute_calls=0,confirmation_scored=False,source_sha256={n:sha(ROOT/n) for n in sources})
    write(PLAN,plan)
    for name in sources:
        dest=ART/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    shutil.copy2(PLAN,ART/'source/frozen-plan.json')
    model=fresh(plan);torch.mps.manual_seed(plan['seed']);opt=optimizer_for(model,plan);n=sum(p.numel() for p in model.parameters())
    if n>plan['max_parameters']:raise ValueError('Lattice parameter cap')
    report=dict(id=IDENT,status='health_running',parameters=n,plan_sha256=sha(PLAN),initial_weight_sha256=weight_digest(model),health_updates=0,quality_updates=0,curves=[],failures=[],health_weights_used=False,counterfactual_exposure=0,cloud_compute_calls=0,confirmation_scored=False)
    write(REPORT,report);journal(f'完整10ms路线冻结，{n}参数，2000真实训练视图容量审计；100健康后fresh1000；不追认其他网格为本路线。')
    def stopped(*_):raise KeyboardInterrupt('External interruption')
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped);stage='health';started=time.monotonic()
    try:
        specs=list(zip(plan['health_indices'],('draft','single_condition','random_groups','middle')))+[(plan['empty_health_index'],'draft')];batches=[]
        for index,view in specs:
            e=explicit_row(row(index,view));inp,tgt=lattice_batch(e);batches.append((e,inp,tgt))
        report['initial_diagnostic']=diagnostics(model,batches)
        cpu=fresh(plan,'cpu').eval();e,inp,tgt=batches[0];ci,ct=({k:v.cpu() for k,v in d.items()} for d in (inp,tgt))
        with torch.no_grad():a,b=cpu(ci),model(inp)
        forward=max(float((a[k]-b[k].cpu()).abs().max()) for k in a)
        cl,_=lattice_loss(cpu,ci,ct);ml,_=lattice_loss(model,inp,tgt);cl.backward();ml.backward();cp=dict(cpu.named_parameters());gradient=0.
        for name,p in model.named_parameters():
            if (p.grad is None)!=(cp[name].grad is None):raise ValueError('Gradient presence parity')
            if p.grad is not None:gradient=max(gradient,float((p.grad.cpu()-cp[name].grad).abs().max()))
        report['parity']=dict(forward=forward,gradient=gradient)
        if max(forward,gradient)>1e-4:raise ValueError('CPU/MPS parity')
        del cpu,a,b,cl,ml,cp
        for step in range(1,101):
            if time.monotonic()-started>plan['health_wall_s']:raise TimeoutError('Health wall')
            model.train();opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']=rate(plan,step)
            total=0.
            for e,inp,tgt in batches:
                loss,_=lattice_loss(model,inp,tgt);(loss/5).backward();total+=float(loss.detach())/5
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();report['health_updates']=step
            if step%10==0 or step==1:
                point=dict(stage=stage,step=step,loss=total,gradient=grad,wall_s=time.monotonic()-started);report['curves'].append(point);write(REPORT,report);print(json.dumps(point),flush=True)
            if step%25==0:checkpoint(model,opt,stage,step)
        report['final_diagnostic']=diagnostics(model,batches);preds=[]
        for e,inp,tgt in batches:
            pred=generate_lattice(model,inp,e['request']);preds.append(dict(index=e['index'],prediction=dict(complete=True,events=[asdict(v) for v in pred]),target=[asdict(v) for v in e['targets']],scores=score(pred,e['targets'])))
        write(ART/'health-predictions.json',preds);a,b=report['initial_diagnostic'],report['final_diagnostic']
        gates=dict(loss=bool(np.mean([v['loss'] for v in b])<=.9*np.mean([v['loss'] for v in a])),complete=len(preds)==5)
        for part in ('kind','residual'):gates[part]=bool(np.mean([v['parts'][part] for v in b])<=1.05*np.mean([v['parts'][part] for v in a]))
        report.update(health_passed=all(gates.values()),health_gates=gates,health_wall_s=time.monotonic()-started)
        if not report['health_passed']:
            report['status']='health_failed';write(REPORT,report);journal(f'健康失败结案{gates}。');return
        checkpoint(model,opt,'health',100);del model,opt,batches;torch.mps.empty_cache();model=fresh(plan);torch.mps.manual_seed(plan['seed']);opt=optimizer_for(model,plan)
        if weight_digest(model)!=report['initial_weight_sha256']:raise ValueError('Fresh initialization mismatch')
        stage='quality';started=time.monotonic();report['status']='quality_running';write(REPORT,report);rng=np.random.default_rng(plan['seed']);journal('健康通过；fresh1000、counterfactual hinge启用。')
        for step in range(1,1001):
            if time.monotonic()-started>plan['quality_wall_s']:raise TimeoutError('Quality wall')
            model.train();opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']=rate(plan,step)
            total=0.
            for spec in plan['sample_stream'][step-1]:
                e=explicit_row(row(spec['index'],spec['view'],spec['field_index']));inp,tgt=lattice_batch(e);loss,_=lattice_loss(model,inp,tgt)
                if rng.random()<.5:
                    bands,_=training_labels('phira',canonical(e['targets']+e.get('visible',[])+list(e['request'].known_carry),'phira'),(e['request'].start_us,e['request'].end_us))
                    eligible=[f for f,name in FIELD_NAMES.items() if name in bands]
                    field=int(rng.choice(eligible));name=FIELD_NAMES[field]
                    current=bands.get(name);choices=[(lo,hi) for lo,hi in zip(BAND_EDGES[name],BAND_EDGES[name][1:]) if current is None or hi<=current[0] or lo>=current[1]]
                    lower,upper=choices[int(rng.integers(len(choices)))];cf={**inp,'controls':inp['controls'].clone()}
                    scale=64 if field==0 else 1;cf['controls'][0,field,:3]=torch.tensor([lower/scale,upper/scale,1.],device='mps')
                    hinge=counterfactual_hinge(model(cf),cf,field_index=field,lower=lower,upper=upper);loss=loss+.3*hinge;report['counterfactual_exposure']+=1
                (loss/2).backward();total+=float(loss.detach())/2
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();report['quality_updates']=step
            if step%25==0 or step==1:
                point=dict(stage=stage,step=step,loss=total,gradient=grad,wall_s=time.monotonic()-started);report['curves'].append(point);write(REPORT,report);print(json.dumps(point),flush=True)
            if step%50==0:checkpoint(model,opt,stage,step)
        path=checkpoint(model,opt,'quality',1000);report['final_weight_sha256']=sha(path);report['status']='evaluating';write(REPORT,report)
        moment=time.monotonic();result=evaluate(model,plan,row,panel,min(started+plan['quality_wall_s'],moment+plan['evaluation_wall_s']))
        report.update(status='completed',evaluations=result,evaluation_wall_s=time.monotonic()-moment,total_quality_wall_s=time.monotonic()-started)
        write(REPORT,report);journal('固定1000及全部自由评价完成；无需全部产品能力作为出场门，不改阈值/温度/配额。')
    except BaseException as exc:
        report.update(status='stopped',stopped_stage=stage,stage_wall_s=time.monotonic()-started);report['failures'].append(f'{type(exc).__name__}: {exc}')
        checkpoint(model,opt,f'{stage}-interrupted',report[f'{stage}_updates']);write(REPORT,report);journal(f'停止保留现场：{report["failures"][-1]}。');raise


if __name__=='__main__':main()
