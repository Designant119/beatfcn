"""One fresh whole-package diffusion quality run after disclosed admission revision.

The original five-request count/empty health failure remains failed. Its fit,
finite gradients, parity and complete reverse chains support one bounded
independent-song test, not a retrospective health pass.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from beatfcn.model.local_group_diffusion import group_diffusion_loss,sample_group_diffusion
from beatfcn.model.sample import match_f1
from beatfcn.train.canonical_packages import canonical_batch,explicit_reference_row
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_group_diffusion_health import fresh_model,noisy_for,optimizer_for,rate
from scripts.run_local_portfolio_scale import row_factory,weight_digest,sha,write
from scripts.run_local_typed_canvas import score
from scripts.review_frontend_canvas_comparison import metrics


ROOT=Path(__file__).resolve().parents[1];IDENT='local-group-diffusion-quality-v1'
PLAN=ROOT/f'bench/rhythm-v2/{IDENT}.json';REPORT=ROOT/'reports/local_group_diffusion_quality_v1.json';ART=ROOT/f'artifacts/{IDENT}'


def checkpoint(model,opt,step):
    path=ART/f'quality-{step:04d}.pt';path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),updates=step,
        plan_sha256=sha(PLAN),cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state()),path)
    return path


def evaluate(model,plan,row,panel,deadline):
    dev=plan['indices']['dev'];sel=dev[::2]
    tasks={'draft':[(i,'draft',None,False) for i in dev],'empty':[(i,'draft',None,False) for i in plan['indices']['empty_dev']],
        'completion':[(i,'middle',None,False) for i in sel],'reference_masked':[(i,'middle',None,True) for i in sel],
        'density_low':[(i,'draft',(2.,3.),False) for i in sel[::4]],'density_high':[(i,'draft',(6.,8.),False) for i in sel[::4]]}
    summaries={}
    for task,specs in tasks.items():
        samples=[]
        for sample in range(4):
            records=[]
            for index,view,bands,masked in specs:
                if time.monotonic()>deadline:raise TimeoutError('Diffusion evaluation wall')
                e=explicit_reference_row(row(index,view,0,bands));inp,_=canonical_batch([e],'mps',mask_reference=masked)
                try:
                    pred,info=sample_group_diffusion(model,inp,e['request'],seed=plan['seed']+17*index+sample,deadline=deadline)
                    prediction=dict(complete=True,events=[asdict(v) for v in pred],trace=info['trace'])
                except (ValueError,RuntimeError,TimeoutError) as exc:
                    pred=[];prediction=dict(complete=False,events=[],failure=f'{type(exc).__name__}: {exc}')
                start=e['request'].start_us;width=e['request'].end_us-start
                shifts=[match_f1([start+(v.onset_us-start+d)%width for v in pred],[v.onset_us for v in e['targets']],10000)[3] for d in plan['shifts_us']]
                records.append(dict(index=index,component=panel.records[index]['component'],view=view,request=asdict(e['request']),prediction=prediction,
                    target=[asdict(v) for v in e['targets']],scores=score(pred,e['targets']),shifted_onset10=shifts))
                write(ART/f'{task}-sample{sample}.json',records)
            samples.append(metrics(records))
        fields=('full_note_f1','onset10','onset50','shift_null10','density_ratio','nonempty','complete','attempted')
        summaries[task]={k:float(np.mean([s[k] for s in samples])) for k in fields}
        summaries[task].update(samples=samples,components={c:{k:float(np.mean([s['components'][c][k] for s in samples]))
            for k in ('full_note_f1','onset10','onset50','shift_null10')} for c in samples[0]['components']})
    return summaries


def main():
    if PLAN.exists() or REPORT.exists():raise FileExistsError('No automatic retrain')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('No silent fallback')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5)
    oldplan=ROOT/'bench/rhythm-v2/local-group-diffusion-health-v3.json';oldreport=ROOT/'reports/local_group_diffusion_health_v3.json'
    plan=json.loads(oldplan.read_text());health=json.loads(oldreport.read_text())
    if health['status']!='health_failed' or health['health_updates']!=100 or health['quality_updates']!=0:raise ValueError('Require original failed health')
    certificate=dict(joint_fit=health['gates']['joint_loss'],finite=health['gates']['finite'],complete=health['gates']['ancestral_complete'],parity=max(health['parity'].values())<=1e-4)
    for part in ('onset','tail'):
        a=np.mean([v['parts'][part] for v in health['initial_diagnostic']]);b=np.mean([v['parts'][part] for v in health['final_diagnostic']])
        certificate[part]=bool(b<=a*1.05)
    if not all(certificate.values()):raise ValueError('Retained engineering evidence insufficient')
    parent=json.loads((ROOT/'bench/rhythm-v2/local-portfolio-scale-v1.json').read_text())
    plan.update(id=IDENT,indices=parent['indices'],sample_stream=parent['sample_stream']['train64'][:1000],shifts_us=parent['shifts_us'],effective_batch=2,
        quality_wall_s=5400,evaluation_wall_s=1200,quality_steps=1000,health_weights_used=False,
        admission_revision=dict(after_observation=True,original_status='health_failed',original_plan_sha256=sha(oldplan),original_report_sha256=sha(oldreport),certificate=certificate,
            reason='Count/empty limits on a100stepfit were stricter than engineering admission. Preserve original failure; admit exactly one fresh1000 independent-song run, without balancing, more steps, threshold/temperature changes or loading health weights.'),
        local_screen=dict(min_nonempty=24,count_ratio=[.5,1.75],onset10=.20,onset10_minus_shift=.05,positive_components=12,full_note_f1=.20),
        confirmation_scored=False,cloud_compute_calls=0)
    sources=('scripts/run_group_diffusion_quality.py','scripts/run_local_group_diffusion_health.py','beatfcn/model/local_group_diffusion.py','beatfcn/train/canonical_packages.py')
    plan['source_sha256']={n:sha(ROOT/n) for n in sources};write(PLAN,plan)
    for n in sources:
        dest=ART/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    shutil.copy2(PLAN,ART/'source/frozen-plan.json')
    panel,features=setup(json.loads((ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()));row=row_factory(panel,features)
    model=fresh_model(plan);torch.mps.manual_seed(plan['seed']);opt=optimizer_for(model,plan)
    if weight_digest(model)!=health['initial_weight_sha256']:raise ValueError('Must use identical original prehealth initialization')
    report=dict(id=IDENT,status='quality_running',parameters=sum(p.numel() for p in model.parameters()),plan_sha256=sha(PLAN),initial_weight_sha256=weight_digest(model),
        health_weights_used=False,admission_after_observation=True,quality_updates=0,curves=[],failures=[],cloud_compute_calls=0,confirmation_scored=False)
    write(REPORT,report)
    with (ROOT/'EXPERIMENT_LOG.md').open('a') as f:f.write(f'\n\n{IDENT}: explicit post-observation engineering admission; old count/empty health failure unchanged. One fresh1000, no additional tuning/cloud.\n')
    def stop(*_):raise KeyboardInterrupt('External stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);started=time.monotonic()
    try:
        for step in range(1,1001):
            if time.monotonic()-started>plan['quality_wall_s']:raise TimeoutError('Diffusion quality wall')
            model.train();opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']=rate(plan,step)
            total=0.
            for spec in plan['sample_stream'][step-1]:
                e=explicit_reference_row(row(spec['index'],spec['view'],spec['field_index']));inp,tgt=canonical_batch([e],'mps')
                t=int(torch.randint(1,33,(1,)).item());steps,noisy=noisy_for(inp,tgt,e['request'],t,model)
                loss,_=group_diffusion_loss(model,inp,tgt,e['request'],steps,noisy=noisy);(loss/2).backward();total+=float(loss.detach())/2
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();report['quality_updates']=step
            if step%25==0 or step==1:
                point=dict(step=step,loss=total,gradient=grad,wall_s=time.monotonic()-started);report['curves'].append(point);write(REPORT,report);print(json.dumps(point),flush=True)
            if step%250==0:checkpoint(model,opt,step)
        weight=checkpoint(model,opt,1000);report.update(status='evaluating',final_weight_sha256=sha(weight),training_wall_s=time.monotonic()-started);write(REPORT,report)
        moment=time.monotonic();result=evaluate(model,plan,row,panel,min(started+plan['quality_wall_s'],moment+plan['evaluation_wall_s']))
        d=result['draft'];g=plan['local_screen'];gates=dict(complete=d['complete']==d['attempted'],nonempty=d['nonempty']>=g['min_nonempty'],count_ratio=g['count_ratio'][0]<=d['density_ratio']<=g['count_ratio'][1],onset10=d['onset10']>=g['onset10'],onset_null=d['onset10']-d['shift_null10']>=g['onset10_minus_shift'],components=sum(v['onset10']>v['shift_null10'] for v in d['components'].values())>=g['positive_components'],full_f1=d['full_note_f1']>=g['full_note_f1'])
        report.update(status='completed',evaluations=result,local_screen=gates,local_screen_passed=all(gates.values()),evaluation_wall_s=time.monotonic()-moment,total_quality_wall_s=time.monotonic()-started);write(REPORT,report)
    except BaseException as exc:
        report.update(status='stopped',wall_s=time.monotonic()-started);report['failures'].append(f'{type(exc).__name__}: {exc}');checkpoint(model,opt,report['quality_updates']);write(REPORT,report);raise


if __name__=='__main__':main()
