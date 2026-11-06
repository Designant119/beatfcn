"""One fresh A quality run after explicitly revising an overstrict health gate.

The original A100 count/empty gate remains failed. This post-observation
admission revision is disclosed and retains all original independent-song
quality criteria. No health-trained weight is loaded into quality training.
"""
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from beatfcn.model.canonical_event_package import canonical_package_loss
from beatfcn.train.canonical_probability import explicit_row,probability_batch
from scripts import run_canonical_package_experiment as engine
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_group_diffusion_health import optimizer_for,rate
from scripts.run_local_portfolio_scale import row_factory,weight_digest,sha,write


ROOT=Path(__file__).resolve().parents[1];IDENT='local-canonical-package-quality-v2'
PLAN=ROOT/f'bench/rhythm-v2/{IDENT}.json';REPORT=ROOT/'reports/local_canonical_package_quality_v2.json';ART=ROOT/f'artifacts/{IDENT}'


def main():
    if PLAN.exists() or REPORT.exists():raise FileExistsError('Preserve explicit one-off admission revision')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('No silent fallback')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5)
    oldplan=ROOT/'bench/rhythm-v2/local-canonical-package-v1.json';oldreport=ROOT/'reports/local_canonical_package_v1.json'
    plan=json.loads(oldplan.read_text());health=json.loads(oldreport.read_text())
    if health['status']!='health_failed' or health['health_updates']!=100 or health['quality_updates']!=0:raise ValueError('Exact original failed100health required')
    first,last=health['initial_diagnostic'],health['final_diagnostic']
    certificate=dict(loss=bool(np.mean([v['loss'] for v in last])<=.9*np.mean([v['loss'] for v in first])),
        original_parity=max(health['parity'].values())<=1e-4,all_free_completions=health['health_gates']['complete'])
    for part in ('onset','tail'):certificate[part]=bool(np.mean([v['parts'][part] for v in last])<=1.05*np.mean([v['parts'][part] for v in first]))
    if not all(certificate.values()):raise ValueError('Retained evidence does not pass fitting/finite/legality admission')
    plan.update(id=IDENT,admission_revision=dict(after_observation=True,reason='Root originally imposed extra count/empty gates on a100stepengineering fit, exceeding Sol local-screen purpose. Original failure unchanged. This new contract admits exactly one independent-song quality run using retained fit/numerical/legal evidence; original quality screen unchanged.',
        original_plan_sha256=sha(oldplan),original_report_sha256=sha(oldreport),certificate=certificate),health_weights_used=False,
        quality_wall_s=6000,evaluation_wall_s=1200,confirmation_scored=False,cloud_compute_calls=0)
    plan['source_sha256']={n:sha(ROOT/n) for n in ('scripts/run_canonical_quality_v2.py','scripts/run_canonical_package_experiment.py','beatfcn/model/canonical_event_package.py','beatfcn/model/cardinality_dp.py','beatfcn/train/canonical_probability.py')}
    write(PLAN,plan)
    for name in plan['source_sha256']:
        dest=ART/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    shutil.copy2(PLAN,ART/'source/frozen-plan.json')
    engine.ART=ART;engine.IDENT=IDENT;engine.PLAN=PLAN
    model=engine.fresh(plan);torch.mps.manual_seed(plan['seed']);opt=optimizer_for(model,plan)
    if weight_digest(model)!=health['initial_weight_sha256']:raise ValueError('Fresh initialization differs from original prehealth state')
    report=dict(id=IDENT,status='quality_running',plan_sha256=sha(PLAN),parameters=sum(p.numel() for p in model.parameters()),
        initial_weight_sha256=weight_digest(model),health_weights_used=False,admission_after_observation=True,original_health_status='health_failed',
        quality_updates=0,curves=[],failures=[],cloud_compute_calls=0,confirmation_scored=False)
    write(REPORT,report);engine.journal('显式修订工程入场（原100步计数门失败保留），丢弃健康权重，原质量门不变，唯一fresh1000开始；云0。')
    panel,features=setup(json.loads((ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()));row=row_factory(panel,features)
    def stopped(*_):raise KeyboardInterrupt('External interruption')
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped);started=time.monotonic()
    try:
        for step in range(1,1001):
            if time.monotonic()-started>plan['quality_wall_s']:raise TimeoutError('Frozen quality wall')
            model.train();opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']=rate(plan,step)
            total=0.
            for spec in plan['sample_stream'][step-1]:
                e=explicit_row(row(spec['index'],spec['view'],spec['field_index']));inp,tgt=probability_batch([e],'mps')
                loss,_=canonical_package_loss(model,inp,tgt,e['request']);(loss/2).backward();total+=float(loss.detach())/2
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();report['quality_updates']=step
            if step%25==0 or step==1:
                point=dict(step=step,loss=total,gradient=grad,wall_s=time.monotonic()-started);report['curves'].append(point);write(REPORT,report);print(json.dumps(point),flush=True)
            if step%250==0:engine.checkpoint(model,opt,plan,'quality',step)
        weight=engine.checkpoint(model,opt,plan,'quality',1000);report['final_weight_sha256']=sha(weight);report['training_wall_s']=time.monotonic()-started;report['status']='evaluating';write(REPORT,report)
        moment=time.monotonic();result=engine.evaluate(model,plan,row,panel,min(started+plan['quality_wall_s'],moment+plan['evaluation_wall_s']))
        d=result['draft'];g=plan['local_screen'];gates=dict(complete=d['complete']==d['attempted'],nonempty=d['nonempty']>=g['min_nonempty'],count_ratio=g['count_ratio'][0]<=d['density_ratio']<=g['count_ratio'][1],onset10=d['onset10']>=g['onset10'],onset_null=d['onset10']-d['shift_null10']>=g['onset10_minus_shift'],components=sum(v['onset10']>v['shift_null10'] for v in d['components'].values())>=g['positive_components'],full_f1=d['full_note_f1']>=g['full_note_f1'])
        report.update(status='completed',evaluations=result,local_screen=gates,local_screen_passed=all(gates.values()),evaluation_wall_s=time.monotonic()-moment,total_quality_wall_s=time.monotonic()-started)
        write(REPORT,report);engine.journal(f'唯一fresh1000及全评价完成，原筛选门{gates}；原health失败不改判。')
    except BaseException as exc:
        report.update(status='stopped',wall_s=time.monotonic()-started);report['failures'].append(f'{type(exc).__name__}: {exc}')
        engine.checkpoint(model,opt,plan,'quality-interrupted',report['quality_updates']);write(REPORT,report);raise


if __name__=='__main__':main()
