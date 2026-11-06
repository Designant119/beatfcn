"""Execute independently grouped AR; no SSD/diffusion mixture or cloud."""
from dataclasses import asdict
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.grouped_chart_process import GroupedChartProcess,grouped_process_loss,generate_grouped_process
from beatfcn.model.sample import match_f1
from beatfcn.train.canonical_probability import explicit_row,probability_batch
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_group_diffusion_health import optimizer_for,prior_for,rate
from scripts.run_local_portfolio_scale import row_factory,weight_digest,sha,write
from scripts.run_local_typed_canvas import score
from scripts.review_frontend_canvas_comparison import metrics


ROOT=Path(__file__).resolve().parents[1];IDENT='local-grouped-chart-v1'
PLAN=ROOT/f'bench/rhythm-v2/{IDENT}.json';REPORT=ROOT/'reports/local_grouped_chart_v1.json';ART=ROOT/f'artifacts/{IDENT}'


def fresh(plan,device='mps'):
    torch.manual_seed(plan['seed'])
    model=GroupedChartProcess(FullEventConfig(**plan['model']),density_reference=plan['density_reference'],history_dropout=.1)
    p=torch.tensor(plan['count_prior'],dtype=torch.float32)
    with torch.no_grad():
        model.count_gate.weight.zero_();model.count_positive.weight.zero_()
        model.count_gate.bias.copy_((1-p[:,0]).log()-p[:,0].log())
        model.count_positive.bias.copy_((p[:,1:]/(1-p[:,:1])).log().flatten())
        model.stop_read.bias.fill_(plan['stop_bias'])
    return model.to(device)


def journal(text):
    with (ROOT/'EXPERIMENT_LOG.md').open('a') as f:f.write(f'\n\n{IDENT} {text}\n')


def checkpoint(model,opt,stage,step):
    path=ART/f'{stage}-{step:04d}.pt';path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),stage=stage,updates=step,plan_sha256=sha(PLAN),
        cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state()),path);return path


def diagnostics(model,batches):
    model.eval();out=[]
    with torch.no_grad():
        for e,inputs in batches:
            loss,parts=grouped_process_loss(model,inputs,e['targets'],e['request'])
            out.append(dict(index=e['index'],loss=float(loss),parts={k:float(v) for k,v in parts.items()}))
    return out


def evaluate(model,plan,row,panel,deadline):
    dev=plan['indices']['dev'];sel=dev[::2]
    tasks={'draft':[(i,'draft',None,False,False) for i in dev],
        'empty':[(i,'draft',None,False,False) for i in plan['indices']['empty_dev']],
        'completion':[(i,'middle',None,False,False) for i in sel],
        'reference_masked':[(i,'middle',None,True,False) for i in sel],
        'density_low':[(i,'draft',(2.,3.),False,False) for i in sel[::4]],
        'density_high':[(i,'draft',(6.,8.),False,False) for i in sel[::4]],
        'greedy':[(i,'draft',None,False,True) for i in dev]}
    summaries={}
    for task,specs in tasks.items():
        samples=[]
        for sample in range(1 if task=='greedy' else 4):
            records=[]
            for index,view,band,masked,greedy in specs:
                if time.monotonic()>deadline:raise TimeoutError('Frozen grouped evaluation wall')
                e=explicit_row(row(index,view,0,band));inp,_=probability_batch([e],'mps',mask_reference=masked)
                try:
                    pred,info=generate_grouped_process(model,inp,e['request'],seed=plan['seed']+17*index+sample,greedy=greedy,deadline=deadline)
                    prediction=dict(complete=True,events=[asdict(v) for v in pred],diagnostics=info)
                except (ValueError,RuntimeError,TimeoutError) as exc:
                    pred=[];prediction=dict(complete=False,events=[],failure=f'{type(exc).__name__}: {exc}')
                s=score(pred,e['targets']);start=e['request'].start_us;width=e['request'].end_us-start
                s['extra_onset_tolerances']={str(ms):match_f1([v.onset_us for v in pred],[v.onset_us for v in e['targets']],ms*1000)[3] for ms in (1,5,25)}
                shifts=[match_f1([start+(v.onset_us-start+d)%width for v in pred],[v.onset_us for v in e['targets']],10000)[3] for d in plan['shifts_us']]
                records.append(dict(index=index,component=panel.records[index]['component'],view=view,request=asdict(e['request']),prediction=prediction,target=[asdict(v) for v in e['targets']],scores=s,shifted_onset10=shifts))
                write(ART/f'{task}-sample{sample}.json',records)
            samples.append(metrics(records))
        summary={field:float(np.mean([s[field] for s in samples])) for field in ('full_note_f1','onset10','onset50','shift_null10','density_ratio','nonempty','complete','attempted')}
        summary.update(samples=samples,components={c:{field:float(np.mean([s['components'][c][field] for s in samples])) for field in ('full_note_f1','onset10','onset50','shift_null10')} for c in samples[0]['components']})
        summaries[task]=summary
    # Teacher-only dependence diagnostic on fixed four train health requests.
    batches=[]
    for index in plan['health_indices']:
        e=explicit_row(row(index,'draft'));inp,_=probability_batch([e],'mps');batches.append((e,inp))
    model.history_content=True;control=diagnostics(model,batches)
    model.history_content=False;no_history=diagnostics(model,batches);model.history_content=True
    no_audio=diagnostics(model,[(e,{**inp,**{k:torch.zeros_like(inp[k]) for k in ('local','song','mel')}}) for e,inp in batches])
    summaries['teacher_diagnostics']=dict(real=control,history_removed=no_history,all_audio_removed=no_audio,interpretation='Same weights, clocks/reference fixed; teacher histories only, not free quality evidence')
    return summaries


def main():
    global IDENT,PLAN,REPORT,ART
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repair',action='store_true')
    repair=parser.parse_args().repair;previous=None
    if repair:
        previous=json.loads(REPORT.read_text())
        if previous['status']!='stopped' or previous['health_updates']!=0 or not any('CPU/MPS parity' in v for v in previous['failures']):
            raise ValueError('Repair only the documented zero-update phase-clock parity failure')
        IDENT='local-grouped-chart-v2';PLAN=ROOT/f'bench/rhythm-v2/{IDENT}.json'
        REPORT=ROOT/'reports/local_grouped_chart_v2.json';ART=ROOT/f'artifacts/{IDENT}'
    if PLAN.exists() or REPORT.exists():raise FileExistsError('Preserve grouped experiment')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('No silent fallback')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5)
    parent=json.loads((ROOT/'bench/rhythm-v2/local-portfolio-scale-v1.json').read_text())
    panel,features=setup(json.loads((ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()));row=row_factory(panel,features)
    histogram,prior=prior_for(panel,parent['indices']['train64'])
    group_mean=np.mean([len({e.onset_us for e in panel.targets(i)}) for i in parent['indices']['train64']])
    sources=('scripts/run_grouped_chart_experiment.py','beatfcn/model/grouped_chart_process.py','beatfcn/model/canonical_event_package.py','beatfcn/train/canonical_probability.py','beatfcn/model/local_event_package.py','beatfcn/model/local_typed_canvas.py')
    plan=dict(id=IDENT,seed=20261055,model=parent['models']['capacity64'],density_reference=parent['density_reference'],
        count_prior=prior.tolist(),count_histogram=histogram.tolist(),stop_bias=math.log(8000/max(group_mean,1)),train_mean_onset_groups=float(group_mean),
        indices=parent['indices'],sample_stream=parent['sample_stream']['train64'][:1000],health_indices=parent['health_indices'],empty_health_index=parent['empty_health_index'],
        health_steps=100,quality_steps=1000,effective_batch=2,learning_rate=.0003,minimum_learning_rate=.00003,warmup=50,weight_decay=.01,clip=1.,
        health_wall_s=1200,quality_wall_s=6000,evaluation_wall_s=1200,max_parameters=12_000_000,shifts_us=parent['shifts_us'],
        hypothesis='Physical audio candidate queries plus strict same-head mixed multiset AR and legal STOP replace rolling fine-time/type token decisions as one complete alternative.',
        primary='Four fixed T1 ancestral complete charts equal-weighted. Greedy is secondary and never replaces endpoint.',
        history_dropout=dict(health=0.,quality=.1,semantics='Content dropout of complete prior group embedding; not scheduled sampling'),
        health_gate='JointNLL decreases10%, head/count/tail means no worse5%; actual CPU/MPS parity1e-4, finite and5 legal T1 completions; no release-quality or empty-count gate.',
        local_screen=dict(min_nonempty=24,count_ratio=[.5,1.75],onset10=.20,onset10_minus_shift=.05,positive_components=10,full_note_f1=.20),
        cloud_compute_calls=0,confirmation_scored=False,source_sha256={n:sha(ROOT/n) for n in sources})
    if repair:
        plan.update(health_wall_s=1200-previous['stage_wall_s']-120,
            correctness_repair=dict(reason='MPS2.6 rounds absolute-us int remainder; bounded relative phase and exact CPU clock conversion preserve original science and initial weights.',
                original_report='reports/local_grouped_chart_v1.json',original_report_sha256=sha(ROOT/'reports/local_grouped_chart_v1.json'),
                charged_prior_wall_s=previous['stage_wall_s'],charged_diagnostic_reserve_s=120,original_health_updates=0,
                repair_parity='reports/local_grouped_chart_parity_repair_v1.json'))
    write(PLAN,plan)
    for name in sources:
        dest=ART/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    shutil.copy2(PLAN,ART/'source/frozen-plan.json')
    model=fresh(plan);torch.mps.manual_seed(plan['seed']);opt=optimizer_for(model,plan)
    parameters=sum(p.numel() for p in model.parameters())
    if parameters>plan['max_parameters']:raise ValueError('Parameter cap')
    report=dict(id=IDENT,status='health_running',parameters=parameters,plan_sha256=sha(PLAN),initial_weight_sha256=weight_digest(model),health_updates=0,quality_updates=0,curves=[],failures=[],health_weights_used=False,cloud_compute_calls=0,confirmation_scored=False)
    if previous is not None and report['initial_weight_sha256']!=previous['initial_weight_sha256']:
        raise ValueError('Correctness repair changed initial parameters')
    write(REPORT,report);journal(f'冻结完整成组AR，{parameters}参数；100真实健康后fresh1000；四T1主读出、greedy次要；云0。')
    def stopped(*_):raise KeyboardInterrupt('External interruption')
    signal.signal(signal.SIGINT,stopped);signal.signal(signal.SIGTERM,stopped)
    stage='health';started=time.monotonic()
    try:
        model.history_dropout=0.
        specs=list(zip(plan['health_indices'],('draft','single_condition','random_groups','middle')))+[(plan['empty_health_index'],'draft')]
        batches=[]
        for index,view in specs:
            e=explicit_row(row(index,view));inp,_=probability_batch([e],'mps');batches.append((e,inp))
        report['input_audit']=[dict(index=e['index'],shapes={k:list(v.shape) for k,v in inp.items()},finite=all(bool(torch.isfinite(v).all()) for v in inp.values())) for e,inp in batches]
        report['initial_diagnostic']=diagnostics(model,batches)
        cpu=fresh(plan,'cpu').eval();e,inp=batches[0];ci={k:v.cpu() for k,v in inp.items()}
        cl,_=grouped_process_loss(cpu,ci,e['targets'],e['request']);ml,_=grouped_process_loss(model,inp,e['targets'],e['request'])
        forward=abs(float(cl.detach())-float(ml.detach()));cl.backward();ml.backward();cp=dict(cpu.named_parameters());gradient=0.
        for name,p in model.named_parameters():
            if (p.grad is None)!=(cp[name].grad is None):raise ValueError('Gradient presence parity')
            if p.grad is not None:gradient=max(gradient,float((p.grad.cpu()-cp[name].grad).abs().max()))
        report['parity']=dict(joint_nll=forward,gradient=gradient)
        if max(forward,gradient)>1e-4:raise ValueError('CPU/MPS parity')
        del cpu,cl,ml,cp
        for step in range(1,101):
            if time.monotonic()-started>plan['health_wall_s']:raise TimeoutError('Health wall')
            model.train();opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']=rate(plan,step)
            total=0.
            for e,inp in batches:
                loss,_=grouped_process_loss(model,inp,e['targets'],e['request']);(loss/5).backward();total+=float(loss.detach())/5
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();report['health_updates']=step
            if step%10==0 or step==1:
                point=dict(stage=stage,step=step,loss=total,gradient=grad,wall_s=time.monotonic()-started);report['curves'].append(point);write(REPORT,report);print(json.dumps(point),flush=True)
            if step%25==0:checkpoint(model,opt,stage,step)
        report['final_diagnostic']=diagnostics(model,batches);predictions=[]
        for e,inp in batches:
            pred,info=generate_grouped_process(model,inp,e['request'],seed=plan['seed']+e['index'],deadline=started+plan['health_wall_s'])
            predictions.append(dict(index=e['index'],prediction=dict(complete=True,events=[asdict(v) for v in pred],diagnostics=info),target=[asdict(v) for v in e['targets']],scores=score(pred,e['targets'])))
        write(ART/'health-predictions.json',predictions);a,b=report['initial_diagnostic'],report['final_diagnostic']
        gates=dict(loss=bool(np.mean([v['loss'] for v in b])<=.9*np.mean([v['loss'] for v in a])),complete=len(predictions)==5)
        for part in ('group_head_stop','counts','tail'):gates[part]=bool(np.mean([v['parts'][part] for v in b])<=1.05*np.mean([v['parts'][part] for v in a]))
        report.update(health_gates=gates,health_passed=all(gates.values()),health_wall_s=time.monotonic()-started)
        if not report['health_passed']:
            report['status']='health_failed';write(REPORT,report);journal(f'健康失败结案{gates}。');return
        checkpoint(model,opt,'health',100);del model,opt,batches;torch.mps.empty_cache()
        model=fresh(plan);torch.mps.manual_seed(plan['seed']);opt=optimizer_for(model,plan)
        if weight_digest(model)!=report['initial_weight_sha256']:raise ValueError('Fresh initialization mismatch')
        stage='quality';started=time.monotonic();report['status']='quality_running';write(REPORT,report);journal('真实健康通过，fresh1000质量开始。')
        for step in range(1,1001):
            if time.monotonic()-started>plan['quality_wall_s']:raise TimeoutError('Quality wall')
            model.train();opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']=rate(plan,step)
            total=0.
            for spec in plan['sample_stream'][step-1]:
                e=explicit_row(row(spec['index'],spec['view'],spec['field_index']));inp,_=probability_batch([e],'mps')
                loss,_=grouped_process_loss(model,inp,e['targets'],e['request']);(loss/2).backward();total+=float(loss.detach())/2
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();report['quality_updates']=step
            if step%25==0 or step==1:
                point=dict(stage=stage,step=step,loss=total,gradient=grad,wall_s=time.monotonic()-started);report['curves'].append(point);write(REPORT,report);print(json.dumps(point),flush=True)
            if step%50==0:checkpoint(model,opt,stage,step)
        path=checkpoint(model,opt,'quality',1000);report['final_weight_sha256']=sha(path);report['training_wall_s']=time.monotonic()-started;report['status']='evaluating';write(REPORT,report)
        moment=time.monotonic();result=evaluate(model,plan,row,panel,min(started+plan['quality_wall_s'],moment+plan['evaluation_wall_s']))
        d=result['draft'];g=plan['local_screen']
        gates=dict(complete=d['complete']==d['attempted'],nonempty=d['nonempty']>=g['min_nonempty'],count_ratio=g['count_ratio'][0]<=d['density_ratio']<=g['count_ratio'][1],onset10=d['onset10']>=g['onset10'],onset_null=d['onset10']-d['shift_null10']>=g['onset10_minus_shift'],components=sum(v['onset10']>v['shift_null10'] for v in d['components'].values())>=g['positive_components'],full_f1=d['full_note_f1']>=g['full_note_f1'])
        report.update(status='completed',evaluations=result,evaluation_wall_s=time.monotonic()-moment,total_quality_wall_s=time.monotonic()-started,local_screen=gates,local_screen_passed=all(gates.values()))
        write(REPORT,report);journal(f'固定1000及全评价完成，本地筛选{gates}。')
    except BaseException as exc:
        report.update(status='stopped',stopped_stage=stage,stage_wall_s=time.monotonic()-started);report['failures'].append(f'{type(exc).__name__}: {exc}')
        checkpoint(model,opt,f'{stage}-interrupted',report[f'{stage}_updates']);write(REPORT,report);journal(f'停止保留现场：{report["failures"][-1]}。');raise


if __name__=='__main__':main()
