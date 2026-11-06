"""Complete SSM-context/continuous-flow candidate with a reversible event code."""
from dataclasses import asdict
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from beatfcn.model.event_code_flow import EventCodeFlow,encode_event_code,decode_event_code,flow_loss,generate_flow
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.sample import match_f1
from beatfcn.train.canonical_probability import explicit_row,probability_batch
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_group_diffusion_health import optimizer_for,rate
from scripts.run_local_portfolio_scale import row_factory,weight_digest,sha,write
from scripts.run_local_typed_canvas import score
from scripts.review_frontend_canvas_comparison import metrics


ROOT=Path(__file__).resolve().parents[1];IDENT='local-event-code-flow-v1'
PLAN=ROOT/f'bench/rhythm-v2/{IDENT}.json';REPORT=ROOT/'reports/local_event_code_flow_v1.json';ART=ROOT/f'artifacts/{IDENT}'


def fresh(plan,device='mps'):
    torch.manual_seed(plan['seed'])
    return EventCodeFlow(FullEventConfig(**plan['model']),density_reference=plan['density_reference']).to(device)


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
            code=encode_event_code(tgt,e['request']);noise=torch.zeros_like(code);t=code.new_tensor([.5])
            loss=flow_loss(model,inp,tgt,e['request'],time_value=t,noise=noise)
            out.append(dict(index=e['index'],velocity_mse=float(loss),noise='fixed zero',time=.5))
    return out


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
                if time.monotonic()>deadline:raise TimeoutError('Frozen flow evaluation wall')
                e=explicit_row(row(index,view,0,bands));inp,_=probability_batch([e],'mps',mask_reference=masked)
                try:
                    events,info=generate_flow(model,inp,e['request'],seed=plan['seed']+17*index+sample,steps=16,deadline=deadline)
                    prediction=dict(complete=True,events=[asdict(v) for v in events],diagnostics=info)
                except (ValueError,RuntimeError,TimeoutError) as exc:events=[];prediction=dict(complete=False,events=[],failure=f'{type(exc).__name__}: {exc}')
                start=e['request'].start_us;width=e['request'].end_us-start
                shifts=[match_f1([start+(v.onset_us-start+d)%width for v in events],[v.onset_us for v in e['targets']],10000)[3] for d in plan['shifts_us']]
                records.append(dict(index=index,component=panel.records[index]['component'],view=view,request=asdict(e['request']),prediction=prediction,target=[asdict(v) for v in e['targets']],scores=score(events,e['targets']),shifted_onset10=shifts))
                write(ART/f'{task}-sample{sample}.json',records)
            samples.append(metrics(records))
        summary={field:float(np.mean([s[field] for s in samples])) for field in ('full_note_f1','onset10','onset50','shift_null10','density_ratio','nonempty','complete','attempted')}
        summary.update(samples=samples,components={c:{field:float(np.mean([s['components'][c][field] for s in samples])) for field in ('full_note_f1','onset10','onset50','shift_null10')} for c in samples[0]['components']})
        summaries[task]=summary
    return summaries


def main():
    global IDENT,PLAN,REPORT,ART
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repair',action='store_true')
    repair=parser.parse_args().repair;previous=original=None
    if repair:
        previous=json.loads(REPORT.read_text());original=json.loads(PLAN.read_text())
        if previous['status']!='stopped' or previous['health_updates']!=0 or not any('index_copy.out' in v for v in previous['failures']):raise ValueError('Repair restricted to zero-update unsupportedMPScontext operator')
        before=ast.parse((ART/'source/beatfcn/model/event_code_flow.py').read_text());after=ast.parse((ROOT/'beatfcn/model/event_code_flow.py').read_text())
        for name in ('encode_event_code','decode_event_code'):
            a=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name==name)
            if ast.dump(a)!=ast.dump(b):raise ValueError('Context-only repair changed the already audited event codec')
        IDENT='local-event-code-flow-v2';PLAN=ROOT/f'bench/rhythm-v2/{IDENT}.json'
        REPORT=ROOT/'reports/local_event_code_flow_v2.json';ART=ROOT/f'artifacts/{IDENT}'
    if PLAN.exists() or REPORT.exists():raise FileExistsError('Preserve flow candidate')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('No silent backend fallback')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5)
    parent=json.loads((ROOT/'bench/rhythm-v2/local-portfolio-mamba-quality-v1.json').read_text())
    panel,features=setup(json.loads((ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()));row=row_factory(panel,features)
    audited=0
    # Code fidelity precedes learning. Every training view's native/COPY event
    # multiset must roundtrip, not merely have valid tensor shapes.
    if repair:
        audited=original['actual_target_roundtrips']
        if original['model']!=parent['model'] or original['sample_stream']!=parent['sample_stream'][:1000]:raise ValueError('Repair changed data/model')
    else:
        for specs in parent['sample_stream'][:1000]:
            for spec in specs:
                e=explicit_row(row(spec['index'],spec['view'],spec['field_index']));inp,tgt=probability_batch([e],'cpu')
                code=encode_event_code(tgt,e['request'])
                if decode_event_code(code,inp,e['request'])!=sorted(e['targets']):raise ValueError('Actual-view event-code roundtrip')
                audited+=1
    sources=('scripts/run_event_code_flow.py','beatfcn/model/event_code_flow.py','beatfcn/model/local_mamba_canvas.py','beatfcn/model/mamba2_reference.py','beatfcn/train/canonical_probability.py')
    plan=dict(id=IDENT,seed=20261058,model=parent['model'],density_reference=parent['density_reference'],indices=parent['indices'],sample_stream=parent['sample_stream'][:1000],
        health_indices=json.loads((ROOT/'bench/rhythm-v2/local-portfolio-scale-v1.json').read_text())['health_indices'],
        empty_health_index=json.loads((ROOT/'bench/rhythm-v2/local-portfolio-scale-v1.json').read_text())['empty_health_index'],
        health_steps=100,quality_steps=1000,effective_batch=2,learning_rate=.0003,minimum_learning_rate=.00003,warmup=50,weight_decay=.01,clip=1.,
        health_wall_s=1200,quality_wall_s=5400,evaluation_wall_s=1200,max_parameters=12_000_000,actual_target_roundtrips=audited,
        noise='Independent standard Gaussian',path='x(t)=(1-t)z+t*event_code; target velocity=event_code-z',ode=dict(method='Euler',steps=16,primary_seeds=4,equal_weighted=True,audio_encodes=1),
        code='One normalized count+16 complete canonical native/COPY members per40ms/type; deterministic float32-reversible81-vector, no learned VAE; final nearest legal code retains duplicates and native1ms/COPY times.',
        hypothesis='Single SSD context encoding and a lightweight repeated continuous velocity field jointly generate complete event codes, instead of independent slot decisions or masked diffusion.',
        boundaries='This tests a deterministic reversible code, not a learned VAE latent. Sparse code geometry may make count/millisecond learning hard. Whole recipe only, no claim that SSD or flow alone caused a result. No clipping counts to a requested quota, no NMS/min-gap/dedup.',
        health_gate='100 real5request updates; fixedzero-noise t=.5 velocityMSE decreases10%, finite CPU/MPS<=1e-4, and5 full16step free transports complete. No releaseF1 requirement.',
        shifts_us=parent['shifts_us'],cloud_compute_calls=0,confirmation_scored=False,source_sha256={n:sha(ROOT/n) for n in sources})
    if repair:
        plan.update(health_wall_s=1200-previous['stage_wall_s']-60,
            correctness_repair=dict(reason='Replace unsupportedMPSindex_copy with mathematically identical functional padding on audited contiguous-prefix request frames. No backend fallback or scientific change.',
                original_report='reports/local_event_code_flow_v1.json',original_report_sha256=sha(ROOT/'reports/local_event_code_flow_v1.json'),
                charged_previous_wall_s=previous['stage_wall_s'],diagnostic_reserve_s=60,reused_original_target_roundtrips=audited,codec_AST_unchanged=True))
    write(PLAN,plan)
    for name in sources:
        dest=ART/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    shutil.copy2(PLAN,ART/'source/frozen-plan.json')
    model=fresh(plan);torch.mps.manual_seed(plan['seed']);opt=optimizer_for(model,plan);n=sum(p.numel() for p in model.parameters())
    if n>plan['max_parameters']:raise ValueError('Flow parameter cap')
    report=dict(id=IDENT,status='health_running',parameters=n,plan_sha256=sha(PLAN),initial_weight_sha256=weight_digest(model),health_updates=0,quality_updates=0,curves=[],failures=[],health_weights_used=False,cloud_compute_calls=0,confirmation_scored=False)
    if previous is not None and report['initial_weight_sha256']!=previous['initial_weight_sha256']:raise ValueError('Correctness repair changed initial parameters')
    write(REPORT,report);journal(f'完整SSD+连续flow冻结，{n}参数，{audited}真实视图无损；100健康后fresh1000；四种子16步Euler，无云。')
    def stopped(*_):raise KeyboardInterrupt('External interruption')
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped);stage='health';started=time.monotonic()
    try:
        specs=list(zip(plan['health_indices'],('draft','single_condition','random_groups','middle')))+[(plan['empty_health_index'],'draft')];batches=[]
        for index,view in specs:
            e=explicit_row(row(index,view));inp,tgt=probability_batch([e],'mps');batches.append((e,inp,tgt))
        report['initial_diagnostic']=diagnostics(model,batches)
        cpu=fresh(plan,'cpu').eval();e,inp,tgt=batches[0];ci,ct=({k:v.cpu() for k,v in d.items()} for d in (inp,tgt))
        code=encode_event_code(ct,e['request']);noise=torch.zeros_like(code);t=torch.tensor([.5])
        with torch.no_grad():a=cpu(ci,code*.5,t);b=model(inp,code.to('mps')*.5,t.to('mps'))
        forward=float((a-b.cpu()).abs().max());cl=flow_loss(cpu,ci,ct,e['request'],time_value=t,noise=noise);ml=flow_loss(model,inp,tgt,e['request'],time_value=t.to('mps'),noise=noise.to('mps'))
        cl.backward();ml.backward();cp=dict(cpu.named_parameters());gradient=0.
        for name,p in model.named_parameters():
            if (p.grad is None)!=(cp[name].grad is None):raise ValueError('Gradient presence parity')
            if p.grad is not None:gradient=max(gradient,float((p.grad.cpu()-cp[name].grad).abs().max()))
        report['parity']=dict(forward=forward,gradient=gradient)
        if max(forward,gradient)>1e-4:raise ValueError('CPU/MPS flow parity')
        del cpu,a,b,cl,ml,cp
        for step in range(1,101):
            if time.monotonic()-started>plan['health_wall_s']:raise TimeoutError('Health wall')
            model.train();opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']=rate(plan,step)
            total=0.
            for e,inp,tgt in batches:
                loss=flow_loss(model,inp,tgt,e['request']);(loss/5).backward();total+=float(loss.detach())/5
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();report['health_updates']=step
            if step%10==0 or step==1:
                point=dict(stage=stage,step=step,loss=total,gradient=grad,wall_s=time.monotonic()-started);report['curves'].append(point);write(REPORT,report);print(json.dumps(point),flush=True)
            if step%25==0:checkpoint(model,opt,stage,step)
        report['final_diagnostic']=diagnostics(model,batches);preds=[]
        for e,inp,tgt in batches:
            pred,info=generate_flow(model,inp,e['request'],seed=plan['seed']+e['index'],steps=16,deadline=started+plan['health_wall_s'])
            preds.append(dict(index=e['index'],prediction=dict(complete=True,events=[asdict(v) for v in pred],diagnostics=info),target=[asdict(v) for v in e['targets']],scores=score(pred,e['targets'])))
        write(ART/'health-predictions.json',preds)
        initial=np.mean([v['velocity_mse'] for v in report['initial_diagnostic']]);final=np.mean([v['velocity_mse'] for v in report['final_diagnostic']])
        gates=dict(velocity=bool(final<=initial*.9),complete=len(preds)==5)
        report.update(health_gates=gates,health_passed=all(gates.values()),health_wall_s=time.monotonic()-started)
        if not report['health_passed']:
            report['status']='health_failed';write(REPORT,report);journal(f'健康失败结案{gates}。');return
        checkpoint(model,opt,'health',100);del model,opt,batches;torch.mps.empty_cache();model=fresh(plan);torch.mps.manual_seed(plan['seed']);opt=optimizer_for(model,plan)
        if weight_digest(model)!=report['initial_weight_sha256']:raise ValueError('Fresh initialization mismatch')
        stage='quality';started=time.monotonic();report['status']='quality_running';write(REPORT,report);journal('健康通过；fresh1000质量开始。')
        for step in range(1,1001):
            if time.monotonic()-started>plan['quality_wall_s']:raise TimeoutError('Quality wall')
            model.train();opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']=rate(plan,step)
            total=0.
            for spec in plan['sample_stream'][step-1]:
                e=explicit_row(row(spec['index'],spec['view'],spec['field_index']));inp,tgt=probability_batch([e],'mps')
                loss=flow_loss(model,inp,tgt,e['request']);(loss/2).backward();total+=float(loss.detach())/2
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();report['quality_updates']=step
            if step%25==0 or step==1:
                point=dict(stage=stage,step=step,loss=total,gradient=grad,wall_s=time.monotonic()-started);report['curves'].append(point);write(REPORT,report);print(json.dumps(point),flush=True)
            if step%50==0:checkpoint(model,opt,stage,step)
        path=checkpoint(model,opt,'quality',1000);report['final_weight_sha256']=sha(path);report['status']='evaluating';write(REPORT,report)
        moment=time.monotonic();result=evaluate(model,plan,row,panel,min(started+plan['quality_wall_s'],moment+plan['evaluation_wall_s']))
        report.update(status='completed',evaluations=result,evaluation_wall_s=time.monotonic()-moment,total_quality_wall_s=time.monotonic()-started)
        write(REPORT,report);journal('固定1000及四种子全部自由评价完成，不择优、不加步、不上云。')
    except BaseException as exc:
        report.update(status='stopped',stopped_stage=stage,stage_wall_s=time.monotonic()-started);report['failures'].append(f'{type(exc).__name__}: {exc}')
        checkpoint(model,opt,f'{stage}-interrupted',report[f'{stage}_updates']);write(REPORT,report);journal(f'停止保留现场：{report["failures"][-1]}。');raise


if __name__=='__main__':main()
