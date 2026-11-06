"""Execute canonical count-law + cardinality decision, health then fresh quality.

All scientific decisions are frozen before any optimizer update. Failed health
closes this attempt; quality is not manufactured from teacher-forced fitting.
"""
from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from beatfcn.model.canonical_event_package import CanonicalPackageCanvas,canonical_package_loss,generate_canonical,package_support
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.sample import match_f1
from beatfcn.train.canonical_probability import explicit_row,probability_batch,encode_canonical
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_group_diffusion_health import prior_for,optimizer_for,rate
from scripts.run_local_portfolio_scale import row_factory,weight_digest,sha,write
from scripts.run_local_typed_canvas import score
from scripts.review_frontend_canvas_comparison import metrics


ROOT=Path(__file__).resolve().parents[1]
IDENT='local-canonical-package-v1'
PLAN=ROOT/f'bench/rhythm-v2/{IDENT}.json'
REPORT=ROOT/'reports/local_canonical_package_v1.json'
ART=ROOT/f'artifacts/{IDENT}'
MODEL_OVERRIDE=None
EXTRA_SOURCES=()
COUNT_HEALTH_GATES=True


def fresh(plan,device='mps'):
    torch.manual_seed(plan['seed'])
    model=CanonicalPackageCanvas(FullEventConfig(**plan['model']),density_reference=plan['density_reference'])
    prior=torch.tensor(plan['count_prior'],dtype=torch.float32)
    with torch.no_grad():
        model.count_gate.weight.zero_();model.count_positive.weight.zero_()
        model.count_gate.bias.copy_((1-prior[:,0]).log()-prior[:,0].log())
        model.count_positive.bias.copy_((prior[:,1:]/(1-prior[:,:1])).log().flatten())
    return model.to(device)


def diagnostics(model,batches):
    model.eval();records=[]
    with torch.no_grad():
        for e,inputs,target in batches:
            loss,parts=canonical_package_loss(model,inputs,target,e['request'])
            output=model(inputs);valid=package_support(inputs,e['request'])
            expected=(output['count']*torch.arange(17,device=inputs['mode'].device)).sum(-1)
            records.append(dict(index=e['index'],loss=float(loss),expected=float((expected*valid).sum()),
                parts={k:float(v) for k,v in parts.items()}))
    return records


def checkpoint(model,opt,plan,stage,step):
    path=ART/f'{stage}-{step:04d}.pt';path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),plan_sha256=sha(PLAN),
        stage=stage,updates=step,cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state()),path)
    return path


def prediction(model,inputs,request,decision,seed,deadline):
    try:
        events,info=generate_canonical(model,inputs,request,decision=decision,seed=seed,deadline=deadline)
        # Coefficients/counts are probability artifacts, not JSON display data.
        arrays={k:info.pop(k) for k in ('polynomial','counts') if k in info}
        return dict(complete=True,events=[asdict(e) for e in events],decision=info),events,arrays
    except (ValueError,RuntimeError,TimeoutError) as exc:
        return dict(complete=False,events=[],failure=f'{type(exc).__name__}: {exc}'),[],{}


def record(model,plan,row,panel,index,view,band,masked,decision,sample_id,deadline,*,swap=None):
    e=explicit_row(row(index,view,0,band));inputs,target=probability_batch([e],'mps',mask_reference=masked)
    if swap is not None:
        donor=explicit_row(row(swap,'draft'));other,_=probability_batch([donor],'mps')
        # Every acoustic content route swaps together. Request clocks, validity,
        # controls and references stay fixed; this is a feature-layer control.
        inputs={**inputs,**{k:other[k] for k in ('local','song','mel')}}
    pred,events,arrays=prediction(model,inputs,e['request'],decision,plan['seed']+17*index+sample_id,deadline)
    scores=score(events,e['targets'])
    scores['extra_onset_tolerances']={str(ms):match_f1([v.onset_us for v in events],
        [v.onset_us for v in e['targets']],ms*1000)[3] for ms in (1,5,25)}
    start=e['request'].start_us;width=e['request'].end_us-start
    shifts=[match_f1([start+(v.onset_us-start+d)%width for v in events],
        [v.onset_us for v in e['targets']],10000)[3] for d in plan['shifts_us']]
    with torch.no_grad():
        loss,parts=canonical_package_loss(model,inputs,target,e['request'])
        output=model(inputs)
        arrays['count_log_probs']=output['count_log_probs'].cpu().numpy()
    return dict(index=index,component=panel.records[index]['component'],view=view,request=asdict(e['request']),
        prediction=pred,target=[asdict(v) for v in e['targets']],scores=scores,shifted_onset10=shifts,
        probability_nll=float(loss),probability_parts={k:float(v) for k,v in parts.items()},
        audio_donor_index=swap),arrays


def evaluate(model,plan,row,panel,deadline):
    dev=plan['indices']['dev'];selected=dev[::2]
    tasks={
        'draft':[(i,'draft',None,False,'cardinality',0,None) for i in dev],
        'empty':[(i,'draft',None,False,'cardinality',0,None) for i in plan['indices']['empty_dev']],
        'completion':[(i,'middle',None,False,'cardinality',0,None) for i in selected],
        'reference_masked':[(i,'middle',None,True,'cardinality',0,None) for i in selected],
        'density_low':[(i,'draft',(2.,3.),False,'cardinality',0,None) for i in selected[::4]],
        'density_high':[(i,'draft',(6.,8.),False,'cardinality',0,None) for i in selected[::4]],
        'audio_swap':[(i,'draft',None,False,'cardinality',0,int(plan['audio_donors'][str(i)])) for i in dev],
        'individual_mode':[(i,'draft',None,False,'individual_mode',0,None) for i in dev],
    }
    for sample_id in range(4):tasks[f'ancestral_{sample_id}']=[(i,'draft',None,False,'ancestral',sample_id,None) for i in dev]
    summaries={}
    for task,specs in tasks.items():
        rows=[]
        for i,view,band,masked,decision,sample_id,swap in specs:
            if time.monotonic()>deadline:raise TimeoutError('Frozen evaluation wall')
            result,arrays=record(model,plan,row,panel,i,view,band,masked,decision,sample_id,deadline,swap=swap)
            rows.append(result)
            path=ART/'probabilities'/f'{task}-{i}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,**arrays)
            write(ART/f'{task}-final.json',rows)
        summaries[task]=metrics(rows)
        summaries[task]['joint_nll']=float(np.mean([v['probability_nll'] for v in rows]))
    return summaries


def journal(text):
    with (ROOT/'EXPERIMENT_LOG.md').open('a') as f:f.write(f'\n\n{IDENT} {text}\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-health',action='store_true')
    resume=parser.parse_args().resume_health
    prior_report=None
    if resume:
        prior_report=json.loads(REPORT.read_text())
        if prior_report['status']!='health_running' or prior_report['health_updates']!=40:
            raise ValueError('Recovery only for the documented external interruption at health40')
    elif PLAN.exists() or REPORT.exists():raise FileExistsError('Preserve the complete attempt; no implicit rerun')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('No silent backend fallback')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5)
    parent=json.loads((ROOT/'bench/rhythm-v2/local-portfolio-scale-v1.json').read_text())
    panel,store=setup(json.loads((ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()))
    row=row_factory(panel,store);histogram,prior=prior_for(panel,parent['indices']['train64'])
    dev=parent['indices']['dev'];donors={}
    for index in dev:
        donors[str(index)]=next(j for j in dev if panel.records[j]['component']!=panel.records[index]['component'])
    sources=('scripts/run_canonical_package_experiment.py','beatfcn/model/canonical_event_package.py',
        'beatfcn/model/cardinality_dp.py','beatfcn/train/canonical_probability.py',
        'beatfcn/model/local_event_package.py','beatfcn/model/local_typed_canvas.py',
        'beatfcn/model/canvas_diffusion.py','beatfcn/train/local_bundle.py',
        'scripts/run_local_group_diffusion_health.py','bench/rhythm-v2/local-portfolio-scale-v1.json')+EXTRA_SOURCES
    plan=dict(id=IDENT,seed=20261053,model=parent['models']['capacity64'],density_reference=parent['density_reference'],
        indices=parent['indices'],sample_stream=parent['sample_stream']['train64'][:1000],
        health_indices=parent['health_indices'],empty_health_index=parent['empty_health_index'],
        count_histogram=histogram.tolist(),count_prior=prior.tolist(),
        health_steps=100,health_effective_batch=5,quality_steps=1000,effective_batch=2,
        learning_rate=.0003,minimum_learning_rate=.00003,warmup=50,weight_decay=.01,clip=1.,
        health_wall_s=900,quality_wall_s=6000,evaluation_wall_s=1200,max_parameters=12_000_000,
        shifts_us=parent['shifts_us'],audio_donors=donors,
        hypothesis='Complete canonical joint likelihood, full17count tilt and model-cardinality decision repair probability/action mismatches. Entire recipe, not a component causal claim.',
        health_gate=dict(loss_ratio=.9,positive_count_ratio=[.5,1.75],empty_expected=5,empty_generated=2,complete=True,parity=1e-4),
        local_screen=dict(min_nonempty=24,count_ratio=[.5,1.75],onset10=.20,onset10_minus_shift=.05,positive_components=10,full_note_f1=.20),
        endpoint='Fresh fixed1000, never best checkpoint. FourT1samples secondary and equal; individualmode diagnostic; cardinality primary unchanged after scores.',
        preserved='Six real request views, train64 and native multisets, cross-window holds, exact explicitCOPY heads/tails; generated history only within each local group. No online replay, quota/NMS/min-gap/threshold search.',
        confirmation_scored=False,cloud_compute_calls=0,source_sha256={k:sha(ROOT/k) for k in sources})
    if MODEL_OVERRIDE is not None:
        plan.update(model=MODEL_OVERRIDE,seed=20261059,
            hypothesis='Independent complete SSD encoder plus canonical hurdle/orderedGRU/joint nativeCOPY tail/model-cardinality decision. Same data stream; complete recipe, not a backbone-only causal comparison.',
            health_gate='JointNLL decreases10%, onset/tail means no worse5%, exact parity and legal completed decoding. No count or productF1 gate.',
            source_model='beatfcn/model/ssm_event_package.py')
    # Audit all actual training requests before training, not merely draft forms.
    audited=0;maximum=0
    for specs in plan['sample_stream']:
        for spec in specs:
            e=row(spec['index'],spec['view'],spec['field_index'])
            encoded=encode_canonical(e['targets'],e['request']);maximum=max(maximum,int(encoded['count'].max()));audited+=1
    plan.update(target_views_audited=audited,max_cell_type_count=maximum)
    if resume:
        plan=json.loads(PLAN.read_text())
        for name,expected in plan['source_sha256'].items():
            if name!='scripts/run_canonical_package_experiment.py' and sha(ROOT/name)!=expected:
                raise ValueError('Scientific source changed before exact checkpoint recovery: '+name)
    else:
        write(PLAN,plan)
        for name in sources:
            dest=ART/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        shutil.copy2(PLAN,ART/'source/frozen-plan.json')
    model=fresh(plan);torch.mps.manual_seed(plan['seed'])
    n=sum(p.numel() for p in model.parameters())
    if n>plan['max_parameters']:raise ValueError('Parameter cap')
    report=dict(id=IDENT,status='health_running',health_updates=0,quality_updates=0,parameters=n,
        plan_sha256=sha(PLAN),initial_weight_sha256=weight_digest(model),health_weights_used=False,
        curves=[],failures=[],cloud_compute_calls=0,confirmation_scored=False)
    if resume:
        write(ART/'health-external-interruption-at40.json',prior_report)
        report=prior_report
        report['recovery']=dict(reason='External runtime/provider interruption; process absent. Restore exact health25 model/optimizer/CPU+MPS RNG; replay26-40 charged15 extra updates, same gates and final100.',
            recovery_driver_sha256=sha(Path(__file__)),restored_step=25,repeated_updates=15)
    write(REPORT,report);journal(f'完整合同执行，{n}参数；native/COPY统一尾端、共同似然、规范排序、完整基数DP；100健康后fresh1000。恢复={resume}，确认密封，云0。')
    def stopped(*_):raise KeyboardInterrupt('External interruption')
    signal.signal(signal.SIGINT,stopped);signal.signal(signal.SIGTERM,stopped)
    opt=optimizer_for(model,plan);stage='health';started=time.monotonic()
    try:
        specs=list(zip(plan['health_indices'],('draft','single_condition','random_groups','middle')))+[(plan['empty_health_index'],'draft')]
        batches=[]
        for index,view in specs:
            e=explicit_row(row(index,view));inputs,target=probability_batch([e],'mps');batches.append((e,inputs,target))
        report['input_audit']=[dict(index=e['index'],shapes={k:list(v.shape) for k,v in inp.items()},
            statistics={k:dict(finite=bool(torch.isfinite(v).all()),minimum=float(v.min()),maximum=float(v.max()),mean=float(v.float().mean()),std=float(v.float().std())) for k,v in inp.items()},
            visible_events=len(e.get('visible',())),known_carry=len(e['request'].known_carry)) for e,inp,_ in batches]
        report['initial_diagnostic']=diagnostics(model,batches)
        cpu=fresh(plan,'cpu').eval();e,inp,tgt=batches[0]
        ci,ct=({k:v.cpu() for k,v in d.items()} for d in (inp,tgt))
        with torch.no_grad():a,b=cpu(ci),model(inp)
        forward=max(float((a[k]-b[k].cpu()).abs().max()) for k in a)
        cl,_=canonical_package_loss(cpu,ci,ct,e['request']);ml,_=canonical_package_loss(model,inp,tgt,e['request'])
        cl.backward();ml.backward();cp=dict(cpu.named_parameters());gradient=0.
        for name,p in model.named_parameters():
            if (p.grad is None)!=(cp[name].grad is None):raise ValueError('Gradient presence parity')
            if p.grad is not None:gradient=max(gradient,float((p.grad.cpu()-cp[name].grad).abs().max()))
        report['parity']=dict(forward=forward,gradient=gradient)
        if max(forward,gradient)>1e-4:raise ValueError('CPU/MPS parity')
        for e,inp,tgt in batches:
            if not torch.equal(package_support({k:v.cpu() for k,v in inp.items()},e['request']),package_support(inp,e['request']).cpu()):raise ValueError('CPU/MPS support parity')
        del cpu,a,b,cl,ml,cp
        restore_step=0
        if resume:
            saved=torch.load(ART/'health-0025.pt',map_location='cpu',weights_only=False)
            if saved['plan_sha256']!=sha(PLAN) or saved['updates']!=25 or saved['stage']!='health':raise ValueError('Recovery checkpoint identity')
            model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
            # Optimizer moments must follow parameters on the same device.
            for state in opt.state.values():
                for key,value in state.items():
                    if isinstance(value,torch.Tensor) and key!='step':state[key]=value.to('mps')
            torch.set_rng_state(saved['cpu_rng']);torch.mps.set_rng_state(saved['mps_rng'])
            restore_step=25
        for step in range(restore_step+1,101):
            if time.monotonic()-started>plan['health_wall_s']:raise TimeoutError('Health wall')
            model.train();opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']=rate(plan,step)
            total=0.
            for e,inp,tgt in batches:
                loss,_=canonical_package_loss(model,inp,tgt,e['request']);(loss/5).backward();total+=float(loss.detach())/5
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step()
            report['health_updates']=step
            if step%25==0:checkpoint(model,opt,plan,'health',step)
            if step==1 or step%10==0:
                point=dict(stage='health',step=step,loss=total,gradient=grad,wall_s=time.monotonic()-started)
                report['curves'].append(point);write(REPORT,report);print(json.dumps(point),flush=True)
        report['final_diagnostic']=diagnostics(model,batches);preds=[];ratios=[]
        for e,inp,tgt in batches:
            pred,events,_=prediction(model,inp,e['request'],'cardinality',plan['seed']+e['index'],started+plan['health_wall_s'])
            preds.append(dict(index=e['index'],prediction=pred,target=[asdict(v) for v in e['targets']],scores=score(events,e['targets'])))
            if e['targets']:ratios.append(len(events)/len(e['targets']))
        write(ART/'health-predictions.json',preds)
        initial=np.mean([r['loss'] for r in report['initial_diagnostic']]);final=np.mean([r['loss'] for r in report['final_diagnostic']])
        empty=next(v for v in report['final_diagnostic'] if v['index']==plan['empty_health_index'])
        empty_pred=next(v for v in preds if v['index']==plan['empty_health_index'])
        gates=dict(joint_loss=final<=initial*.9,empty_expected=empty['expected']<=5,
            empty_generated=len(empty_pred['prediction']['events'])<=2,positive_ratios=all(.5<=v<=1.75 for v in ratios),
            complete=all(v['prediction']['complete'] for v in preds))
        gates={k:bool(v) for k,v in gates.items()}
        if not COUNT_HEALTH_GATES:
            gates={k:gates[k] for k in ('joint_loss','complete')}
            for part in ('onset','tail'):
                gates[part]=bool(np.mean([v['parts'][part] for v in report['final_diagnostic']])<=1.05*np.mean([v['parts'][part] for v in report['initial_diagnostic']]))
        report.update(health_passed=all(gates.values()),health_gates=gates,health_wall_s=time.monotonic()-started)
        if not report['health_passed']:
            report['status']='health_failed';write(REPORT,report);journal(f'健康失败即结案：{gates}；不派发质量、不加步、不改门。');return
        checkpoint(model,opt,plan,'health',100)
        # All healthy weights are discarded. Exact initial identity is checked.
        del model,opt,batches;torch.mps.empty_cache()
        model=fresh(plan);torch.mps.manual_seed(plan['seed'])
        if weight_digest(model)!=report['initial_weight_sha256']:raise ValueError('Fresh initialization identity')
        opt=optimizer_for(model,plan);stage='quality';started=time.monotonic();report['status']='quality_running';write(REPORT,report)
        journal('真实健康通过，丢弃健康权重；同seed新初始化开始固定1000质量更新。')
        for step in range(1,1001):
            if time.monotonic()-started>plan['quality_wall_s']:raise TimeoutError('Quality total wall')
            model.train();opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']=rate(plan,step)
            total=0.
            for spec in plan['sample_stream'][step-1]:
                e=explicit_row(row(spec['index'],spec['view'],spec['field_index']));inp,tgt=probability_batch([e],'mps')
                loss,_=canonical_package_loss(model,inp,tgt,e['request']);(loss/2).backward();total+=float(loss.detach())/2
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();report['quality_updates']=step
            if step%250==0:checkpoint(model,opt,plan,'quality',step)
            if step==1 or step%25==0:
                point=dict(stage='quality',step=step,loss=total,gradient=grad,wall_s=time.monotonic()-started,
                    mps_tensor_bytes=torch.mps.current_allocated_memory(),mps_driver_bytes=torch.mps.driver_allocated_memory())
                report['curves'].append(point);write(REPORT,report);print(json.dumps(point),flush=True)
        path=checkpoint(model,opt,plan,'quality',1000);report['final_weight_sha256']=sha(path)
        report['quality_training_wall_s']=time.monotonic()-started;report['status']='evaluating';write(REPORT,report)
        moment=time.monotonic();result=evaluate(model,plan,row,panel,min(started+plan['quality_wall_s'],moment+plan['evaluation_wall_s']))
        report.update(evaluations=result,evaluation_wall_s=time.monotonic()-moment)
        d=result['draft'];g=plan['local_screen']
        gates=dict(complete=d['complete']==d['attempted'],nonempty=d['nonempty']>=g['min_nonempty'],
            count_ratio=g['count_ratio'][0]<=d['density_ratio']<=g['count_ratio'][1],onset10=d['onset10']>=g['onset10'],
            onset_null=d['onset10']-d['shift_null10']>=g['onset10_minus_shift'],
            components=sum(v['onset10']>v['shift_null10'] for v in d['components'].values())>=g['positive_components'],
            full_f1=d['full_note_f1']>=g['full_note_f1'])
        report.update(status='completed',local_screen=gates,local_screen_passed=all(gates.values()),total_quality_wall_s=time.monotonic()-started)
        write(REPORT,report);journal(f'固定1000及全评价完成；本地筛选{gates}。不改读出、不延长、不上云。')
    except BaseException as exc:
        report.update(status='stopped',stopped_stage=stage,stage_wall_s=time.monotonic()-started)
        report['failures'].append(f'{type(exc).__name__}: {exc}')
        checkpoint(model,opt,plan,f'{stage}-interrupted',report[f'{stage}_updates'])
        write(REPORT,report);journal(f'停止并保存现场：{report["failures"][-1]}。');raise


if __name__=='__main__':main()
