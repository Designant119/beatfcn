"""Fresh bounded quality curves for independently healthy mel/package candidates."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import signal
import time

import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_event_package import EventPackageCanvas,generate_package,package_loss
from beatfcn.model.local_typed_canvas import canvas_loss
from beatfcn.model.mel_only_canvas import MelOnlyCanvas
from beatfcn.model.sample import match_f1
from beatfcn.train.local_typed_canvas import canvas_batch
from beatfcn.train.relative_canvas import relative_batch,generate_relative
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_mamba_health import sha,write
from scripts.run_local_portfolio_scale import row_factory,weight_digest
from scripts.run_local_typed_canvas import score
from scripts.review_frontend_canvas_comparison import metrics


ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'bench/rhythm-v2/local-portfolio-scale-v1.json'


def select_inputs(inputs,candidate):
    # Loader clocks are common. Pretrained payloads are not even passed to mel's forward.
    if candidate=='mel':
        inputs={k:v for k,v in inputs.items() if not k.startswith(('local','song'))}
    return {k:v.to('mps') for k,v in inputs.items()}


def batch(example,candidate,masked=False):
    function=relative_batch if candidate=='mel' else canvas_batch
    inputs,target=function([example],mask_reference=masked)
    return select_inputs(inputs,candidate),{k:v.to('mps') for k,v in target.items()}


def loss_for(model,inputs,target,request,candidate):
    if candidate=='mel':return canvas_loss(model(inputs),target,inputs)
    return package_loss(model,inputs,target,[request])


@torch.no_grad()
def generate(model,inputs,request,candidate,deadline):
    model.eval()
    if candidate=='mel':return generate_relative(model,inputs,request,deadline=deadline)
    try:
        events=generate_package(model,inputs,request)
        if time.monotonic()>deadline:raise TimeoutError('Package generation deadline')
        return dict(complete=True,failure=None,events=[asdict(e) for e in events])
    except (ValueError,RuntimeError,TimeoutError) as exc:
        return dict(complete=False,failure=f'{type(exc).__name__}: {exc}',events=[])


def evaluate(model,plan,row,panel,art,tag,deadline,full):
    tasks={'draft':[(i,'draft',None,False) for i in plan['indices']['dev']],
           'empty':[(i,'draft',None,False) for i in plan['indices']['empty_dev']]}
    if full:
        selected=plan['indices']['dev'][::2]
        tasks.update(completion=[(i,'middle',None,False) for i in selected],
            reference_masked=[(i,'middle',None,True) for i in selected],
            density_low=[(i,'draft',(2.,3.),False) for i in selected[::4]],
            density_high=[(i,'draft',(6.,8.),False) for i in selected[::4]])
    summary={}
    for task,specs in tasks.items():
        records=[]
        for index,view,bands,masked in specs:
            if time.monotonic()>deadline:raise TimeoutError('Frozen total evaluation wall')
            example=row(index,view,0,bands);inputs,_=batch(example,plan['candidate'],masked)
            prediction=generate(model,inputs,example['request'],plan['candidate'],deadline)
            pred=[RhythmEvent(**e) for e in prediction['events']];scores=score(pred,example['targets'])
            scores['extra_onset_tolerances']={str(ms):match_f1([e.onset_us for e in pred],
                [e.onset_us for e in example['targets']],ms*1000)[3] for ms in (1,5,25)}
            start=example['request'].start_us;width=example['request'].end_us-start
            shifts=[match_f1([start+(e.onset_us-start+d)%width for e in pred],
                [e.onset_us for e in example['targets']],10000)[3] for d in plan['shifts_us']]
            records.append(dict(index=index,component=panel.records[index]['component'],view=view,
                request=asdict(example['request']),prediction=prediction,
                target=[asdict(e) for e in example['targets']],scores=scores,shifted_onset10=shifts))
            write(art/f'{tag}-{task}.json',records)
        summary[task]=metrics(records)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',choices=('mel','package'),required=True)
    candidate=parser.parse_args().candidate
    ident=f'local-portfolio-{candidate}-quality-v1'
    plan_path=ROOT/f'bench/rhythm-v2/{ident}.json'
    report_path=ROOT/f'reports/local_portfolio_{candidate}_quality_v1.json'
    art=ROOT/f'artifacts/local-portfolio-v1/{candidate}-quality'
    health_plan_path=ROOT/f'bench/rhythm-v2/local-portfolio-{candidate}-health-v1.json'
    health_report_path=ROOT/f'reports/local_portfolio_{candidate}_health_v1.json'
    if plan_path.exists() or report_path.exists():raise FileExistsError('Preserve the existing quality run')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('No silent MPS fallback')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5)
    parent=json.loads(PARENT.read_text());health_plan=json.loads(health_plan_path.read_text())
    health=json.loads(health_report_path.read_text())
    if not health['health_passed'] or health['health_updates']!=100 or health['plan_sha256']!=sha(health_plan_path):
        raise ValueError('Require exact independent passed health')
    for source,expected in health_plan['source_sha256'].items():
        if sha(ROOT/source)!=expected:raise ValueError(f'Frozen health source changed: {source}')
    model_class=MelOnlyCanvas if candidate=='mel' else EventPackageCanvas
    model_source=f'beatfcn/model/{"mel_only_canvas" if candidate=="mel" else "local_event_package"}.py'
    plan=dict(id=ident,candidate=candidate,seed=health_plan['seed'],steps=2000,effective_batch=2,
        model=health_plan['model'],density_reference=health_plan['density_reference'],
        indices=parent['indices'],sample_stream=parent['sample_stream']['train64'],
        learning_rate=health_plan['learning_rate'],minimum_learning_rate=health_plan['minimum_learning_rate'],
        warmup=health_plan['warmup'],weight_decay=0.,clip=1.,
        quality_checkpoints=[1000,2000],diagnostic_checkpoints=[250,500,1500],
        training_wall_s=5400 if candidate=='package' else 3600,evaluation_wall_s=1200,
        shifts_us=parent['shifts_us'],health_plan_sha256=sha(health_plan_path),
        health_checkpoint_sha256=sha(ROOT/f'artifacts/local-portfolio-v1/{candidate}-health/health-0100.pt'),
        health_weights_used=False,confirmation_scored=False,cloud_compute_calls=0,
        hypothesis=('Learned acoustic-only10msmel+local0.5s context can acquire cross-song rhythm placement without pretrained representations.'
                    if candidate=='mel' else 'Normalized counts and count/head-conditioned localGRU event packages improve coordinated free generation.'),
        interpretation='Complete-candidate local investment screen. Different route sizes/seeds are reported; not a causal pretrained-frontend or backbone-only comparison. No product-readiness conjunction.',
        endpoint='Fixed2000; report1000 separately, no best-checkpoint selection. Confirm8 remains sealed.',
        deferred_capabilities='No human emotion/phrase judgment, fullsong rollout, listening, editing timing or native playtest is claimed by window F1.',
        source_sha256={name:sha(ROOT/name) for name in (
            'scripts/run_local_portfolio_candidate_quality.py',model_source,
            'beatfcn/model/local_typed_canvas.py','beatfcn/train/relative_canvas.py',
            'beatfcn/train/local_typed_canvas.py','scripts/run_local_portfolio_scale.py',
            'scripts/run_frontend_canvas_comparison.py','scripts/review_frontend_canvas_comparison.py',
            str(PARENT.relative_to(ROOT)),str(health_plan_path.relative_to(ROOT)))})
    write(plan_path,plan)
    comparison=json.loads((ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text())
    panel,store=setup(comparison);row=row_factory(panel,store)
    torch.manual_seed(plan['seed'])
    model=model_class(FullEventConfig(**plan['model']),density_reference=plan['density_reference']).to('mps')
    initial=weight_digest(model)
    if initial!=health['initial_weight_sha256']:raise ValueError('Fresh quality does not match health initial state')
    opt=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=0.)
    report=dict(id=ident,status='prepared',updates=0,parameters=sum(p.numel() for p in model.parameters()),
        plan_sha256=sha(plan_path),initial_weight_sha256=initial,health_weights_used=False,
        curves=[],evaluations={},failures=[],wall_s={},cloud_compute_calls=0,confirmation_scored=False,
        exposure=dict(windows=0,events=0,indices={},types={}))
    write(report_path,report)
    def checkpoint(stage,step):
        path=art/f'{stage}-{step:04d}.pt';path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),config=plan['model'],
            plan_sha256=sha(plan_path),updates=step,stage=stage,cpu_rng=torch.get_rng_state(),
            mps_rng=torch.mps.get_rng_state()),path)
    def journal(message):
        with (ROOT/'EXPERIMENT_LOG.md').open('a') as f:f.write(f'\n\n{ident} {message}\n')
    journal(f'冻结：真实健康通过后fresh初始化，参数{report["parameters"]}；train64同冻结请求流、2000步/1000复盘，lr={plan["learning_rate"]}，不加载健康权重。确认密封，云0。')
    def stop(*_):raise KeyboardInterrupt('External stop')
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    started=time.monotonic();evaluation_s=0.
    try:
        report['status']='training'
        for step in range(1,plan['steps']+1):
            if time.monotonic()-started>plan['training_wall_s']:raise TimeoutError('Frozen training wall includes evaluation')
            if step<=plan['warmup']:lr=plan['learning_rate']*step/plan['warmup']
            else:
                progress=(step-plan['warmup'])/(plan['steps']-plan['warmup'])
                lr=plan['minimum_learning_rate']+(plan['learning_rate']-plan['minimum_learning_rate'])*(1+math.cos(math.pi*progress))/2
            model.train();opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']=lr
            total=0.
            for spec in plan['sample_stream'][step-1]:
                example=row(spec['index'],spec['view'],spec['field_index']);inputs,target=batch(example,candidate)
                loss,detail=loss_for(model,inputs,target,example['request'],candidate)
                (loss/2).backward();total+=float(loss.detach())/2
                exposure=report['exposure'];exposure['windows']+=1;exposure['events']+=len(example['targets'])
                key=str(spec['index']);exposure['indices'][key]=exposure['indices'].get(key,0)+1
                for event in example['targets']:exposure['types'][event.type]=exposure['types'].get(event.type,0)+1
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),plan['clip'],error_if_nonfinite=True))
            opt.step();report['updates']=step
            if step==1 or step%25==0:
                point=dict(step=step,learning_rate=lr,loss=total,preclip_gradient_norm=grad,wall_s=time.monotonic()-started)
                report['curves'].append(point);report['wall_s']['training']=point['wall_s'];write(report_path,report)
                print(json.dumps(dict(stage=f'{candidate}-quality',**point)),flush=True)
            if step in plan['quality_checkpoints']+plan['diagnostic_checkpoints']:checkpoint('quality',step)
            if step in plan['quality_checkpoints']:
                moment=time.monotonic();deadline=min(started+plan['training_wall_s'],moment+plan['evaluation_wall_s']-evaluation_s)
                report['evaluations'][str(step)]=evaluate(model,plan,row,panel,art,f'quality-{step:04d}',deadline,full=step==plan['steps'])
                evaluation_s+=time.monotonic()-moment;write(report_path,report)
        report.update(status='completed',complete=True)
        report['wall_s'].update(training=time.monotonic()-started,evaluation=evaluation_s)
        report['final_weight_sha256']=sha(art/'quality-2000.pt');write(report_path,report)
        journal(f'终态：{report["updates"]}/2000与开发评价完成，确认未评分，云0。')
    except BaseException as exc:
        report['status']='stopped';report['failures'].append(dict(error=f'{type(exc).__name__}: {exc}',updates=report['updates']))
        report['wall_s']['training']=time.monotonic()-started;write(report_path,report)
        journal(f'停止：{report["failures"][-1]}。只保留已写checkpoint，不自动续训、不延长墙钟，云0。');raise


if __name__=='__main__':main()
