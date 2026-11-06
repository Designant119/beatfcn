"""Bounded serial local successors under the owner's open research mandate."""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import gc
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
from beatfcn.model.grouped_chart_process import grouped_process_loss, generate_grouped_process
from beatfcn.model.local_evolution_controls import AudioInteractionGroupedProcess, reset_mamba_boundaries
from beatfcn.model.local_mamba_canvas import MambaCanvas
from beatfcn.model.local_typed_canvas import canvas_loss
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.train.canonical_probability import explicit_row, probability_batch
from beatfcn.train.relative_canvas import relative_batch, generate_relative
from scripts import run_grouped_chart_experiment as grouped
from scripts import run_local_portfolio_data_scale as canvas_eval
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_scale import row_factory, weight_digest, sha, write
from scripts.run_local_typed_canvas import score
from scripts.validate_mechanism_reviews import validate_successor


ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'reports/local-evolution-v1'
PLAN=ROOT/'bench/rhythm-v2/local-evolution-v1.json'
ART=ROOT/'artifacts/local-evolution-v1'
ARMS=('ar_interaction','ssd_full','ssd_reset')
SOURCES=('scripts/run_local_evolution.py','beatfcn/model/local_evolution_controls.py',
    'beatfcn/model/grouped_chart_process.py','beatfcn/model/local_mamba_canvas.py',
    'beatfcn/model/mamba2_reference.py','beatfcn/model/fullstack_mamba.py',
    'beatfcn/model/canonical_event_package.py','beatfcn/model/local_event_package.py',
    'beatfcn/model/local_typed_canvas.py','beatfcn/train/canonical_probability.py',
    'beatfcn/train/relative_canvas.py','scripts/run_grouped_chart_experiment.py',
    'scripts/run_local_portfolio_data_scale.py','tests/test_local_evolution_controls.py')


def read(path):return json.loads((ROOT/path).read_text())


def event(kind,**values):
    row=dict(kind=kind,recorded_at=datetime.now(timezone.utc).isoformat(),**values)
    with (FOLDER/'events.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)


def update_state(**values):
    state=read('reports/local-evolution-v1/state.json');state.update(values)
    write(FOLDER/'state.json',state)


def prepare():
    if PLAN.exists():raise FileExistsError('Preserve the frozen evolution plan')
    ar=read('bench/rhythm-v2/local-grouped-chart-v2.json')
    ssd=read('bench/rhythm-v2/local-portfolio-mamba-quality-v1.json')
    plan=dict(id='local-evolution-v1',created_at=datetime.now(timezone.utc).isoformat(),
        owner_authority='design/LOCAL_EVOLUTION.md',serial=True,cloud_compute_calls=0,
        historical_results_unchanged=True,arms=list(ARMS),execution_ready=False,
        predecessors=['local-grouped-chart-v2','ssd-reference1000'],
        shared_failure_modes=['within_cell_audio_alias','ssd_attribution_not_isolated'],
        mechanism_changes={
          'within_cell_audio_alias':dict(change='keys=audio+phase+audio*phase, no added parameters; every other AR equation unchanged',
            prediction='Holding global query fixed, local frame audio changes relative odds of distinct millisecond candidates',
            discriminating_control='Exact old/fixed state identity; synthetic and actual frame log-odds autograd; actual CPU/MPS loss/gradient parity'),
          'ssd_attribution_not_isolated':dict(change='Only erase SSD state across64-token chunk boundaries; retain conv/gates/norms/FFN/routing/parameters',
            prediction='If cross-chunk state is useful, full beats reset on held-out localization with matched initial weights/updates',
            discriminating_control='Short-sequence value/gradient identity; long-sequence boundary intervention; two fresh paired arms, same input stream and schedule')},
        pretraining_probe_receipts=[],
        ar={**ar,'id':'evolution-ar-audio-interaction-v1','quality_wall_s':4200,'evaluation_wall_s':1200,
            'baseline_report':'reports/local_grouped_chart_v2.json',
            'comparison':'Same seed/parameters/stream/objective/T1 sampler as historical AR; exact historical scientific sources and initial tensors checked; no new baseline training.',
            'benefit_gate':dict(onset10_gain=.02,full_f1_max_drop=.01,positive_component_gains=10,count_ratio=[.5,1.75]),
            'mechanism_scope':'A parameter-free bilinear interaction repair, not a claim new fine audio was introduced or all AR exposure failures are solved.'},
        ssd={**ssd,'id':'evolution-ssd-boundary-causal-v1','steps':1000,'schedule_steps':2000,
            'health_steps':100,'health_wall_s':900,'quality_wall_s':3600,'evaluation_wall_s':900,
            'hypothesis':'Cross-chunk recurrent carry helps this SSD canvas, independent of parameter/initialization/data/readout differences.',
            'benefit_gate':dict(onset10_full_minus_reset=.02,full_f1_max_drop=.01,positive_components=10,count_ratio=[.5,1.75]),
            'scope':'Tests carry at64-token boundaries across all7 bidirectional cores. Convolution, within-chunk SSD and multilayer receptive fields remain; not SSD vs attention or global music understanding.'},
        health_gate=dict(loss_ratio=.9,field_max_ratio=1.05,parity_max_error=1e-4,
            finite=True,all_five_free_decodes_complete=True,note='Engineering learning gate only; no product F1/type/empty quota gate.'),
        sample_policy='AR four fixed T1 equal weights plus secondary greedy; SSD deterministic expected_tail_rank; no best-checkpoint/sample/threshold search.',
        quality_endpoint='Exactly1000freshupdates after discarded100health; trace250/500 are diagnostic only.',
        stop_policy='No automatic retries or resume. Numerical/resource failure stops and retains checkpoint. Health/quality failure remains pending mechanism review, never gate-only closed.',
        sources={name:sha(ROOT/name) for name in SOURCES},
        baseline_identity={p:sha(ROOT/p) for p in ['bench/rhythm-v2/local-grouped-chart-v2.json','reports/local_grouped_chart_v2.json',
            'bench/rhythm-v2/local-portfolio-mamba-quality-v1.json','reports/local_portfolio_terminal_catalog_v1.json']})
    write(PLAN,plan)
    for name in SOURCES:
        dst=ART/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
    shutil.copy2(PLAN,ART/'source/preflight-plan.json')
    event('contracts_frozen',plan=str(PLAN.relative_to(ROOT)),training_updates=0)


def fresh(plan,arm,device='cpu'):
    p=plan['ar'] if arm=='ar_interaction' else plan['ssd']
    torch.manual_seed(p['seed'])
    if arm=='ar_interaction':
        model=AudioInteractionGroupedProcess(FullEventConfig(**p['model']),density_reference=p['density_reference'],history_dropout=.1)
        prior=torch.tensor(p['count_prior'],dtype=torch.float32)
        with torch.no_grad():
            model.count_gate.weight.zero_();model.count_positive.weight.zero_()
            model.count_gate.bias.copy_((1-prior[:,0]).log()-prior[:,0].log())
            model.count_positive.bias.copy_((prior[:,1:]/(1-prior[:,:1])).log().flatten())
            model.stop_read.bias.fill_(p['stop_bias'])
    else:
        model=MambaCanvas(FullEventConfig(**p['model']),density_reference=p['density_reference'])
        if arm=='ssd_reset':
            if reset_mamba_boundaries(model)!=7:raise ValueError('Unexpected SSD core count')
    return model.to(device)


def example_batch(row,index,view,arm,device,field=0):
    example=row(index,view,field)
    if arm=='ar_interaction':
        example=explicit_row(example);inputs,target=probability_batch([example],device)
    else:inputs,target=relative_batch([example],device)
    return example,inputs,target


def loss_for(model,batch,arm):
    e,inputs,target=batch
    return (grouped_process_loss(model,inputs,e['targets'],e['request']) if arm=='ar_interaction'
            else canvas_loss(model(inputs),target,inputs))


def diagnostic(model,batches,arm):
    model.eval();rows=[]
    with torch.no_grad():
        for batch in batches:
            loss,parts=loss_for(model,batch,arm)
            rows.append(dict(index=batch[0]['index'],loss=float(loss),parts={k:float(v) for k,v in parts.items()}))
    return rows


def load_data():
    panel,store=setup(read('bench/rhythm-v2/local-frontend-comparison-v2.json'))
    return panel,row_factory(panel,store)


def preflight():
    plan=read(str(PLAN.relative_to(ROOT)));panel,row=load_data();results={}
    for name,expected in plan['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Source drift: '+name)
    for old_plan in ('bench/rhythm-v2/local-grouped-chart-v2.json','bench/rhythm-v2/local-portfolio-mamba-quality-v1.json'):
        for name,expected in read(old_plan)['source_sha256'].items():
            if sha(ROOT/name)!=expected:raise ValueError('Historical scientific boundary drift: '+name)
    for arm in ARMS:
        p=plan['ar'] if arm=='ar_interaction' else plan['ssd']
        cpu=fresh(plan,arm).eval();gpu=fresh(plan,arm,'mps').eval()
        digest=weight_digest(cpu)
        if arm=='ar_interaction':
            if digest!=read(p['baseline_report'])['initial_weight_sha256']:raise ValueError('AR initialization changed')
        elif results.get('ssd_full',{}).get('initial_weight_sha256',digest)!=digest:
            raise ValueError('SSD paired initialization differs')
        e,ci,ct=example_batch(row,p['health_indices'][0],'draft',arm,'cpu')
        gb=(e,{k:v.to('mps') for k,v in ci.items()},{k:v.to('mps') for k,v in ct.items()})
        cl,_=loss_for(cpu,(e,ci,ct),arm);ml,_=loss_for(gpu,gb,arm)
        loss_error=abs(float(cl.detach())-float(ml.detach()));cl.backward();ml.backward();torch.mps.synchronize()
        gp=dict(gpu.named_parameters());gradient=0.
        for name,param in cpu.named_parameters():
            g=gp[name].grad
            if (g is None)!=(param.grad is None):raise ValueError('Gradient presence differs')
            if g is not None:gradient=max(gradient,float((param.grad-g.cpu()).abs().max()))
        if max(loss_error,gradient)>1e-4:raise ValueError(f'{arm} numerical parity {loss_error}/{gradient}')
        if not all(bool(torch.isfinite(v).all()) for v in ci.values()):raise ValueError('Nonfinite real inputs')
        mechanism={}
        if arm=='ar_interaction':
            context=cpu.context(ci,e['request']);choices=(context['frame']==100).nonzero().flatten()[[0,-1]]
            query=cpu.chart_query(torch.cat((cpu.chart_start,context['pooled']),-1)).detach()
            odds=((context['keys'][choices[0]]-context['keys'][choices[1]])*query).sum()
            grad=torch.autograd.grad(odds,context['x'])[0][100]
            mechanism['within_cell_audio_odds_gradient_max']=float(grad.abs().max())
            if mechanism['within_cell_audio_odds_gradient_max']<1e-6:raise ValueError('Audio interaction still aliases')
        shapes={k:list(v.shape) for k,v in ci.items()}
        stats={k:dict(min=float(ci[k].min()),max=float(ci[k].max()),mean=float(ci[k].float().mean()),std=float(ci[k].float().std()))
               for k in ('local','mel','song','reference','controls')}
        results[arm]=dict(status='completed',initial_weight_sha256=digest,parameters=sum(p.numel() for p in cpu.parameters()),
            index=e['index'],target_events=len(e['targets']),sample_targets=[asdict(v) for v in e['targets'][:5]],
            input_shapes=shapes,input_stats=stats,loss_cpu=float(cl.detach()),loss_mps=float(ml.detach()),loss_error=loss_error,gradient_error=gradient,
            mechanism=mechanism,mps_allocated_bytes=torch.mps.current_allocated_memory(),mps_driver_bytes=torch.mps.driver_allocated_memory(),
            mps_recommended_bytes=torch.mps.recommended_max_memory())
        event('preflight_arm_completed',arm=arm,loss_error=loss_error,gradient_error=gradient)
        del cpu,gpu,cl,ml,gp,gb;gc.collect();torch.mps.empty_cache()
    # Separate frozen-weight deletion evidence from fresh paired training.
    old=next(v for v in read('reports/local_portfolio_terminal_catalog_v1.json')['candidates'] if v['id']=='ssd-reference1000')
    if sha(ROOT/old['checkpoint'])!=old['checkpoint_sha256']:raise ValueError('Old SSD weight drift')
    saved=torch.load(ROOT/old['checkpoint'],map_location='cpu',weights_only=False)
    frozen={}
    canvas_eval.LIVE['panel']=panel;canvas_eval.ART=ART/'frozen-ssd-carry'
    for arm in ('ssd_full','ssd_reset'):
        model=fresh(plan,arm,'mps');model.load_state_dict(saved['model']);model.eval()
        frozen[arm]=canvas_eval.evaluate({arm:model},plan['ssd'],row,'frozen',time.monotonic()+300,full=False)[arm]
        del model;gc.collect();torch.mps.empty_cache()
    write(FOLDER/'ssd-frozen-carry.json',dict(status='completed',optimizer_updates=0,cloud_compute_calls=0,
        checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],evaluations=frozen,
        scope='Same learned weights; erase cross-chunk carry only. Distribution intervention, not a retrained architecture limit.'))
    receipt=FOLDER/'preflight.json';write(receipt,dict(status='completed',scope='new_implementation',arms=results,optimizer_updates=0,cloud_compute_calls=0,source_sha256=plan['sources']))
    plan.update(execution_ready=True,pretraining_probe_receipts=[dict(status='completed',scope='new_implementation',evidence=str(receipt.relative_to(ROOT)))])
    write(PLAN,plan);shutil.copy2(PLAN,ART/'source/frozen-plan.json')
    event('preflight_passed',receipt=str(receipt.relative_to(ROOT)))


def rate(p,step,arm):
    if arm=='ar_interaction':return grouped.rate(p,step)
    if step<=p['warmup']:return p['learning_rate']*step/p['warmup']
    q=(step-p['warmup'])/(p['schedule_steps']-p['warmup'])
    return p['minimum_learning_rate']+(p['learning_rate']-p['minimum_learning_rate'])*(1+math.cos(math.pi*q))/2


def checkpoint(model,opt,arm,stage,step):
    path=ART/arm/f'{stage}-{step:04d}.pt';path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),stage=stage,updates=step,
        plan_sha256=sha(PLAN),cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state()),path)
    return path


def train_arm(plan,arm,panel,row):
    path=FOLDER/f'{arm}.json'
    if path.exists():raise FileExistsError('No silent restart: '+str(path))
    p=plan['ar'] if arm=='ar_interaction' else plan['ssd']
    report=dict(id=p['id'],arm=arm,status='health_running',health_updates=0,quality_updates=0,
        plan_sha256=sha(PLAN),failures=[],curves=[],health_weights_used=False,cloud_compute_calls=0)
    write(path,report);update_state(status='running',active_arm=arm,active_report=str(path.relative_to(ROOT)),pid=os.getpid())
    model=fresh(plan,arm,'mps');torch.mps.manual_seed(p['seed'])
    if arm=='ar_interaction':model.history_dropout=0.
    opt=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    report['initial_weight_sha256']=weight_digest(model)
    specifications=list(zip(p.get('health_indices',plan['ar']['health_indices']),('draft','single_condition','random_groups','middle')))
    specifications.append((p.get('empty_health_index',plan['ar']['empty_health_index']),'draft'))
    batches=[example_batch(row,i,v,arm,'mps') for i,v in specifications]
    report['initial_health']=diagnostic(model,batches,arm);write(path,report)
    stage='health';started=time.monotonic()
    try:
        for stage,steps in [('health',100),('quality',1000)]:
            if stage=='quality':
                del model,opt,batches;gc.collect();torch.mps.empty_cache()
                model=fresh(plan,arm,'mps');torch.mps.manual_seed(p['seed'])
                if weight_digest(model)!=report['initial_weight_sha256']:raise ValueError('Fresh quality initialization mismatch')
                opt=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
                report['status']='quality_running';write(path,report)
            started=time.monotonic();event('training_stage_started',arm=arm,stage=stage)
            for step in range(1,steps+1):
                if time.monotonic()-started>p[stage+'_wall_s']:raise TimeoutError(stage+' wall')
                model.train();opt.zero_grad(set_to_none=True)
                for group in opt.param_groups:group['lr']=rate(p,step,arm)
                if stage=='health':items=batches
                else:
                    stream=p['sample_stream'][step-1]
                    items=[example_batch(row,s['index'],s['view'],arm,'mps',s['field_index']) for s in stream]
                loss_value=0.
                for batch in items:
                    loss,_=loss_for(model,batch,arm);(loss/len(items)).backward();loss_value+=float(loss.detach())/len(items)
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),p['clip'],error_if_nonfinite=True));opt.step();torch.mps.synchronize()
                report[stage+'_updates']=step
                if step==1 or step%10==0:
                    point=dict(stage=stage,step=step,loss=loss_value,preclip_gradient_norm=norm,wall_s=time.monotonic()-started,
                        mps_allocated_bytes=torch.mps.current_allocated_memory(),mps_driver_bytes=torch.mps.driver_allocated_memory())
                    report['curves'].append(point);write(path,report);print(json.dumps(dict(arm=arm,**point)),flush=True)
                if step in (100,250,500,1000):checkpoint(model,opt,arm,stage,step)
            report[stage+'_wall_s']=time.monotonic()-started
            if stage=='health':
                report['final_health']=diagnostic(model,batches,arm)
                before,after=report['initial_health'],report['final_health'];gates={}
                gates['loss']=np.mean([v['loss'] for v in after])<=.9*np.mean([v['loss'] for v in before])
                for name in (('group_head_stop','counts','tail') if arm=='ar_interaction' else ('count','offset')):
                    gates[name]=np.mean([v['parts'][name] for v in after])<=1.05*np.mean([v['parts'][name] for v in before])
                predictions=[];model.eval()
                for e,inp,target in batches:
                    if arm=='ar_interaction':
                        events,info=generate_grouped_process(model,inp,e['request'],seed=p['seed']+e['index'],deadline=started+p['health_wall_s'])
                        pred=dict(complete=True,events=[asdict(v) for v in events],diagnostics=info)
                    else:pred=generate_relative(model,inp,e['request'],deadline=started+p['health_wall_s'])
                    events=[RhythmEvent(**v) for v in pred['events']]
                    predictions.append(dict(index=e['index'],prediction=pred,target=[asdict(v) for v in e['targets']],scores=score(events,e['targets'])))
                gates['complete']=all(v['prediction']['complete'] for v in predictions)
                report['health_gates']={k:bool(v) for k,v in gates.items()};report['health_passed']=all(gates.values())
                write(ART/arm/'health-predictions.json',predictions);write(path,report)
                if not report['health_passed']:
                    report['status']='health_failed_pending_mechanism_review';write(path,report)
                    event('health_failed_not_closed',arm=arm,gates=report['health_gates']);return
        final=checkpoint(model,opt,arm,'quality',1000);report['final_weight_sha256']=sha(final)
        report['status']='evaluating';write(path,report);moment=time.monotonic()
        if arm=='ar_interaction':
            grouped.ART=ART/arm
            report['evaluations']=grouped.evaluate(model,p,row,panel,moment+p['evaluation_wall_s'])
        else:
            canvas_eval.ART=ART;canvas_eval.LIVE['panel']=panel
            report['evaluations']=canvas_eval.evaluate({arm:model},p,row,'quality-1000',moment+p['evaluation_wall_s'],full=True)[arm]
        report.update(status='terminal_pending_mechanism_review',evaluation_wall_s=time.monotonic()-moment)
        write(path,report);event('fixed_terminal_ready',arm=arm,report=str(path.relative_to(ROOT)),quality_updates=1000)
    except BaseException as exc:
        report.update(status='stopped_pending_mechanism_review',stopped_stage=stage)
        report['failures'].append(dict(stage=stage,updates=report[stage+'_updates'],error=f'{type(exc).__name__}: {exc}'))
        checkpoint(model,opt,arm,stage+'-interrupted',report[stage+'_updates']);write(path,report)
        event('experiment_stopped_not_closed',arm=arm,error=report['failures'][-1]);raise
    finally:
        del model,opt;gc.collect();torch.mps.empty_cache()


def run():
    plan=read(str(PLAN.relative_to(ROOT)))
    known={r['id'] for r in read('reports/local-portfolio-v1/mechanism-reviews-v2/index.json')['candidates']}
    errors=validate_successor(plan,known)
    if errors:raise ValueError(errors)
    for name,expected in plan['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen source drift: '+name)
    panel,row=load_data()
    for arm in ARMS:train_arm(plan,arm,panel,row)
    update_state(status='terminals_ready_pending_mechanism_review',active_arm=None,pid=None)
    event('serial_queue_finished',next='Deep terminal reviews and council if useful; historical pool closure is not a research stop.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','preflight','run'));args=parser.parse_args()
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('Silent MPS fallback prohibited')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5)
    def stop(*_):raise KeyboardInterrupt('External stop')
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    {'prepare':prepare,'preflight':preflight,'run':run}[args.action]()


if __name__=='__main__':main()
