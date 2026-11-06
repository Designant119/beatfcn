"""C06 one-contract objective intervention; bounded local run and diagnostics."""
from datetime import datetime,timezone
from dataclasses import asdict
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
from beatfcn.model.local_evolution_controls import reset_mamba_boundaries
from beatfcn.model.local_mamba_canvas import MambaCanvas
from beatfcn.model.unified_count_canvas import UnifiedCountCanvas,unified_count_loss
from beatfcn.model.count_readout import expected_tail_rank
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.train.relative_canvas import relative_batch,generate_relative
from scripts import run_local_portfolio_data_scale as evaluator
from scripts.run_local_evolution import load_data
from scripts.run_local_portfolio_scale import sha,write,weight_digest
from scripts.run_local_typed_canvas import score
from scripts.diagnose_evolution_terminals import FrozenOutput
from scripts.validate_mechanism_reviews import load_registry,validate_successor


ROOT=Path(__file__).resolve().parents[1];FOLDER=ROOT/'reports/local-evolution-v2'
PLAN=ROOT/'bench/rhythm-v2/c06-reset-unified-count-law-v1.json'
REPORT=FOLDER/'unified-count.json';ART=ROOT/'artifacts/c06-reset-unified-count-law-v1'
SOURCES=('scripts/run_evolution_unified_count.py','beatfcn/model/unified_count_canvas.py',
 'beatfcn/model/local_evolution_controls.py','beatfcn/model/local_mamba_canvas.py','beatfcn/model/mamba2_reference.py',
 'beatfcn/model/fullstack_mamba.py','beatfcn/model/local_typed_canvas.py','beatfcn/train/relative_canvas.py',
 'scripts/run_local_portfolio_data_scale.py','scripts/run_local_portfolio_scale.py','scripts/run_frontend_canvas_comparison.py',
 'scripts/review_frontend_canvas_comparison.py','tests/test_unified_count_canvas.py')


def read(p):return json.loads((ROOT/p).read_text())


def event(kind,**values):
    v=dict(kind=kind,recorded_at=datetime.now(timezone.utc).isoformat(),**values)
    with (FOLDER/'events.jsonl').open('a') as f:f.write(json.dumps(v,ensure_ascii=False)+'\n')
    print(json.dumps(v,ensure_ascii=False),flush=True)


def state(**values):
    p=FOLDER/'state.json';r=json.loads(p.read_text()) if p.exists() else {};r.update(values);write(p,r)


def fresh(plan,device='mps',old=False):
    torch.manual_seed(plan['seed'])
    cls=MambaCanvas if old else UnifiedCountCanvas
    model=cls(FullEventConfig(**plan['model']),density_reference=plan['density_reference'])
    if reset_mamba_boundaries(model)!=7:raise ValueError('Reset core count changed')
    return model.to(device)


def prepare():
    if PLAN.exists():raise FileExistsError('Keep frozen contracts')
    parent=read('bench/rhythm-v2/local-evolution-v1.json')['ssd']
    baseline=read('reports/local-evolution-v2/baseline-preflight.json')
    if not baseline['count']['training_support_sufficient'] or baseline['offset']['masked_target_fraction']>.01:
        raise ValueError('Support or target-mask precheck failed')
    plan={**parent,'id':'c06-reset-unified-count-law-v1','experiment_class':'local_mechanism','seed':20261032,
      'execution_ready':False,'predecessors':['evolution-ssd-reset-v1','evolution-ssd-full-v1','evolution-ar-audio-interaction-v1'],
      'shared_failure_modes':['split_count_presence_supervision','rare_type_allocation'],
      'mechanism_changes':{
       'split_count_presence_supervision':dict(change='Normalized hurdle count using existing existence and positive quantity rows; remove only independent balanced-existence loss',prediction='Natural-frequency count probability and selection become more consistent in this budget',discriminating_control='Same initial tensors, member heads and expected-tail-rank; natural scores and double count swaps'),
       'rare_type_allocation':dict(change='Count gate trained by the same natural NLL law decoded at inference, no type quotas',prediction='Non-tap cell/type multisetF1 improves>=.03 without count blowup or worse empty behavior',discriminating_control='Baseline exposure/probability/rank audit and fixed typed-cell count comparisons')},
      'pretraining_probe_receipts':[],'health_steps':100,'steps':1000,'schedule_steps':2000,
      'health_wall_s':900,'quality_wall_s':3600,'evaluation_wall_s':600,'process_wall_s':5400,'preflight_wall_s':300,
      'primary_mechanism_gate':dict(non_tap_count_f1=baseline['count']['non_tap']['f1']+.03,l1_max=309,positive_components=9,
         full_f1_min=.3041968377678335,onset10_min=.2960084493468935,count_ratio=[.5,1.75],empty_false_events_max=20),
      'health_gate':dict(joint_loss_ratio=.9,count_loss_ratio=.9,offset_tail_max_ratio=1.05,all_complete=True,parity_max=1e-4),
      'baseline_report':'reports/local-evolution-v1/ssd_reset.json','baseline_weight':'artifacts/local-evolution-v1/ssd_reset/quality-1000.pt',
      'baseline_preflight':'reports/local-evolution-v2/baseline-preflight.json',
      'single_intervention':'Unified presence/positive-count probability and NLL; original member heads, total/interval weights, native support and readout fixed',
      'effective_parameter_note':'Nominal count unchanged; quantity class0 rows unused (1028 parameters), report their zero gradients. Hurdle is not more expressive than categorical.',
      'post_diagnostics':'Same16firstwindows new/old count swaps;400ms valid local/mel value shift with each model original predicted count; exclude only fixed400ms edge region in diagnostic.',
      'failure_policy':'No automatic retries, resume, seed, gate change or prolonged training. Failure stays pending mechanism review.',
      'cloud_compute_calls':0,'sources':{n:sha(ROOT/n) for n in SOURCES}}
    write(PLAN,plan)
    for name in SOURCES:
        dst=ART/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
    shutil.copy2(PLAN,ART/'source/preflight-plan.json');event('contract_frozen',plan=str(PLAN.relative_to(ROOT)))


def diagnostic(model,batches):
    rows=[];model.eval()
    with torch.no_grad():
        for e,inp,t in batches:
            loss,parts=unified_count_loss(model(inp),t,inp)
            rows.append(dict(index=e['index'],loss=float(loss),parts={k:float(v) for k,v in parts.items()}))
    return rows


def checkpoint(model,opt,stage,step):
    path=ART/f'{stage}-{step:04d}.pt';path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),stage=stage,updates=step,
        plan_sha256=sha(PLAN),cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state()),path)
    return path


def probabilities(out):
    x=out['count'][0].detach().cpu().numpy().astype(np.float64)
    x=np.exp(x-x.max(-1,keepdims=True));return x/x.sum(-1,keepdims=True)


def count_metrics(out,target,inp,request):
    p=probabilities(out);valid=inp['canvas_valid'][0].cpu().numpy();truth=target['count'][0].cpu().numpy()
    chosen,_=expected_tail_rank(p[valid].reshape(-1,17),event_limit=request.max_events)
    pred=np.zeros((200,4),np.int64);pred[valid]=chosen.reshape(-1,4)
    a,b=pred[valid,1:],truth[valid,1:];inter=int(np.minimum(a,b).sum());pa=int(a.sum());tb=int(b.sum())
    scores={}
    for k,name in enumerate(('tap','hold','flick','drag')):
        pp=p[valid,k];tt=truth[valid,k];nll=-np.log(pp[np.arange(len(tt)),tt].clip(1e-300))
        brier=((pp-np.eye(17)[tt])**2).sum(-1);positive=tt>0
        scores[name]={s:dict(n=int(mask.sum()),nll=float(nll[mask].mean()) if mask.any() else None,brier=float(brier[mask].mean()) if mask.any() else None)
                      for s,mask in [('all',np.ones(len(tt),bool)),('positive',positive),('zero',~positive)]}
    return dict(pred=pa,true=tb,intersection=inter,f1=2*inter/max(pa+tb,1),l1=int(abs(a-b).sum()),proper_scores=scores)


@torch.no_grad()
def post_diagnostics(model,plan,row,panel,deadline):
    old=fresh(plan,'mps',old=True);old.load_state_dict(torch.load(ROOT/plan['baseline_weight'],map_location='cpu',weights_only=False)['model']);old.eval();model.eval()
    baseline=read(plan['baseline_preflight']);indices=[r['index'] for r in read('artifacts/c06-candidate-preflight-v1/reset-counts.json') if r['index'] not in plan['indices']['empty_dev']]
    rows=[];raw=[];saved_logits={}
    for index in indices:
        if time.monotonic()>deadline:raise TimeoutError('Frozen diagnostics deadline')
        e=row(index,'draft');inp,t=relative_batch([e],'mps');outs=dict(old=old(inp),new=model(inp))
        saved_logits[index]={name:{field:out[field].detach().cpu() for field in ('count','base_count','existence')} for name,out in outs.items()}
        cm={name:count_metrics(out,t,inp,e['request']) for name,out in outs.items()}
        conditions={**outs,'old_members_new_count':{**outs['old'],'count':outs['new']['count']},'new_members_old_count':{**outs['new'],'count':outs['old']['count']}}
        scored={}
        for name,out in conditions.items():
            pred=generate_relative(FrozenOutput(out),inp,e['request'],deadline=deadline)
            scored[name]=dict(prediction=pred,scores=score([RhythmEvent(**v) for v in pred['events']],e['targets']))
        shifted={**inp}
        for field,mask_field,axis,shift in [('local','local_valid',2,10),('mel','mel_valid',1,40)]:
            ids=inp[mask_field][0].nonzero().flatten();value=inp[field].clone()
            if field=='local':value[:,:,ids]=value[:,:,ids].roll(shift,axis)
            else:value[:,ids]=value[:,ids].roll(shift,axis)
            shifted[field]=value
        lo,hi=e['request'].start_us+400000,e['request'].end_us-400000
        audio={}
        for name,network in [('old',old),('new',model)]:
            shifted_out=network(shifted);shifted_out['count']=outs[name]['count']
            pred=generate_relative(FrozenOutput(shifted_out),inp,e['request'],deadline=deadline)
            values=[RhythmEvent(**v) for v in pred['events'] if lo<=v['onset_us']<hi]
            real=[RhythmEvent(**v) for v in scored[name]['prediction']['events'] if lo<=v['onset_us']<hi]
            truth=[v for v in e['targets'] if lo<=v.onset_us<hi]
            audio[name]=dict(real=score(real,truth),shifted=score(values,truth),same_member_denominator=len(values)==len(real))
        rows.append(dict(index=index,component=panel.records[index]['component'],count=cm,audio=audio))
        raw.append(dict(index=index,request=asdict(e['request']),target=[asdict(v) for v in e['targets']],conditions=scored))
        write(ART/'diagnostic-events.json',raw);write(ART/'count-audio-diagnostics.json',rows)
    total={name:{key:sum(r['count'][name][key] for r in rows) for key in ('pred','true','intersection','l1')} for name in ('old','new')}
    for name,v in total.items():v['f1']=2*v['intersection']/max(v['pred']+v['true'],1)
    total['component_wins']=sum(r['count']['new']['f1']>r['count']['old']['f1'] for r in rows)
    total['old_baseline_reproduced']=abs(total['old']['f1']-baseline['count']['non_tap']['f1'])<1e-12
    torch.save(saved_logits,ART/'count-logits.pt')
    del old;torch.mps.empty_cache()
    return dict(count_multiset=total,records='artifacts/c06-reset-unified-count-law-v1/count-audio-diagnostics.json',logits_sha256=sha(ART/'count-logits.pt'),
        fixed_count_scope='Entire predicted cell/type/count tensor; no hidden counts.400ms shift is encoded-value intervention, not waveform/emotion evidence.')


def run():
    if REPORT.exists():raise FileExistsError('No automatic restart')
    plan=read(str(PLAN.relative_to(ROOT)));started=time.monotonic();panel,row=load_data()
    for name,expected in plan['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Source drift: '+name)
    cpu=fresh(plan,'cpu').eval();gpu=fresh(plan,'mps').eval();initial=weight_digest(cpu)
    if initial!=read('reports/local-evolution-v1/ssd_reset.json')['initial_weight_sha256']:raise ValueError('Common initialization changed')
    e=row(plan['health_indices'][0],'draft');ci,ct=relative_batch([e],'cpu');mi,mt=relative_batch([e],'mps')
    co=cpu(ci);mo=gpu(mi);cl,_=unified_count_loss(co,ct,ci);ml,_=unified_count_loss(mo,mt,mi);cl.backward();ml.backward();torch.mps.synchronize()
    if not all(bool(torch.isfinite(v).all()) for v in (*ci.values(),cl,ml)):
        raise ValueError('Nonfinite actual input or loss')
    gp=dict(gpu.named_parameters());gradient=0.
    for name,param in cpu.named_parameters():
        if (param.grad is None)!=(gp[name].grad is None):raise ValueError('Gradient presence mismatch')
        if param.grad is not None:
            if not bool(torch.isfinite(param.grad).all() and torch.isfinite(gp[name].grad).all()):raise ValueError('Nonfinite actual gradient')
            gradient=max(gradient,float((param.grad-gp[name].grad.cpu()).abs().max()))
    parity=max(abs(float(cl.detach())-float(ml.detach())),gradient)
    if parity>1e-4:raise ValueError('CPU/MPS parity')
    old=fresh(plan,'cpu',old=True);old.load_state_dict(torch.load(ROOT/plan['baseline_weight'],map_location='cpu',weights_only=False)['model']);old.eval()
    frozen=read('artifacts/evolution-terminal-diagnostics-v1/ssd-reset_off.json');exact=0
    for saved in frozen:
        sample=row(saved['index'],'draft');inp,_=relative_batch([sample],'cpu');pred=generate_relative(old,inp,sample['request'])
        exact+=pred['events']==saved['prediction']['events']
    if exact!=16:raise ValueError('Frozen baseline replay drift')
    receipt=dict(status='completed',passed=True,scope='new_implementation',initial_weight_sha256=initial,parity_max=parity,
        loss_error=abs(float(cl.detach())-float(ml.detach())),gradient_error=gradient,exact_baseline_replays=exact,
        nominal_parameters=sum(v.numel() for v in cpu.parameters()),unused_quantity_parameters=1028,optimizer_updates=0,cloud_compute_calls=0,
        shapes={k:list(v.shape) for k,v in ci.items()},finite_inputs=all(bool(torch.isfinite(v).all()) for v in ci.values()),
        probability_sum_error=float((co['count'].exp().sum(-1)-1).abs().max()),
        unused_quantity_zero_row_gradient_max=float(cpu.quantity.weight.grad[torch.arange(4)*17].abs().max()))
    write(FOLDER/'unified-count-preflight.json',receipt)
    if time.monotonic()-started>plan['preflight_wall_s']:raise TimeoutError('Preflight wall')
    del cpu,gpu,old,co,mo,cl,ml,gp;gc.collect();torch.mps.empty_cache()
    plan.update(execution_ready=True,pretraining_probe_receipts=[dict(status='completed',scope='new_implementation',evidence='reports/local-evolution-v2/unified-count-preflight.json')]);write(PLAN,plan)
    if validate_successor(plan,{r['id'] for r in load_registry()}):raise ValueError('Mechanism dispatch validation')
    shutil.copy2(PLAN,ART/'source/frozen-plan.json')
    report=dict(id=plan['id'],status='health_running',plan_sha256=sha(PLAN),initial_weight_sha256=initial,
        health_updates=0,quality_updates=0,health_weights_used=False,curves=[],failures=[],cloud_compute_calls=0)
    state(status='running',active_report=str(REPORT.relative_to(ROOT)),pid=os.getpid());write(REPORT,report)
    model=fresh(plan);torch.mps.manual_seed(plan['seed']);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0)
    specs=list(zip(plan['health_indices'],('draft','single_condition','random_groups','middle')))+[(plan['empty_health_index'],'draft')]
    batches=[]
    for i,v in specs:
        sample=row(i,v);batches.append((sample,*relative_batch([sample],'mps')))
    report['initial_health']=diagnostic(model,batches);stage='health'
    try:
        for stage,steps in [('health',100),('quality',1000)]:
            if stage=='quality':
                del model,opt,batches;gc.collect();torch.mps.empty_cache();model=fresh(plan);torch.mps.manual_seed(plan['seed'])
                if weight_digest(model)!=initial:raise ValueError('Fresh quality initialization')
                opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0)
            moment=time.monotonic();report['status']=stage+'_running';event('stage_started',stage=stage)
            for step in range(1,steps+1):
                if time.monotonic()-started>plan['process_wall_s'] or time.monotonic()-moment>plan[stage+'_wall_s']:raise TimeoutError(stage+' time cap')
                model.train();opt.zero_grad(set_to_none=True)
                progress=(step-plan['warmup'])/(2000-plan['warmup'])
                lr=.001*step/plan['warmup'] if step<=plan['warmup'] else .0001+(.001-.0001)*(1+math.cos(math.pi*progress))/2
                for g in opt.param_groups:g['lr']=lr
                items=batches if stage=='health' else [(e,*relative_batch([e],'mps')) for spec in plan['sample_stream'][step-1] for e in [row(spec['index'],spec['view'],spec['field_index'])]]
                total=0.
                for e,inp,t in items:
                    loss,_=unified_count_loss(model(inp),t,inp);(loss/len(items)).backward();total+=float(loss.detach())/len(items)
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();torch.mps.synchronize();report[stage+'_updates']=step
                if step==1 or step%10==0:
                    report['curves'].append(dict(stage=stage,step=step,loss=total,gradient=norm,lr=lr,wall_s=time.monotonic()-moment,
                        mps_allocated_bytes=torch.mps.current_allocated_memory(),mps_driver_bytes=torch.mps.driver_allocated_memory(),mps_recommended_bytes=torch.mps.recommended_max_memory()))
                    write(REPORT,report);print(json.dumps(report['curves'][-1]),flush=True)
                if step in (100,250,500,1000):checkpoint(model,opt,stage,step)
            report[stage+'_wall_s']=time.monotonic()-moment
            if stage=='health':
                after=diagnostic(model,batches);before=report['initial_health'];report['final_health']=after;gates={}
                gates['loss']=np.mean([v['loss'] for v in after])<=.9*np.mean([v['loss'] for v in before])
                for field in ('count','offset','tail'):
                    limit=.9 if field=='count' else 1.05;gates[field]=np.mean([v['parts'][field] for v in after])<=limit*np.mean([v['parts'][field] for v in before])
                predictions=[];model.eval()
                for e,inp,t in batches:
                    pred=generate_relative(model,inp,e['request'],deadline=min(started+5400,moment+900))
                    predictions.append(dict(index=e['index'],prediction=pred,target=[asdict(v) for v in e['targets']],scores=score([RhythmEvent(**v) for v in pred['events']],e['targets'])))
                gates['complete']=all(v['prediction']['complete'] for v in predictions);report['health_gates']={k:bool(v) for k,v in gates.items()};report['health_passed']=all(gates.values())
                write(ART/'health-predictions.json',predictions);write(REPORT,report)
                if not report['health_passed']:
                    report['status']='health_failed_pending_mechanism_review';write(REPORT,report);state(status=report['status'],pid=None);return
        final=checkpoint(model,opt,'quality',1000);report['final_weight_sha256']=sha(final);report['status']='evaluating';write(REPORT,report)
        moment=time.monotonic();deadline=min(started+5400,moment+600)
        evaluator.ART=ART;evaluator.LIVE['panel']=panel
        report['evaluations']=evaluator.evaluate({'unified':model},plan,row,'quality-1000',deadline,full=True)['unified']
        report['mechanism_diagnostics']=post_diagnostics(model,plan,row,panel,deadline)
        q=report['evaluations']['draft'];c=report['mechanism_diagnostics']['count_multiset'];gate=plan['primary_mechanism_gate']
        report['empty_false_events']=sum(report['evaluations']['empty']['predicted_types'].values())
        report['mechanism_gate']=dict(non_tap_gain=c['new']['f1']>=gate['non_tap_count_f1'],l1=c['new']['l1']<=309,components=c['component_wins']>=9,
            full_noninferiority=q['full_note_f1']>=gate['full_f1_min'],onset_noninferiority=q['onset10']>=gate['onset10_min'],
            density=.5<=q['density_ratio']<=1.75,empty=report['empty_false_events']<=20)
        report.update(status='terminal_pending_mechanism_review',evaluation_wall_s=time.monotonic()-moment,total_process_wall_s=time.monotonic()-started)
        write(REPORT,report);state(status=report['status'],pid=None);event('terminal_ready',report=str(REPORT.relative_to(ROOT)),mechanism_gate=report['mechanism_gate'])
    except BaseException as exc:
        report['status']='stopped_pending_mechanism_review';report['failures'].append(dict(stage=stage,error=f'{type(exc).__name__}: {exc}'))
        checkpoint(model,opt,stage+'-interrupted',report[stage+'_updates']);write(REPORT,report);state(status=report['status'],pid=None);raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run'));args=parser.parse_args()
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('Silent fallback prohibited')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5)
    def stop(*_):raise KeyboardInterrupt('External stop')
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    {'prepare':prepare,'run':run}[args.action]()


if __name__=='__main__':main()
