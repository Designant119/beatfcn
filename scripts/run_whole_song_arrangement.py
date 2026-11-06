"""One complete local mel/flux/full-song IOHMM candidate; no cloud calls."""
from collections import Counter
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

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.sample import match_f1
from beatfcn.model.whole_song_arrangement import WholeSongArrangement,arrangement_loss,generate_arrangement
from beatfcn.train.canonical_probability import explicit_row,encode_canonical
from beatfcn.train.whole_song_mel import WholeMelStore
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_group_diffusion_health import optimizer_for,prior_for,rate
from scripts.run_local_portfolio_scale import row_factory,weight_digest,sha,write
from scripts.run_local_typed_canvas import score
from scripts.review_frontend_canvas_comparison import metrics


ROOT=Path(__file__).resolve().parents[1]
IDENT='local-whole-song-arrangement-v1'
PLAN=ROOT/f'bench/rhythm-v2/{IDENT}.json'
REPORT=ROOT/'reports/local_whole_song_arrangement_v1.json'
ART=ROOT/f'artifacts/{IDENT}'


def fresh(plan,device='mps'):
    torch.manual_seed(plan['seed'])
    model=WholeSongArrangement(FullEventConfig(**plan['model']),density_reference=plan['density_reference'])
    prior=torch.tensor(plan['count_prior'],dtype=torch.float32)
    with torch.no_grad():
        model.count_gate.weight.zero_();model.count_positive.weight.zero_()
        model.count_gate.bias.copy_((1-prior[:,0]).log()-prior[:,0].log())
        model.count_positive.bias.copy_((prior[:,1:]/(1-prior[:,:1])).log().flatten())
    return model.to(device)


def journal(text):
    with (ROOT/'EXPERIMENT_LOG.md').open('a') as f:f.write(f'\n\n{IDENT} {text}\n')


def structure(events,request):
    records=[]
    for start in range(request.start_us,request.end_us,2_000_000):
        end=min(start+2_000_000,request.end_us);selected=[e for e in events if start<=e.onset_us<end]
        c=Counter(e.onset_us for e in selected);n=len(selected)
        cover=sum(max(0,min(e.end_us,end)-max(e.onset_us,start)) for e in events if e.type=='hold')/(end-start)
        records.append([math.log1p(n)]+[sum(e.type==k for e in selected)/max(n,1) for k in ('tap','hold','flick','drag')]+
            [sum(v for v in c.values() if v>1)/max(n,1),cover])
    return np.asarray(records,np.float64)


def diagnostics(model,batches):
    model.eval();records=[]
    with torch.no_grad():
        for e,inputs,target in batches:
            loss,parts=arrangement_loss(model,inputs,target,e['request'])
            records.append(dict(index=e['index'],loss=float(loss),parts={k:float(v) for k,v in parts.items()}))
    return records


def checkpoint(model,opt,stage,step):
    path=ART/f'{stage}-{step:04d}.pt';path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),stage=stage,updates=step,plan_sha256=sha(PLAN),
        cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state()),path)
    return path


def evaluate(model,plan,row,panel,store,deadline):
    dev=plan['indices']['dev'];selected=dev[::2]
    tasks={
        'draft':[(i,'draft',None,False,False,False) for i in dev],
        'empty':[(i,'draft',None,False,False,False) for i in plan['indices']['empty_dev']],
        'remote_permuted':[(i,'draft',None,False,True,False) for i in dev],
        'completion':[(i,'middle',None,False,False,False) for i in selected],
        'reference_masked':[(i,'middle',None,True,False,False) for i in selected],
        'density_low':[(i,'draft',(2.,3.),False,False,False) for i in selected[::4]],
        'density_high':[(i,'draft',(6.,8.),False,False,False) for i in selected[::4]],
        'prior_viterbi_cardinality':[(i,'draft',None,False,False,True) for i in dev],
    }
    summaries={};scales=np.asarray(plan['structure_scale'])
    for task,specs in tasks.items():
        samples=[]
        for sample_id in range(1 if task=='prior_viterbi_cardinality' else 4):
            records=[]
            for index,view,band,masked,remote,deterministic in specs:
                if time.monotonic()>deadline:raise TimeoutError('Arrangement evaluation wall')
                e=explicit_row(row(index,view,0,band));audio=panel.records[index]['audio_sha256']
                inputs,target=store.batch(e,audio,mask_reference=masked,remote_permutation=remote)
                try:
                    pred,info=generate_arrangement(model,inputs,e['request'],seed=plan['seed']+17*index+sample_id,
                        deterministic=deterministic,deadline=deadline)
                    prediction=dict(complete=True,events=[asdict(v) for v in pred],diagnostics=info)
                except (ValueError,RuntimeError,TimeoutError) as exc:
                    pred=[];prediction=dict(complete=False,events=[],failure=f'{type(exc).__name__}: {exc}')
                s=score(pred,e['targets']);start=e['request'].start_us;width=e['request'].end_us-start
                s['extra_onset_tolerances']={str(ms):match_f1([v.onset_us for v in pred],[v.onset_us for v in e['targets']],ms*1000)[3] for ms in (1,5,25)}
                shifts=[match_f1([start+(v.onset_us-start+d)%width for v in pred],[v.onset_us for v in e['targets']],10000)[3] for d in plan['shifts_us']]
                pvec,tvec=structure(pred,e['request']),structure(e['targets'],e['request'])
                error=np.mean(np.abs(pvec-tvec)/scales,axis=0)
                records.append(dict(index=index,component=panel.records[index]['component'],view=view,request=asdict(e['request']),
                    prediction=prediction,target=[asdict(v) for v in e['targets']],scores=s,shifted_onset10=shifts,
                    structure_predicted=pvec.tolist(),structure_target=tvec.tolist(),structure_error=error.tolist()))
                write(ART/f'{task}-sample{sample_id}.json',records)
            result=metrics(records);result['structure_error']=np.mean([v['structure_error'] for v in records],axis=0).tolist();samples.append(result)
        summary={field:float(np.mean([s[field] for s in samples])) for field in ('full_note_f1','onset10','onset50','shift_null10','density_ratio','nonempty','complete','attempted')}
        summary.update(samples=samples,structure_error=np.mean([s['structure_error'] for s in samples],axis=0).tolist(),
            components={c:{field:float(np.mean([s['components'][c][field] for s in samples])) for field in ('full_note_f1','onset10','onset50','shift_null10')} for c in samples[0]['components']})
        summaries[task]=summary
    return summaries


def main():
    if PLAN.exists() or REPORT.exists():raise FileExistsError('Preserve this candidate, no implicit restart')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('No silent backend fallback')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5)
    parent=json.loads((ROOT/'bench/rhythm-v2/local-portfolio-scale-v1.json').read_text())
    panel,features=setup(json.loads((ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()))
    row=row_factory(panel,features);store=WholeMelStore(ROOT,panel.features.records,ART/'whole-mel')
    histogram,prior=prior_for(panel,parent['indices']['train64'])
    model={**parent['models']['capacity64'],'d':128,'heads':4,'kv_heads':2,'ffn':384,'encoder_layers':0}
    started=time.monotonic();cache={};vectors=[]
    admitted=set(parent['indices']['train64']+parent['indices']['dev']+parent['indices']['empty_dev'])
    for index in admitted:
        audio=panel.records[index]['audio_sha256']
        if audio not in cache:
            summary,times=store.open(audio);rec=panel.features.records[audio]
            cache[audio]=dict(source=rec['complete_mel'],source_sha256=sha(ROOT/rec['complete_mel']),
                decoded_samples=rec['decoded_samples'],blocks=len(times),coverage_end_s=rec['decoded_samples']/24000,
                cache_sha256=sha(ART/'whole-mel'/f'{audio}.npz'))
        if index in parent['indices']['train64']:
            e=row(index,'draft');vectors.extend(structure(e['targets'],e['request']))
        if time.monotonic()-started>1800:raise TimeoutError('CPU preparation1800s')
    scales=np.maximum(np.std(vectors,axis=0),.1)
    sources=('scripts/run_whole_song_arrangement.py','beatfcn/model/whole_song_arrangement.py','beatfcn/model/arrangement_iohmm.py',
        'beatfcn/train/whole_song_mel.py','beatfcn/model/canonical_event_package.py','beatfcn/model/cardinality_dp.py',
        'beatfcn/train/canonical_probability.py','beatfcn/model/local_typed_canvas.py','beatfcn/model/full_event_model.py')
    plan=dict(id=IDENT,seed=20261054,model=model,density_reference=parent['density_reference'],count_prior=prior.tolist(),
        count_histogram=histogram.tolist(),indices=parent['indices'],sample_stream=parent['sample_stream']['train64'][:1000],
        health_indices=parent['health_indices'],empty_health_index=parent['empty_health_index'],health_steps=100,quality_steps=1000,
        learning_rate=.0003,minimum_learning_rate=.00003,warmup=50,weight_decay=.01,clip=1.,effective_batch=2,
        health_wall_s=900,quality_wall_s=9000,evaluation_wall_s=1800,max_parameters=8_000_000,
        whole_audio_identity=cache,cpu_preparation_s=time.monotonic()-started,structure_scale=scales.tolist(),shifts_us=parent['shifts_us'],
        core_hypothesis='From-scratch localmel+multiscaleFlux, complete-song deep/narrow context and2s4state marginal event likelihood acquire useful long-context chart choices. Complete recipe, no emotion labels.',
        primary='FourT1prior-path and canonical-emitter samples equal-weighted; deterministic input-only priorViterbi+cardinality is secondary.',
        health_gate='Real5request100update: jointNLL decreases10%, onset/tail diagnostic means not worse5%, finite CPU/MPS<=1e-4 and5 legal free samples; no productF1 or type threshold.',
        local_screen=dict(min_nonempty=24,count_ratio=[.5,1.75],onset10=.20,onset10_minus_shift=.05,positive_components=10,full_note_f1=.20),
        boundaries='Whole-song input is complete admitted10msmel, no new pretrained feature/weights. Sparse requests label only H; remaining emissions E=1. No hidden carry, no quotas/NMS/min-gap. Emotion/listening/edit/play evidence absent.',
        whole_song_rollout='Prior path reused across adjacent windows if later rollout requested; this experiment evaluates frozen sparse windows, does not claim whole-song rollout.',
        confirmation_scored=False,cloud_compute_calls=0,source_sha256={name:sha(ROOT/name) for name in sources})
    write(PLAN,plan)
    for name in sources:
        dest=ART/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    shutil.copy2(PLAN,ART/'source/frozen-plan.json')
    model=fresh(plan);torch.mps.manual_seed(plan['seed']);opt=optimizer_for(model,plan)
    parameters=sum(p.numel() for p in model.parameters())
    if parameters>plan['max_parameters']:raise ValueError('Parameter cap')
    report=dict(id=IDENT,status='health_running',plan_sha256=sha(PLAN),parameters=parameters,initial_weight_sha256=weight_digest(model),
        health_updates=0,quality_updates=0,curves=[],failures=[],health_weights_used=False,cloud_compute_calls=0,confirmation_scored=False)
    write(REPORT,report);journal(f'完整合同冻结，{parameters}参数，整曲音频{len(cache)}个；100真实健康后fresh1000；确认密封、云0。')
    def stopped(*_):raise KeyboardInterrupt('External interruption')
    signal.signal(signal.SIGINT,stopped);signal.signal(signal.SIGTERM,stopped)
    stage='health';started=time.monotonic()
    try:
        specs=list(zip(plan['health_indices'],('draft','single_condition','random_groups','middle')))+[(plan['empty_health_index'],'draft')]
        batches=[]
        for index,view in specs:
            e=explicit_row(row(index,view));inputs,target=store.batch(e,panel.records[index]['audio_sha256']);batches.append((e,inputs,target))
        report['input_audit']=[dict(index=e['index'],shapes={k:list(v.shape) for k,v in inp.items()},
            statistics={k:dict(finite=bool(torch.isfinite(v).all()),min=float(v.min()),max=float(v.max()),mean=float(v.float().mean()),std=float(v.float().std())) for k,v in inp.items()}) for e,inp,_ in batches]
        report['initial_diagnostic']=diagnostics(model,batches)
        cpu=fresh(plan,'cpu').eval();e,inp,tgt=batches[0]
        ci,ct=({k:v.cpu() for k,v in d.items()} for d in (inp,tgt))
        with torch.no_grad():a,b=cpu(ci),model(inp)
        forward=max(float((a[k]-b[k].cpu()).abs().max()) for k in a)
        cl,_=arrangement_loss(cpu,ci,ct,e['request']);ml,_=arrangement_loss(model,inp,tgt,e['request']);cl.backward();ml.backward()
        cp=dict(cpu.named_parameters());gradient=0.
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
                loss,_=arrangement_loss(model,inp,tgt,e['request']);(loss/5).backward();total+=float(loss.detach())/5
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();report['health_updates']=step
            if step%10==0 or step==1:
                point=dict(stage=stage,step=step,loss=total,gradient=grad,wall_s=time.monotonic()-started)
                report['curves'].append(point);write(REPORT,report);print(json.dumps(point),flush=True)
            if step%25==0:checkpoint(model,opt,stage,step)
        report['final_diagnostic']=diagnostics(model,batches);predictions=[]
        for e,inp,tgt in batches:
            events,info=generate_arrangement(model,inp,e['request'],seed=plan['seed']+e['index'],deadline=started+plan['health_wall_s'])
            predictions.append(dict(index=e['index'],prediction=dict(complete=True,events=[asdict(v) for v in events],diagnostics=info),target=[asdict(v) for v in e['targets']],scores=score(events,e['targets'])))
        write(ART/'health-predictions.json',predictions)
        first,last=report['initial_diagnostic'],report['final_diagnostic']
        gates=dict(loss=np.mean([v['loss'] for v in last])<=.9*np.mean([v['loss'] for v in first]),
            onset=np.mean([v['parts']['onset'] for v in last])<=1.05*np.mean([v['parts']['onset'] for v in first]),
            tail=np.mean([v['parts']['tail'] for v in last])<=1.05*np.mean([v['parts']['tail'] for v in first]),complete=len(predictions)==5)
        gates={k:bool(v) for k,v in gates.items()}
        report.update(health_gates=gates,health_passed=bool(all(gates.values())),health_wall_s=time.monotonic()-started)
        if not report['health_passed']:
            report['status']='health_failed';write(REPORT,report);journal(f'健康失败结案：{gates}。不延长、不改门。');return
        checkpoint(model,opt,'health',100);del model,opt,batches;torch.mps.empty_cache()
        model=fresh(plan);torch.mps.manual_seed(plan['seed']);opt=optimizer_for(model,plan)
        if weight_digest(model)!=report['initial_weight_sha256']:raise ValueError('Fresh initialization mismatch')
        stage='quality';started=time.monotonic();report['status']='quality_running';write(REPORT,report);journal('真实健康通过，fresh1000开始；不加载健康权重。')
        for step in range(1,1001):
            if time.monotonic()-started>plan['quality_wall_s']:raise TimeoutError('Quality wall')
            model.train();opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']=rate(plan,step)
            total=0.
            for spec in plan['sample_stream'][step-1]:
                e=explicit_row(row(spec['index'],spec['view'],spec['field_index']));inp,tgt=store.batch(e,panel.records[spec['index']]['audio_sha256'])
                loss,_=arrangement_loss(model,inp,tgt,e['request']);(loss/2).backward();total+=float(loss.detach())/2
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();report['quality_updates']=step
            if step%25==0 or step==1:
                point=dict(stage=stage,step=step,loss=total,gradient=grad,wall_s=time.monotonic()-started,
                    tensor_bytes=torch.mps.current_allocated_memory(),driver_bytes=torch.mps.driver_allocated_memory())
                report['curves'].append(point);write(REPORT,report);print(json.dumps(point),flush=True)
            if step%50==0:checkpoint(model,opt,stage,step)
        path=checkpoint(model,opt,'quality',1000);report['final_weight_sha256']=sha(path);report['training_wall_s']=time.monotonic()-started
        report['status']='evaluating';write(REPORT,report);moment=time.monotonic()
        result=evaluate(model,plan,row,panel,store,min(started+plan['quality_wall_s'],moment+plan['evaluation_wall_s']))
        d=result['draft'];g=plan['local_screen']
        gates=dict(complete=d['complete']==d['attempted'],nonempty=d['nonempty']>=g['min_nonempty'],count_ratio=g['count_ratio'][0]<=d['density_ratio']<=g['count_ratio'][1],
            onset10=d['onset10']>=g['onset10'],onset_null=d['onset10']-d['shift_null10']>=g['onset10_minus_shift'],
            components=sum(v['onset10']>v['shift_null10'] for v in d['components'].values())>=g['positive_components'],full_f1=d['full_note_f1']>=g['full_note_f1'])
        report.update(status='completed',evaluations=result,evaluation_wall_s=time.monotonic()-moment,total_quality_wall_s=time.monotonic()-started,
            local_screen=gates,local_screen_passed=bool(all(gates.values())))
        write(REPORT,report);journal(f'固定1000和全部自由评价完成，本地筛选{gates}。')
    except BaseException as exc:
        report.update(status='stopped',stopped_stage=stage,stage_wall_s=time.monotonic()-started)
        report['failures'].append(f'{type(exc).__name__}: {exc}');checkpoint(model,opt,f'{stage}-interrupted',report[f'{stage}_updates'])
        write(REPORT,report);journal(f'停止保留现场：{report["failures"][-1]}。');raise


if __name__=='__main__':main()
