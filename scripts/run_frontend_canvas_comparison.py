"""Local matched Pupu/MERT2-FullSong route screen for an H100/0.25B pilot."""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from dataclasses import asdict
import json
import math
import signal
import time

import numpy as np
import torch

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_typed_canvas import LocalTypedCanvas,canvas_loss
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.train.mert_canvas_features import MertCanvasFeatures
from beatfcn.train.local_bundle import make_row
from beatfcn.train.relative_canvas import relative_batch,generate_relative,relative_targets
from beatfcn.train.local_typed_canvas import encode_canvas,decode_target
from scripts.run_local_bundle import ROOT,write,sha,log,configure
from scripts.run_local_typed_canvas import panel_for,score,tensor_stats


PLAN=ROOT/'bench/rhythm-v2/local-frontend-comparison-v1.json'
REPORT=ROOT/'reports/local_frontend_comparison_v1.json'
ART=ROOT/'artifacts/local-frontend-comparison-v1'
ARMS=('pupu','mert');LIVE={}


def journal(text):
    with (ROOT/'EXPERIMENT_LOG.md').open('a') as f:f.write('\n\nlocal-frontend-comparison-v1 '+text+'\n')


def save(report):
    write(REPORT,report);path=ROOT/'reports/rhythm_v2_budget.json';b=json.loads(path.read_text())
    b['local_cpu_diagnostics']['local-frontend-comparison-v1']={
        'status':report['status'],'updates':report['updates'],'health_updates':report['health_updates'],
        'wall_s':report.get('wall_s',{}),'new_cloud_spend_usd':0,'new_cloud_compute_calls':0,
        'report':str(REPORT.relative_to(ROOT)),'resource_target':'Modal H100 ~0.25B trainable event model, not product release'}
    write(path,b)


def models(plan):
    torch.manual_seed(plan['seed']);p=LocalTypedCanvas(FullEventConfig(**plan['models']['pupu']),density_reference=plan['density_reference'])
    torch.manual_seed(plan['seed']);m=LocalTypedCanvas(FullEventConfig(**plan['models']['mert']),density_reference=plan['density_reference'])
    control=p.state_dict();state=m.state_dict()
    for name,value in state.items():
        if value.shape==control[name].shape:state[name]=control[name].clone()
        elif name in ('layer_mix','band_embedding','band_queries'):
            state[name]=control[name][tuple(slice(0,s) for s in value.shape)].clone()
        else:raise ValueError('Unexpected unshared parameter '+name)
    m.load_state_dict(state)
    for name,value in m.state_dict().items():
        if value.shape==control[name].shape and not torch.equal(value,control[name]):raise AssertionError('Common initialization')
    return dict(pupu=p,mert=m)


def setup(plan):
    panel=panel_for(plan);store=MertCanvasFeatures(ROOT,panel.features,plan['mert_cache_report'])
    return panel,store


def rows_for(panel,store,specs):
    base=[make_row(panel,s['index'],s['view'],field_index=s.get('field_index',0),manual_bands=s.get('bands')) for s in specs]
    native=[]
    for row in base:
        audio=panel.records[row['index']]['audio_sha256'];request=row['request']
        window=store.window(audio,request.start_us,request.end_us)
        if not np.array_equal(window['mel'],row['inputs']['mel']):raise AssertionError('Common mel changed')
        native.append({**row,'inputs':{**row['inputs'],**window}})
    return dict(pupu=base,mert=native)


def verify(plan):
    for path,expected in {**plan['source_sha256'],**plan['mert_cache_sha256']}.items():
        if sha(ROOT/path)!=expected:raise ValueError('Frozen source/cache changed: '+path)


def prepare():
    if PLAN.exists() or REPORT.exists():raise FileExistsError('Preserve old comparison')
    old=json.loads((ROOT/'bench/rhythm-v2/local-typed-canvas-v1.json').read_text())
    fresh_ids=json.loads((ROOT/'bench/rhythm-v2/canvas-count-pilot-v1.json').read_text())
    p={k:deepcopy(old[k]) for k in ('seed','tape','features','indices','chart_ids','data_summary','health_indices','empty_health_index','schedule','density_reference')}
    native=dict(old['model']);native.update(taps=2,bands=1,band_slots=1)
    p.update(id='local-frontend-comparison-v1',models=dict(pupu=old['model'],mert=native),
        mert_cache_report='reports/d96_local_mert_cache_workers_v2.json',
        new_val_indices=fresh_ids['indices'],empty_val_indices=fresh_ids['empty_indices'],
        health_steps=100,steps=900,effective_batch=2,health_wall_s=600,train_wall_s=2400,evaluation_wall_s=600,
        cloud_compute_limit=0,resource_target='Modal H100 ~0.25B trainable event-model pilot',
        purpose='Frontend-route evidence, not small-model release readiness or extrapolated proof of large-model ranking',
        common_changes='Cell-base-relative native-ms tail; fixed model-expected-count/tail-rank decode; no dynamic generated-prefix replay, all six human-reference views retained',
        screen=dict(mert_full_gain=.02,mert_onset10_gain=.03,minimum_components_improved=10,
            acoustic_onset10_null_gain=.05,minimum_acoustic_components=10),
        shifts_us=[137000,281000,419000,733000,1103000,1741000,2593000])
    panel,store=setup(p);indices=p['indices']['train']+p['indices']['val']+p['new_val_indices']+p['empty_val_indices']
    audios={panel.records[i]['audio_sha256'] for i in indices};cache_sha={};contexts=Counter();feature_audit=[]
    for audio in sorted(audios):
        if audio not in store.mert:raise ValueError('Missing MERT cache: '+audio)
        rec=store.mert[audio];folder=ROOT/rec['path']
        for name,entry in rec['files'].items():
            path=folder/name
            if sha(path)!=entry['sha256']:raise ValueError('MERT file digest mismatch')
            cache_sha[str(path.relative_to(ROOT))]=entry['sha256']
        metadata=json.loads((folder/'metadata.json').read_text())
        if metadata['repo']!='m-a-p/MERT-v2-FullSong' or metadata['revision']!='7001f7657276238629e61c7746ce70735e0604c0':
            raise ValueError('Wrong pretrained frontend identity')
        if rec['samples']!=store.records[audio]['decoded_samples']:raise ValueError('Sample clocks differ')
        contexts['full_song' if metadata['full_song_context'] else 'long_song_stitched']+=1
        feature_audit.append(dict(audio_sha256=audio,frames=rec['frames'],samples=rec['samples'],
            full_song_context=metadata['full_song_context'],windows=len(metadata['windows']),
            checkpoint_sha256=metadata['checkpoint_sha256']))
    p['mert_cache_sha256']=cache_sha;p['feature_context_audit']=feature_audit
    # Prove relative endpoint representation roundtrips native events before updates.
    from beatfcn.train.local_request_panel import event_request
    for index in indices:
        request=event_request(panel.records[index]['request']);events=panel.targets(index);absolute=encode_canvas(events,request)
        target={k:torch.from_numpy(v)[None] for k,v in absolute.items()};relative=relative_targets(target,[request])
        delta=relative['tail_block']*2000+relative['tail_cell']*40+relative['tail_ms'];base=request.start_us//1000+torch.arange(200)[None,:,None]*40
        ends=delta+base;holds=relative['slot_valid'][:,:,1]
        restored={k:v.clone() for k,v in target.items()}
        restored['tail_block']=torch.where(holds,ends//2000,0);restored['tail_cell']=torch.where(holds,(ends%2000)//40,0);restored['tail_ms']=torch.where(holds,ends%40,0)
        if decode_target({k:v[0].numpy() for k,v in restored.items()},request)!=sorted(events):raise ValueError('Relative clock lost events')
    made=models(p);parameters={a:sum(v.numel() for v in model.parameters()) for a,model in made.items()}
    if max(parameters.values())>3_000_000:raise ValueError('Small-model parameter cap')
    paths=set(old['source_sha256'])|set(cache_sha)|{
        'beatfcn/train/mert_canvas_features.py','beatfcn/train/relative_canvas.py','beatfcn/model/count_readout.py',
        'scripts/run_frontend_canvas_comparison.py','design/LOCAL_FRONTEND_COMPARISON_v1.md',p['mert_cache_report']}
    for i in p['new_val_indices']+p['empty_val_indices']:paths.add(panel.charts[panel.records[i]['chart_index']]['target_path'])
    p['source_sha256']={x:sha(ROOT/x) for x in paths if x not in cache_sha};p['frozen_at_unix']=time.time()
    ART.mkdir(exist_ok=True);write(ART/'budget-before.json',json.loads((ROOT/'reports/rhythm_v2_budget.json').read_text()))
    write(PLAN,p);report=dict(id=p['id'],status='prepared',complete=False,updates={a:0 for a in ARMS},
        health_updates={a:0 for a in ARMS},parameters=parameters,common_initialization_exact=True,
        feature_context_counts=dict(contexts),health={},curves=[],evaluations={},failures=[],wall_s={},cloud_compute_calls=0)
    save(report);journal(f'准备完成：参数{parameters}，{len(audios)}音频MERT逐文件核验、上下文{dict(contexts)}；共同参数初始化精确，原生/相对尾点无损。两臂100健康+900质量，0云。')
    log(dict(stage='prepared',parameters=parameters,contexts=dict(contexts),cache_files=len(cache_sha)))


def checkpoint(model,opt,arm,stage,step):
    path=ART/arm/f'{stage}-{step:04d}.pt';path.parent.mkdir(exist_ok=True,parents=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),arm=arm,stage=stage,updates=step,
        plan_sha256=sha(PLAN),cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state()),path)


def health(p,r):
    if r['status']!='prepared':raise ValueError('Health already attempted')
    started=time.monotonic();panel,store=setup(p)
    specs=[dict(index=i,view=v) for i,v in zip(p['health_indices'],('draft','single_condition','random_groups','middle'))]
    specs.append(dict(index=p['empty_health_index'],view='draft'));rows=rows_for(panel,store,specs)
    made=models(p);batches={a:[relative_batch([row]) for row in rows[a]] for a in ARMS};parity={};audit={}
    for arm,model in made.items():
        model.eval();audit[arm]=tensor_stats(batches[arm][0][0])
        with torch.no_grad():cpu={k:v.cpu() for k,v in model(batches[arm][0][0]).items()}
        model.to('mps');batches[arm]=[({k:v.to('mps') for k,v in x.items()},{k:v.to('mps') for k,v in y.items()}) for x,y in batches[arm]]
        with torch.no_grad():gpu=model(batches[arm][0][0])
        parity[arm]=max(float((v.cpu()-cpu[k]).abs().max()) for k,v in gpu.items())
        if parity[arm]>1e-4:raise ValueError('CPU/MPS mismatch '+arm)
    opts={a:torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=0) for a,m in made.items()}
    LIVE.update(models=made,opts=opts,stage='health');r.update(status='health_running',input_audit=audit,cpu_mps_parity=parity);save(r)
    for step in range(1,101):
        if time.monotonic()-started>p['health_wall_s']:raise TimeoutError('Paired health wall')
        losses={}
        for arm,model in made.items():
            model.train();opts[arm].zero_grad(set_to_none=True);total=0
            for x,y in batches[arm]:
                loss,_=canvas_loss(model(x),y,x);(loss/len(batches[arm])).backward();total+=float(loss.detach())/len(batches[arm])
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opts[arm].step()
            r['health_updates'][arm]=step;losses[arm]=total
        r['wall_s']['health']=time.monotonic()-started
        if step==1 or step%25==0:
            r.setdefault('health_curve',[]).append(dict(step=step,loss=losses,wall_s=r['wall_s']['health']));save(r)
            log(dict(stage='health',step=step,losses=losses,wall_s=r['wall_s']['health']))
    raw={};results={}
    for arm,model in made.items():
        checkpoint(model,opts[arm],arm,'health',100);raw[arm]=[];results[arm]=[]
        for row,(x,y) in zip(rows[arm],batches[arm]):
            generated=generate_relative(model,x,row['request'],deadline=started+p['health_wall_s'])
            pred=[RhythmEvent(**e) for e in generated['events']];s=score(pred,row['targets'])
            good=generated['complete'] and (not pred if not row['targets'] else s['full_note_f1']>=.95 and (s['onset_10ms']['f1'] or 0)>=.95 and all(v['f1']>=.9 for v in s['per_type'].values() if v['true']))
            exact=sorted(pred)==sorted(row['targets'])
            results[arm].append(dict(index=row['index'],passed=bool(good),full_note_f1=s['full_note_f1'],exact=exact,predicted=len(pred),target=len(row['targets'])))
            raw[arm].append(dict(index=row['index'],prediction=generated,target=[asdict(e) for e in row['targets']],scores=s))
    write(ART/'health-predictions.json',raw);r['health']=results
    passed=all(x['passed'] for values in results.values() for x in values)
    r.update(status='health_passed' if passed else 'health_failed',complete=not passed);r['wall_s']['health']=time.monotonic()-started;save(r)
    journal(f'真实两臂100健康{r["status"]}，逐请求{results}，墙钟{r["wall_s"]["health"]:.3f}s；健康权重全部丢弃。')
    log(dict(stage=r['status'],results=results))


def train(p,r):
    if r['status']!='health_passed' or any(r['updates'].values()):raise ValueError('Need fresh passed health')
    panel,store=setup(p);made={a:m.to('mps') for a,m in models(p).items()}
    opts={a:torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=0) for a,m in made.items()};LIVE.update(models=made,opts=opts,stage='quality')
    started=time.monotonic();r['status']='training';save(r)
    for step in range(1,901):
        if time.monotonic()-started>p['train_wall_s']:raise TimeoutError('Paired training wall')
        rows=rows_for(panel,store,p['schedule'][step-1]);losses={}
        lr=.001*step/50 if step<=50 else .0001+.0009*(1+math.cos(math.pi*(step-50)/850))/2
        for i in range(2):
            yp=relative_batch([rows['pupu'][i]])[1];ym=relative_batch([rows['mert'][i]])[1]
            if any(not torch.equal(yp[k],ym[k]) for k in yp):raise AssertionError('Paired supervision differs')
        for arm in ARMS:
            model=made[arm];opt=opts[arm];model.train();opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']=lr
            total=0
            for row in rows[arm]:
                x,y=relative_batch([row],'mps');loss,_=canvas_loss(model(x),y,x);(loss/2).backward();total+=float(loss.detach())/2
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step()
            r['updates'][arm]=step;losses[arm]=dict(loss=total,gradient_norm=grad)
        r['wall_s']['training']=time.monotonic()-started
        if step==1 or step%25==0:
            point=dict(step=step,lr=lr,arms=losses,wall_s=r['wall_s']['training']);r['curves'].append(point);save(r);log(dict(stage='train',**point))
        if step in (250,500,900):
            for arm in ARMS:checkpoint(made[arm],opts[arm],arm,'quality',step)
    r.update(status='trained');r['wall_s']['training']=time.monotonic()-started;save(r)
    journal(f'正式新训两臂{r["updates"]}完成，{r["wall_s"]["training"]:.3f}s；真实监督逐批一致，无健康续训/动态回放。')


def evaluate(p,r):
    if r['status']!='trained':raise ValueError('Need completed matched training')
    panel,store=setup(p);made=models(p)
    for arm,model in made.items():
        saved=torch.load(ART/arm/'quality-0900.pt',map_location='cpu',weights_only=False)
        if saved['plan_sha256']!=sha(PLAN) or saved['updates']!=900:raise ValueError('Wrong final checkpoint')
        model.load_state_dict(saved['model']);model.eval()
    started=time.monotonic();deadline=started+p['evaluation_wall_s'];r['status']='evaluating';save(r)
    cases={'old_draft':(p['indices']['val'],'draft',None,False),
        'new_draft':(p['new_val_indices'],'draft',None,False),
        'completion':(p['indices']['val'][::2],'middle',None,False),
        'density_low':(p['indices']['val'][::4],'draft',{'density_notes_per_second':[2.,3.]},False),
        'density_high':(p['indices']['val'][::4],'draft',{'density_notes_per_second':[6.,8.]},False),
        'reference_masked':(p['indices']['val'][::2],'middle',None,True),
        'empty':(p['empty_val_indices'],'draft',None,False)}
    for task,(indices,view,bands,masked) in cases.items():
        outputs={a:[] for a in ARMS}
        for index in indices:
            if time.monotonic()>=deadline:raise TimeoutError('Paired evaluation wall')
            rows=rows_for(panel,store,[dict(index=index,view=view,bands=bands)])
            for arm in ARMS:
                row=rows[arm][0];x,_=relative_batch([row],mask_reference=masked)
                pr=generate_relative(made[arm],x,row['request'],deadline=deadline);pred=[RhythmEvent(**e) for e in pr['events']]
                s=score(pred,row['targets']);heads=[e.onset_us for e in pred];truth=[e.onset_us for e in row['targets']]
                start=row['request'].start_us;width=row['request'].end_us-start
                shifts=[match_f1([start+(t-start+d)%width for t in heads],truth,10000)[3] for d in p['shifts_us']]
                outputs[arm].append(dict(index=index,component=panel.records[index]['component'],chart_id=panel.records[index]['chart_id'],
                    request=asdict(row['request']),prediction=pr,target=[asdict(e) for e in row['targets']],scores=s,shifted_onset10=shifts))
        for arm in ARMS:write(ART/arm/f'{task}-final.json',outputs[arm])
        r['evaluations'][task]={a:dict(full_note_f1=float(np.mean([x['scores']['full_note_f1'] for x in rows])),
            onset10=float(np.mean([x['scores']['onset_10ms']['f1'] or 0 for x in rows])),
            predicted=sum(len(x['prediction']['events']) for x in rows),true=sum(len(x['target']) for x in rows)) for a,rows in outputs.items()}
        save(r);log(dict(stage='evaluate',task=task,result=r['evaluations'][task]))
    r.update(status='evaluated');r['wall_s']['evaluation']=time.monotonic()-started;save(r)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','health','train','evaluate']);args=ap.parse_args()
    if args.stage=='prepare':prepare();return
    p=json.loads(PLAN.read_text());r=json.loads(REPORT.read_text());verify(p);configure()
    def stop(sig,_):raise InterruptedError(f'Signal {sig}')
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:{'health':health,'train':train,'evaluate':evaluate}[args.stage](p,r)
    except BaseException as exc:
        if LIVE:
            updates=r['health_updates'] if LIVE['stage']=='health' else r['updates']
            for arm,model in LIVE['models'].items():checkpoint(model,LIVE['opts'][arm],arm,LIVE['stage']+'-stopped',updates[arm])
        r['status']=args.stage+'_stopped';r['failures'].append(f'{type(exc).__name__}: {exc}');save(r)
        journal(f'{r["status"]}，精确步数health={r["health_updates"]}/quality={r["updates"]}，{exc}。不自动续训。')
        raise


if __name__=='__main__':main()
