"""Three frozen controls separating data expansion from model capacity."""
from collections import Counter, defaultdict
from dataclasses import asdict
from functools import lru_cache
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_typed_canvas import LocalTypedCanvas,canvas_loss
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.sample import match_f1
from beatfcn.train.relative_canvas import relative_batch,generate_relative
from beatfcn.train.local_typed_canvas import encode_canvas,decode_target
from beatfcn.train.local_request_panel import event_request
from scripts.run_frontend_canvas_comparison import setup,rows_for
from scripts.run_local_typed_canvas import score
from scripts.review_frontend_canvas_comparison import metrics


ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'bench/rhythm-v2/local-portfolio-scale-v1.json'
REPORT=ROOT/'reports/local_portfolio_scale_v1.json'
ART=ROOT/'artifacts/local-portfolio-v1/scale'
ARMS=('small24','small64','capacity64')
LIVE={}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def save(r):
    write(REPORT,r)


def journal(message):
    with (ROOT/'EXPERIMENT_LOG.md').open('a') as f:f.write('\n\nlocal-portfolio-scale-v1 '+message+'\n')


def fresh(plan,arm,device='mps'):
    torch.manual_seed(plan['seed'])
    return LocalTypedCanvas(FullEventConfig(**plan['models'][arm]),density_reference=plan['density_reference']).to(device)


def weight_digest(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def prepare():
    if PLAN.exists() or REPORT.exists():raise FileExistsError('Preserve old experiment')
    parent=json.loads((ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text())
    cohort=json.loads((ROOT/'bench/rhythm-v2/local-portfolio-cohort-v1.json').read_text())
    panel,store=setup(parent)
    small=parent['models']['mert'];large={**small,'d':256,'heads':8,'kv_heads':2,'ffn':1024,'encoder_layers':2}
    plan=dict(id='local-portfolio-scale-v1',seed=20261028,steps=2000,health_steps=100,effective_batch=2,
        models=dict(small24=small,small64=small,capacity64=large),
        density_reference=sum(cohort['summary']['train64']['events'].values())/(8*len(cohort['indices']['train64'])),
        indices=cohort['indices'],health_indices=cohort['health_indices'],empty_health_index=cohort['empty_health_index'],
        health_wall_s=600,training_wall_s=3600,evaluation_wall_s=1200,cloud_compute_calls=0,
        learning_rate=.001,minimum_learning_rate=.0001,warmup=50,weight_decay=0,clip=1,
        quality_checkpoints=[1000,2000],diagnostic_checkpoints=[250,500,1500],
        health_gate=dict(positive_minimum_full_f1=.75,positive_mean_full_f1=.9,
            minimum_onset10=.8,positive_count_ratio=[.5,2.0],empty_events=0,finite=True),
        views=['draft','single_condition','all_conditions','random_groups','middle','continuation'],
        strategy=dict(dynamic_generated_prefix_replay=False,reason='Fixed reference-free capacity/data controls retain currentfrontend-v2 recipe. Replay/noisy-context alternatives remain separate mechanisms.',
            count='17-class CE + auxiliary balanced existence + expected_total + true_band_interval',
            offsets='Per-type averaged exact-ms/COPY CE',tail='Cell-base-relative ms endpoints, separate3fieldCE',
            readout='Fixed model-expected-count plus slot-tail-probability rank',
            rare_sampling='One component-uniform window and one rotating rare-type-support window',
            frontend='Frozen nativeMERT L12/L23 plus identical10msmel',
            density='Only positive count exponential tilt; no target count at inference'),
        causal_scope='small24/small64: data-scale recipe under identical model and expanded-train normalization; small64/capacity64: common data/schedule/objective, width+depth capacity recipe. Does not isolate width from depth or prove optimized large-model limits.',
        confirmation_policy='No confirm8 scoring in this run. Kept sealed for portfolio coordinator after final candidate identities frozen.',
        investment_screen=dict(onset10_minus_shift_null=.05,minimum_positive_dev_components=10,
            capacity_full_gain=.02,capacity_onset10_gain=.03,minimum_improved_components=10),
        source_sha256={name:sha(ROOT/name) for name in ['scripts/run_local_portfolio_scale.py',
            'bench/rhythm-v2/local-portfolio-cohort-v1.json','bench/rhythm-v2/local-frontend-comparison-v2.json',
            'beatfcn/model/local_typed_canvas.py','beatfcn/train/relative_canvas.py',
            'beatfcn/train/local_typed_canvas.py','beatfcn/train/local_bundle.py',
            'beatfcn/model/evaluation.py','scripts/review_frontend_canvas_comparison.py']},
        shifts_us=parent['shifts_us'],sample_stream={})
    # Full-target audit, including newly admitted dense windows, before updates.
    audited=0
    for i in set(sum(cohort['indices'].values(),[])):
        req=event_request(panel.records[i]['request']);events=panel.targets(i)
        target=encode_canvas(events,req)
        if decode_target(target,req)!=sorted(events):raise ValueError('Cohort multiset roundtrip failed')
        audited+=1
    plan['exact_target_roundtrips']=audited
    # Verify all referenced cache payloads and native target files once.
    for path,expected in cohort['target_sha256'].items():
        if sha(ROOT/path)!=expected:raise ValueError('Native target identity drift')
    for rec in cohort['mert_cache_identities'].values():
        for name,item in rec['files'].items():
            path=ROOT/rec['path']/name
            if sha(path)!=item['sha256']:raise ValueError('Feature cache identity drift')
    for track in ('train24','train64'):
        pool=cohort['indices'][track];components=defaultdict(list)
        for i in pool:components[panel.records[i]['component']].append(i)
        comps=sorted(components);rare={k:[i for i in pool if panel.records[i]['types'].get(k)] for k in ('hold','flick','drag')}
        rng=np.random.default_rng(plan['seed']);stream=[]
        for step in range(plan['steps']):
            c=comps[int(rng.integers(len(comps)))];first=components[c][int(rng.integers(len(components[c])))]
            pp=rare[('hold','flick','drag')[step%3]] if step%2==0 else pool
            second=pp[int(rng.integers(len(pp)))];stream.append([dict(index=i,view=plan['views'][(2*step+j)%6],field_index=step) for j,i in enumerate((first,second))])
        plan['sample_stream'][track]=stream
    plan['parameters']={}
    for arm in ARMS:
        with torch.device('meta'):model=LocalTypedCanvas(FullEventConfig(**plan['models'][arm]),density_reference=plan['density_reference'])
        plan['parameters'][arm]=sum(p.numel() for p in model.parameters())
    write(PLAN,plan)
    r=dict(id=plan['id'],status='prepared',plan_sha256=sha(PLAN),parameters=plan['parameters'],health_updates=0,
        updates={a:0 for a in ARMS},curves=[],health=[],evaluations={},failures=[],wall_s={},cloud_compute_calls=0,
        exposure={a:dict(windows=0,events=0,indices={},types={}) for a in ARMS})
    save(r);journal(f'冻结：参数{plan["parameters"]}，24/64组件174/416窗；同64流比较容量，不将数据增量归给模型。确认8组件不评分。健康100/600秒，正式2000/3600秒，云0。')
    return parent,plan,r,panel,store


def row_factory(panel,store):
    @lru_cache(maxsize=4)
    def row(index,view='draft',field_index=0,bands=None):
        spec=dict(index=index,view=view,field_index=field_index)
        if bands is not None:spec['bands']={'density_notes_per_second':list(bands)}
        return rows_for(panel,store,[spec])['mert'][0]
    return row


def checkpoint(model,opt,p,arm,stage,step):
    path=ART/arm/f'{stage}-{step:04d}.pt';path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),config=p['models'][arm],
        plan_sha256=sha(PLAN),stage=stage,updates=step,cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state()),path)


def evaluate(models,p,row,tag,deadline,full=False):
    tasks={'draft':[(i,'draft',None,False) for i in p['indices']['dev']],
        'empty':[(i,'draft',None,False) for i in p['indices']['empty_dev']]}
    if full:
        sample=p['indices']['dev'][::2]
        tasks.update(completion=[(i,'middle',None,False) for i in sample],
            reference_masked=[(i,'middle',None,True) for i in sample],
            density_low=[(i,'draft',(2.,3.),False) for i in sample[::4]],
            density_high=[(i,'draft',(6.,8.),False) for i in sample[::4]])
    summary={}
    for arm,model in models.items():
        summary[arm]={}
        for task,specs in tasks.items():
            records=[]
            for i,view,bands,masked in specs:
                if time.monotonic()>deadline:raise TimeoutError('Frozen evaluation wall')
                rr=row(i,view,0,bands);x,_=relative_batch([rr],'mps',mask_reference=masked)
                prediction=generate_relative(model,x,rr['request'],deadline=deadline)
                pred=[RhythmEvent(**e) for e in prediction['events']];scores=score(pred,rr['targets'])
                scores['extra_onset_tolerances']={str(ms):match_f1([e.onset_us for e in pred],[e.onset_us for e in rr['targets']],ms*1000)[3] for ms in (1,5,25)}
                start=rr['request'].start_us;width=rr['request'].end_us-start
                shifts=[match_f1([start+(e.onset_us-start+d)%width for e in pred],[e.onset_us for e in rr['targets']],10000)[3] for d in p['shifts_us']]
                records.append(dict(index=i,component=LIVE['panel'].records[i]['component'],view=view,request=asdict(rr['request']),
                    prediction=prediction,target=[asdict(e) for e in rr['targets']],scores=scores,shifted_onset10=shifts))
                write(ART/arm/f'{tag}-{task}.json',records)
            summary[arm][task]=metrics(records)
    return summary


def health(p,r,row):
    started=time.monotonic();made={a:fresh(p,a) for a in ARMS};opts={a:torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=0) for a,m in made.items()}
    LIVE.update(models=made,opts=opts,stage='health');r['initial_weight_sha256']={a:weight_digest(m) for a,m in made.items()}
    specs=list(zip(p['health_indices'],('draft','single_condition','random_groups','middle')))+[(p['empty_health_index'],'draft')]
    batches=[(row(i,v),relative_batch([row(i,v)],'mps')) for i,v in specs]
    r['status']='health_running';save(r)
    for step in range(1,101):
        if time.monotonic()-started>p['health_wall_s']:raise TimeoutError('Frozen health wall')
        for arm,model in made.items():
            model.train();opts[arm].zero_grad(set_to_none=True)
            for rr,(x,y) in batches:
                loss,_=canvas_loss(model(x),y,x);(loss/len(batches)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opts[arm].step()
        r['health_updates']=step
        if step%25==0:print(json.dumps(dict(stage='health',step=step,wall_s=time.monotonic()-started)),flush=True)
    raw={};passed=True
    for arm,model in made.items():
        values=[]
        for rr,(x,y) in batches:
            pred=generate_relative(model,x,rr['request'],deadline=started+p['health_wall_s']);events=[RhythmEvent(**e) for e in pred['events']];ss=score(events,rr['targets'])
            item=dict(index=rr['index'],prediction=pred,target=[asdict(e) for e in rr['targets']],scores=ss);values.append(item)
            if rr['targets']:
                passed &= bool(pred['complete'] and ss['full_note_f1']>=.75 and (ss['onset_10ms']['f1'] or 0)>=.8 and .5<=len(events)/len(rr['targets'])<=2)
            else:passed &= bool(pred['complete'] and not events)
        positive=[v['scores']['full_note_f1'] for v in values if v['target']];passed &= sum(positive)/len(positive)>=.9
        raw[arm]=values;checkpoint(model,opts[arm],p,arm,'health',100)
    write(ART/'health-predictions.json',raw);r.update(health_passed=bool(passed),status='health_passed' if passed else 'health_failed');r['wall_s']['health']=time.monotonic()-started;save(r)
    del made,opts,batches;LIVE.pop('models',None);LIVE.pop('opts',None);gc.collect();torch.mps.empty_cache()
    if not passed:raise RuntimeError('Pre-registered real-decode health gate failed; no quality training')


def train(p,r,row):
    made={a:fresh(p,a) for a in ARMS};opts={a:torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=0) for a,m in made.items()}
    LIVE.update(models=made,opts=opts,stage='quality');r['quality_initial_weight_sha256']={a:weight_digest(m) for a,m in made.items()}
    if r['quality_initial_weight_sha256']!=r['initial_weight_sha256']:raise ValueError('Health initial/fresh quality mismatch')
    if r['quality_initial_weight_sha256']['small24']!=r['quality_initial_weight_sha256']['small64']:raise ValueError('Small controls must share exact initial weights')
    started=time.monotonic();evaluation_s=0.;r['status']='training';save(r)
    for step in range(1,p['steps']+1):
        if time.monotonic()-started>p['training_wall_s']:raise TimeoutError('Frozen training wall includes evaluations')
        lr=.001*step/50 if step<=50 else .0001+.0009*(1+math.cos(math.pi*(step-50)/(p['steps']-50)))/2
        values={}
        for arm,model in made.items():
            track='train24' if arm=='small24' else 'train64';specs=p['sample_stream'][track][step-1]
            model.train();opt=opts[arm];opt.zero_grad(set_to_none=True)
            for g in opt.param_groups:g['lr']=lr
            total=0.
            for spec in specs:
                rr=row(spec['index'],spec['view'],spec['field_index']);x,y=relative_batch([rr],'mps');loss,_=canvas_loss(model(x),y,x)
                (loss/2).backward();total+=float(loss.detach())/2
                exp=r['exposure'][arm];exp['windows']+=1;exp['events']+=len(rr['targets']);key=str(spec['index']);exp['indices'][key]=exp['indices'].get(key,0)+1
                for e in rr['targets']:exp['types'][e.type]=exp['types'].get(e.type,0)+1
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True));opt.step();r['updates'][arm]=step
            values[arm]=dict(loss=total,preclip_gradient_norm=grad)
        if step==1 or step%25==0:
            point=dict(step=step,learning_rate=lr,arms=values,wall_s=time.monotonic()-started);r['curves'].append(point);r['wall_s']['training']=point['wall_s'];save(r);print(json.dumps(dict(stage='quality',**point)),flush=True)
        if step in p['quality_checkpoints']+p['diagnostic_checkpoints']:
            for arm in ARMS:checkpoint(made[arm],opts[arm],p,arm,'quality',step)
        if step in p['quality_checkpoints']:
            t=time.monotonic();deadline=min(started+p['training_wall_s'],t+p['evaluation_wall_s']-evaluation_s)
            r['evaluations'][str(step)]=evaluate(made,p,row,f'quality-{step:04d}',deadline,full=step==p['steps']);evaluation_s+=time.monotonic()-t;save(r)
    r.update(status='completed',complete=True,confirmation_scored=False);r['wall_s'].update(training=time.monotonic()-started,evaluation=evaluation_s)
    r['final_weight_sha256']={a:sha(ART/a/'quality-2000.pt') for a in ARMS};save(r)
    journal(f'终态：updates={r["updates"]}，health={r["health_passed"]}，确认集未评分，云0；结果{REPORT.relative_to(ROOT)}。')


def main():
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('No silent backend fallback')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.65)
    parent,p,r,panel,store=prepare();LIVE['panel']=panel;row=row_factory(panel,store)
    def stop(*_):raise KeyboardInterrupt('External stop')
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:
        health(p,r,row);train(p,r,row)
    except BaseException as exc:
        r.setdefault('failures',[]).append(dict(stage=LIVE.get('stage'),error=f'{type(exc).__name__}: {exc}',updates=dict(r['updates']),health_updates=r['health_updates']))
        if r['status']!='health_failed':r['status']='stopped'
        for arm,model in LIVE.get('models',{}).items():
            checkpoint(model,LIVE['opts'][arm],p,arm,'partial-'+LIVE['stage'],r['updates'][arm] if LIVE['stage']=='quality' else r['health_updates'])
        save(r);journal(f'停止：{r["failures"][-1]}；保留现场，不自动延长/续训，不改旧计数或金额。');raise


if __name__=='__main__':main()
