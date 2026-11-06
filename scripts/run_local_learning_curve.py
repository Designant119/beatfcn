"""Phira-first, song-disjoint local learning curves; no cloud allocation."""
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from beatfcn.events.canonical import Event,LaneLoc,LineLoc
from beatfcn.events.controls import raw_controls
from beatfcn.model.evaluation import score_events
from beatfcn.model.full_event_codec import EventRequest
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.model.full_event_model import FullEventConfig,FullEventModel,event_loss
from beatfcn.train.local_panel import LocalEventPanel
from scripts.run_local_event_experiment import digest,memory,sync

ROOT=Path(__file__).resolve().parents[1]


def canonical(events,mode):
    return [Event(str(i),e['onset_us'],e['end_us'],e['type'],LaneLoc(e['lane'])
                  if mode=='mania_4k' else LineLoc('rhythm','above',0.)) for i,e in enumerate(events)]


def condition_view(row,rng):
    value=np.asarray(row['controls'],np.float32).copy();known=np.flatnonzero(value[:,2])
    choice=rng.random()
    if choice<.1:keep=[];kind='none'
    elif choice<.6:keep=rng.sample(list(known),1);kind='single'
    elif choice<.8:keep=rng.sample(list(known),min(len(known),rng.randint(2,4)));kind='subset'
    else:keep=known;kind='all'
    hide=[i for i in range(len(value)) if i not in keep];value[hide,:3]=0
    return value,kind


def run(plan_path):
    plan=json.loads(plan_path.read_text());out=ROOT/plan['output']
    if out.exists():raise FileExistsError(out)
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')!='0' or not torch.backends.mps.is_available():
        raise RuntimeError('Native MPS required')
    torch.set_num_threads(4);torch.manual_seed(plan['seed']);torch.mps.manual_seed(plan['seed'])
    torch.mps.set_per_process_memory_fraction(.6)
    panel=LocalEventPanel(ROOT,plan['panel'])
    if digest(ROOT/plan['panel'])!=plan['panel_sha256']:raise ValueError('Panel identity changed')
    for path,sha in panel.manifest['inputs_sha256'].items():
        if digest(ROOT/path)!=sha:raise ValueError('Input identity changed: '+path)
    config=FullEventConfig(**plan['config']);model=FullEventModel(config).to('mps')
    optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
    train=[i for i,r in enumerate(panel.records) if r['split']=='train'];val=[i for i,r in enumerate(panel.records) if r['split']=='val']
    pools={mode:[i for i in train if panel.records[i]['request']['mode']==mode] for mode in ['phira','mania_4k']}
    queues={mode:[] for mode in pools};rng=random.Random(plan['seed']);views=Counter();exposure=Counter()
    started=time.monotonic();train_s=0.;prep_s=0.;eval_s=0.
    report=dict(id=plan['id'],complete=False,updates=0,curve=[],generation=[],quality_winner=None,
        new_cloud_spend_usd=0,parameters=sum(p.numel() for p in model.parameters()),
        valid_tokens=0,requests=0,panel_summary=panel.manifest['per_split'],
        scope='32-request legacy diagnostic panel. Song-disjoint short excerpts, not whole-song or 300-song quality evidence.',
        plan_sha256=digest(plan_path),source_sha256={p:digest(ROOT/p) for p in ['beatfcn/train/local_panel.py',
            'beatfcn/model/full_event_model.py','scripts/run_local_learning_curve.py']},tensor_stats=[])

    def save(phase):
        report.update(phase=phase,wall_s=time.monotonic()-started,train_s=train_s,preparation_s=prep_s,
            evaluation_s=eval_s,memory=memory('mps'),condition_views=dict(views),song_exposure=dict(exposure))
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(out)
        print(json.dumps(dict(id=report['id'],phase=phase,step=report['updates'],wall_s=report['wall_s'])),flush=True)

    def check_deadline():
        if time.monotonic()-started>plan['max_wall_s']:raise TimeoutError('Bounded local learning run deadline')

    def choose(mode):
        if not queues[mode]:queues[mode]=list(pools[mode]);rng.shuffle(queues[mode])
        return queues[mode].pop()

    @torch.no_grad()
    def evaluate(step):
        nonlocal eval_s
        model.eval();t=time.monotonic();scores=[]
        for i in val:
            inp,seq=panel.batch([i],device='mps')
            scores.append(dict(index=i,mode=panel.records[i]['request']['mode'],
                               request_ce=float(event_loss(model(inp,seq),seq))))
        point=dict(step=step,valid_tokens=report['valid_tokens'],requests=report['requests'],
            train_s=train_s,val= scores,mode_val_ce={mode:float(np.mean([r['request_ce'] for r in scores if r['mode']==mode])) for mode in pools})
        # Three Phira and one mania request, fixed before training, independent songs where available.
        selected=plan['generation_indices'];generations=[]
        end=min(started+plan['max_wall_s'],time.monotonic()+plan['generation_budget_s'])
        for offset in range(0,len(selected),2):
            ids=selected[offset:offset+2];inp,_=panel.batch(ids,device='mps')
            requests=[EventRequest(**panel.records[i]['request']) for i in ids]
            results=generate_batch(model,inp,requests,plan['generation_token_cap'],deadline=end)
            for i,pred in zip(ids,results):
                row=panel.records[i];mode=row['request']['mode'];p=canonical(pred['events'],mode);truth=canonical(row['target_events'],mode)
                s=score_events(p,truth,mode)
                counts=s['edit_utility']['full_note'];f1=2*counts['matched']/(counts['pred']+counts['true']) if counts['pred']+counts['true'] else None
                generations.append(dict(index=i,chart_id=row['chart_id'],song_group=row['song_group'],mode=mode,
                    prediction=pred,truth=row['target_events'],scores=s,full_note_f1=f1,
                    predicted_types=dict(Counter(e.type for e in p)),truth_types=dict(Counter(e.type for e in truth)),
                    realized_controls=raw_controls(mode,p,[(row['request']['start_us'],row['request']['end_us'])])))
        for mode in pools:
            rows=[g for g in generations if g['mode']==mode]
            point.setdefault('generation_summary',{})[mode]=dict(requests=len(rows),complete=sum(g['prediction']['complete'] for g in rows),
                full_note_f1=float(np.mean([g['full_note_f1'] for g in rows if g['full_note_f1'] is not None])),
                onset_f1=float(np.mean([g['scores']['onset_20ms']['f1'] for g in rows if g['scores']['onset_20ms']['f1'] is not None])))
        report['curve'].append(point);report['generation'].append(dict(step=step,rows=generations))
        sync('mps');eval_s+=time.monotonic()-t;save('heldout_evaluation');model.train()
        return point

    try:
        save('tensor_audit')
        # Audit every actual row before optimisation. Window LRU is bounded at six.
        for i in range(len(panel.records)):
            check_deadline();r=panel.row(i);item=dict(index=i,tensors={})
            for name,a in r['inputs'].items():
                x=np.asarray(a,dtype=np.float32)
                if not np.isfinite(x).all():raise ValueError('Nonfinite panel input')
                item['tensors'][name]=dict(shape=list(x.shape),minimum=float(x.min()),maximum=float(x.max()),mean=float(x.mean()),std=float(x.std()))
            report['tensor_stats'].append(item)
        prep_s=time.monotonic()-started
        evaluate(0)
        for step in range(1,plan['steps']+1):
            check_deadline();ids=[choose('phira') for _ in range(3)]+[choose('mania_4k')]
            rng.shuffle(ids);optimizer.zero_grad(set_to_none=True);losses=[]
            for begin in (0,2):
                t=time.monotonic();batch_ids=ids[begin:begin+2];controls=[]
                for i in batch_ids:
                    value,kind=condition_view(panel.records[i],rng);controls.append(value);views[kind]+=1
                inp,seq=panel.batch(batch_ids,controls=controls,device='mps');sync('mps');prep_s+=time.monotonic()-t
                t=time.monotonic();loss=event_loss(model(inp,seq),seq)/2;loss.backward();sync('mps');train_s+=time.monotonic()-t
                losses.append(float(loss.detach()));report['valid_tokens']+=sum(int(panel.sequence[i]['valid'].sum()) for i in batch_ids)
                for i in batch_ids:exposure[panel.records[i]['song_group']]+=1
                report['requests']+=len(batch_ids)
                del inp,seq,loss
            t=time.monotonic();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();sync('mps');train_s+=time.monotonic()-t
            report['updates']=step;report['latest_train_ce']=sum(losses);report['gradient_norm']=float(grad)
            if step%25==0:save('training')
            if step in plan['checkpoints']:
                point=evaluate(step)
                if (step==100 and plan.get('stop_if_phira_all_incomplete_at_100')
                        and point['generation_summary']['phira']['complete']==0):
                    report['early_stop']='All three Phira generations incomplete at 100; inspect before extending.'
                    break
        latest=report['generation'][-1]['rows']
        phira=[g for g in latest if g['mode']=='phira']
        report['diagnostic_readiness_passed']=(len(phira)==3
            and all(g['prediction']['complete'] and g['prediction']['events'] for g in phira)
            and all(math.isfinite(v) for v in report['curve'][-1]['mode_val_ce'].values()))
        report['complete']=True
        checkpoint=ROOT/plan['checkpoint'];checkpoint.parent.mkdir(exist_ok=True,parents=True)
        if checkpoint.exists():raise FileExistsError(checkpoint)
        torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=asdict(config),
             optimizer_updates=report['updates'],pilot_only=True,panel_sha256=plan['panel_sha256']),checkpoint)
        report['checkpoint_sha256']=digest(checkpoint);save('complete_exploratory_curve')
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}';report['complete']=True;save('failed');raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);run(p.parse_args().plan.resolve())
