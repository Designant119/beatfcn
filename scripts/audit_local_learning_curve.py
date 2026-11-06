"""Inspect local overfitting and a fixed audio swap without optimizer updates."""
from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.full_event_codec import EventRequest
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.train.local_panel import LocalEventPanel
from scripts.probe_full_event_model import metrics
from scripts.run_local_event_experiment import digest, memory
from scripts.run_local_learning_curve import canonical
from beatfcn.model.evaluation import score_events

ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def main():
    plan_path=ROOT/'bench/rhythm-v2/d65-local-audio-audit-v1.json'
    plan=json.loads(plan_path.read_text());out=ROOT/plan['output']
    if out.exists():raise FileExistsError(out)
    for p,h in plan['inputs_sha256'].items():
        if digest(ROOT/p)!=h:raise ValueError('Audit input changed: '+p)
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')!='0':raise RuntimeError('MPS fallback prohibited')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.6)
    saved=torch.load(ROOT/plan['checkpoint'],map_location='cpu',weights_only=False)
    model=FullEventModel(FullEventConfig(**saved['config']));model.load_state_dict(saved['model']);model=model.to('mps').eval()
    panel=LocalEventPanel(ROOT,plan['panel']);started=time.monotonic()
    report=dict(complete=False,optimizer_updates=0,new_cloud_compute_calls=0,teacher=[],audio_swap=[],
        plan_sha256=digest(plan_path),script_sha256=digest(Path(__file__)),quality_winner=None)
    def save():
        report.update(wall_s=time.monotonic()-started,memory=memory('mps'))
        out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    try:
        for i,row in enumerate(panel.records):
            if time.monotonic()-started>plan['max_wall_s']:raise TimeoutError('Audit deadline')
            conditions=[('specified',None),('unknown',np.asarray(row['controls'],np.float32).copy())] if row['split']=='train' else [('unknown',None)]
            if row['split']=='train':conditions[1][1][:,:3]=0
            for label,control in conditions:
                inp,seq=panel.batch([i],controls=None if control is None else [control],device='mps')
                logits=model(inp,seq);stats=metrics(logits,seq);stats['request_mean_ce']=float(event_loss(logits,seq))
                report['teacher'].append(dict(index=i,split=row['split'],mode=row['request']['mode'],
                    conditions=label,metrics=stats))
            if i%8==0:save()
        ids=plan['indices'];inp,_=panel.batch(ids,device='mps')
        requests=[EventRequest(**panel.records[i]['request']) for i in ids]
        originals=generate_batch(model,inp,requests,plan['token_cap'],deadline=time.monotonic()+plan['generation_max_s'])
        permutation=torch.tensor(plan['donor_positions'],device='mps')
        swapped=dict(inp)
        for key in ('local','local_times','local_valid','mel','mel_times','mel_valid','song','song_times','song_valid'):
            swapped[key]=inp[key].index_select(0,permutation)
        altered=generate_batch(model,swapped,requests,plan['token_cap'],deadline=time.monotonic()+plan['generation_max_s'])
        curve=json.loads((ROOT/plan['curve_report']).read_text())
        previous={g['index']:g['prediction'] for g in curve['generation'][-1]['rows']}
        for j,(i,a,b) in enumerate(zip(ids,originals,altered)):
            row=panel.records[i];truth=canonical(row['target_events'],row['request']['mode'])
            score_a=score_events(canonical(a['events'],'phira'),truth,'phira')
            score_b=score_events(canonical(b['events'],'phira'),truth,'phira')
            ca=Counter(json.dumps(e,sort_keys=True) for e in a['events']);cb=Counter(json.dumps(e,sort_keys=True) for e in b['events'])
            report['audio_swap'].append(dict(index=i,donor_index=ids[plan['donor_positions'][j]],
                baseline=a,swapped=b,baseline_scores=score_a,swapped_scores=score_b,
                reproduced_saved_events=a['events']==previous[i]['events'] and a['complete']==previous[i]['complete'],
                event_multiset_changed=ca!=cb,shared_exact_events=sum((ca&cb).values())))
        report['complete']=True;save()
        print(json.dumps(dict(complete=True,teacher_rows=len(report['teacher']),swaps=len(report['audio_swap']),wall_s=report['wall_s'])))
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}';save();raise


if __name__=='__main__':main()
