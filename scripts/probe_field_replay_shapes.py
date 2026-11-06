"""Frozen-health, zero-update comparison of single vs original grouped replay."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch

from beatfcn.train.managed_mps import ManagedMPS
from beatfcn.train.native_cell_chain import cell_batch,decode_cells
from beatfcn.train.native_field_replay import machine_tape,controlled_heads,array
from scripts.run_local_evolution import load_data
from scripts.run_local_portfolio_scale import write,sha
from scripts.run_mps_adamw_admission import fresh
from scripts.run_native_cell_chain import closure
from scripts.run_native_cell_recovery import rows_for_health


ROOT=Path(__file__).resolve().parents[1];ID='c13-field-replay-shape-probe-v1'
PLAN=ROOT/f'bench/rhythm-v2/{ID}.json';ART=ROOT/f'artifacts/{ID}'


def prepare():
    if PLAN.exists():raise FileExistsError('Registered probe exists')
    parent=json.loads((ROOT/'bench/rhythm-v2/c12-native-cell-recovery-v1.json').read_text())
    checkpoint='artifacts/c12-native-cell-recovery-v1/health-diagnostic-state.pt'
    p=dict(id=ID,parent='bench/rhythm-v2/c12-native-cell-recovery-v1.json',parent_sha256=sha(ROOT/'bench/rhythm-v2/c12-native-cell-recovery-v1.json'),
        checkpoint=checkpoint,checkpoint_sha256=sha(ROOT/checkpoint),modes=['single','batched','single_cached_feature'],
        optimizer_updates=0,quality_authorized=False,wall_s=600,
        hypothesis='Field BASE mismatch may be execution batch shape rather than changed machine inputs;hold weights/state/head/history/masks fixed',
        gate='Report head/type probability and exactinteger identity for all4healthwindows;batched <=1e-4 supports exactshape repair only,not health orquality',
        source_sha256={f:sha(ROOT/f) for f in closure('scripts/probe_field_replay_shapes.py')})
    write(PLAN,p)
    for name in p['source_sha256']:
        dest=ART/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    shutil.copy2(PLAN,ART/'source/frozen-plan.json')


@torch.no_grad()
def worker():
    p=json.loads(PLAN.read_text());parent=json.loads((ROOT/p['parent']).read_text());report=dict(id=ID,status='running',rows=[],optimizer_updates=0,cloud_compute_calls=0)
    if torch.__version__!='2.14.0':raise ValueError('Runtime identity')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5)
    guard=ManagedMPS(ROOT,ART,p['wall_s'])
    try:
        if sha(ROOT/p['checkpoint'])!=p['checkpoint_sha256']:raise ValueError('Checkpoint identity')
        model=fresh(parent,'mps');state=torch.load(ROOT/p['checkpoint'],map_location='cpu',weights_only=True)
        model.load_state_dict(state['model']);model.eval();del state;guard.clear()
        panel,row=load_data()
        for rr in rows_for_health(parent,panel,row)[:4]:
            x,t=cell_batch(rr,'mps');out=model(x);layout={k:t[k] for k in ('heads','allowed')}
            pred=decode_cells(model,x,rr['request'],layout,trace=True,output=out,deadline=guard.start+guard.wall_s)
            tape,cert=machine_tape(model,out,x,rr['request'],layout,pred);groups=defaultdict(list)
            for r in tape:groups[r['rank']].append(r)
            stats={name:dict(head_probability_max_diff=0.,type_probability_max_diff=0.,selected_feature_max_diff=0.,head_mismatch=0,type_mismatch=0) for name in p['modes']}
            bad=[]
            for rank,rows in groups.items():
                cells=torch.tensor([r['cell'] for r in rows],device='mps');states=torch.tensor([r['state'] for r in rows],device='mps');queries=torch.tensor([r['query'] for r in rows],device='mps')
                heads=layout['heads'][[r['cell'] for r in rows]]
                grouped,_=controlled_heads(model,out,x,heads,cells,rr['request'].start_us)
                ids=torch.tensor([int(np.flatnonzero(layout['allowed'][r['cell']]&(layout['heads'][r['cell']]==r['event']['onset_us']))[0]) for r in rows],device='mps')
                selected=grouped[torch.arange(len(rows),device='mps'),ids]
                head_mask=torch.tensor([r['head_allowed'] for r in rows],device='mps');type_mask=torch.tensor([r['kind_allowed'] for r in rows],device='mps')
                hp=model.head_logits(queries,grouped).masked_fill(~head_mask,-1e9).softmax(-1)
                kp=model.kind_logits(out,states,selected,cells).masked_fill(~type_mask,-1e9).softmax(-1)
                for i,r in enumerate(rows):
                    for mode in p['modes']:
                        if mode=='batched':features=grouped[i:i+1];feature=selected[i];hprob=hp[i];kprob=kp[i]
                        else:
                            if mode=='single':features,_=controlled_heads(model,out,x,heads[i:i+1],cells[i:i+1],rr['request'].start_us);feature=features[0,ids[i]]
                            else:features=grouped[i:i+1];feature=torch.tensor(r['head_feature'],device='mps')
                            hprob=model.head_logits(queries[i:i+1],features)[0].masked_fill(~head_mask[i],-1e9).softmax(-1)
                            kprob=model.kind_logits(out,states[i:i+1],feature[None],cells[i:i+1])[0].masked_fill(~type_mask[i],-1e9).softmax(-1)
                        hdiff=float(np.abs(array(hprob)-r['head_probability']).max());kdiff=float(np.abs(array(kprob)-r['type_probability']).max())
                        s=stats[mode];s['head_probability_max_diff']=max(s['head_probability_max_diff'],hdiff);s['type_probability_max_diff']=max(s['type_probability_max_diff'],kdiff)
                        s['selected_feature_max_diff']=max(s['selected_feature_max_diff'],float(np.abs(array(feature)-r['head_feature']).max()))
                        s['head_mismatch']+=int(heads[i,int(hprob.argmax())]!=r['event']['onset_us'])
                        s['type_mismatch']+=int(('tap','hold','flick','drag')[int(kprob.argmax())]!=r['event']['type'])
                        if max(hdiff,kdiff)>1e-4:bad.append(dict(mode=mode,cell=r['cell'],rank=rank,head_diff=hdiff,type_diff=kdiff))
                guard.clear()
            report['rows'].append(dict(index=rr['index'],view=rr['view'],members=len(tape),baseline_complete=pred['complete'],certificate=cert,modes=stats,mismatches_over_tolerance=bad))
            write(ART/'report.json',report);print(rr['index'],stats,flush=True)
            del x,t,out,pred,tape,groups;guard.clear()
        report['status']='completed'
    except BaseException as exc:report.update(status='probe_failed',failure=dict(type=type(exc).__name__,error=str(exc)))
    finally:
        report['telemetry']=guard.close();write(ART/'report.json',report)


def run():
    if (ART/'report.json').exists():raise FileExistsError('No automatic retry')
    p=json.loads(PLAN.read_text());parent=json.loads((ROOT/p['parent']).read_text())
    if any(sha(ROOT/f)!=v for f,v in p['source_sha256'].items()):raise ValueError('Source drift')
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/parent['runtime_target'])+os.pathsep+str(ROOT)
    with (ART/'run.log').open('w') as log:subprocess.run([str(ROOT/'.venv/bin/python'),'-u','-m','scripts.probe_field_replay_shapes','worker'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=p['wall_s']+30,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','worker']);args=parser.parse_args();globals()[args.action]()
