"""Frozen100-state completion of full field controls, no training or qualification."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess

import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.train.managed_mps import ManagedMPS
from beatfcn.train.native_cell_chain import cell_batch,decode_cells
from beatfcn.train.native_field_replay_batched import replay_fields
from scripts.run_native_cell_chain import closure,audio_variant
from scripts.run_native_cell_recovery import rows_for_health,zero_audio
from scripts.run_mps_adamw_admission import fresh
from scripts.run_local_evolution import load_data
from scripts.run_local_portfolio_scale import write,sha
from scripts.run_local_typed_canvas import score


ROOT=Path(__file__).resolve().parents[1];ID='c13-batched-field-replay-probe-v1';ART=ROOT/f'artifacts/{ID}';PLAN=ROOT/f'bench/rhythm-v2/{ID}.json'


def prepare():
    if PLAN.exists():raise FileExistsError('Preserve registration')
    cp='artifacts/c12-native-cell-recovery-v1/health-diagnostic-state.pt'
    p=dict(id=ID,parent='bench/rhythm-v2/c12-native-cell-recovery-v1.json',checkpoint=cp,checkpoint_sha256=sha(ROOT/cp),optimizer_updates=0,
        wall_s=600,quality_authorized=False,repair='Only diagnostic replay batching restored to original active-rank shape;no model/loss/decoder/threshold changes',
        controls='All12arms on4frozenhealthrequests;rawfreeoutputs andscores plusidentity;not independentquality or originalC12reclassification',
        source_sha256={f:sha(ROOT/f) for f in closure('scripts/probe_batched_field_replay.py')})
    write(PLAN,p)
    for name in p['source_sha256']:
        dest=ART/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    shutil.copy2(PLAN,ART/'source/frozen-plan.json')


@torch.no_grad()
def worker():
    p=json.loads(PLAN.read_text());parent=json.loads((ROOT/p['parent']).read_text());report=dict(id=ID,status='running',rows=[],optimizer_updates=0,cloud_compute_calls=0)
    if torch.__version__!='2.14.0':raise ValueError('Runtime')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5);guard=ManagedMPS(ROOT,ART,p['wall_s'])
    try:
        if sha(ROOT/p['checkpoint'])!=p['checkpoint_sha256']:raise ValueError('Weight identity')
        model=fresh(parent,'mps');state=torch.load(ROOT/p['checkpoint'],map_location='cpu',weights_only=True);model.load_state_dict(state['model']);del state;model.eval();guard.clear()
        panel,row=load_data()
        for rr in rows_for_health(parent,panel,row)[:4]:
            x,t=cell_batch(rr,'mps');out=model(x);pred=decode_cells(model,x,rr['request'],{k:t[k] for k in ('heads','allowed')},trace=True,output=out,deadline=guard.start+guard.wall_s)
            z=zero_audio(x);s,_=audio_variant(x,x,shift=.733);upstream=dict(zero=(model(z),z),shift=(model(s),s))
            result=replay_fields(model,out,x,rr['request'],{k:t[k] for k in ('heads','allowed')},pred,rr['targets'],upstream,health=False,checkpoint=guard.clear)
            write(ART/f'fields-{rr["index"]}.json',result)
            write(ART/f'prediction-{rr["index"]}.json',dict(index=rr['index'],view=rr['view'],target=[asdict(e) for e in rr['targets']],prediction=pred,scores=score([RhythmEvent(**e) for e in pred['events']],rr['targets'])))
            report['rows'].append(dict(index=rr['index'],members=len(result['tape']),arms=list(result['arms']),certificate=result['certificate'],field_identity_max_diff=result['field_identity_max_diff']))
            write(ART/'report.json',report);print(rr['index'],len(result['arms']),result['field_identity_max_diff'],flush=True)
            del x,t,out,z,s,upstream,pred,result;guard.clear()
        report['status']='completed'
    except BaseException as exc:report.update(status='probe_failed',failure=dict(type=type(exc).__name__,error=str(exc)))
    finally:report['telemetry']=guard.close();write(ART/'report.json',report)


def run():
    if (ART/'report.json').exists():raise FileExistsError('No automatic retry')
    p=json.loads(PLAN.read_text());parent=json.loads((ROOT/p['parent']).read_text())
    if any(sha(ROOT/f)!=v for f,v in p['source_sha256'].items()):raise ValueError('Source drift')
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/parent['runtime_target'])+os.pathsep+str(ROOT)
    with (ART/'run.log').open('w') as log:subprocess.run([str(ROOT/'.venv/bin/python'),'-u','-m','scripts.probe_batched_field_replay','worker'],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=p['wall_s']+30,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','worker']);args=parser.parse_args();globals()[args.action]()
