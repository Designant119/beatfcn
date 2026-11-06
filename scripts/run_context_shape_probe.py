"""Serial isolated zero-update runtime comparison; no quality authorization."""
from datetime import datetime,timezone
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np
import torch

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.native_cell_chain import NativeCellChain
from beatfcn.model.static_context_cell_chain import StaticContextCellChain
from beatfcn.train.native_cell_chain import cell_row,cell_batch,cell_loss
from beatfcn.train.process_memory import physical_footprint
from scripts.check_experiment_storage import admission,GIB
from scripts.run_local_evolution import load_data
from scripts.run_local_portfolio_scale import sha,write,weight_digest
from scripts.run_native_cell_chain import closure


ROOT=Path(__file__).resolve().parents[1];ID='c12-context-shape-probe-v1'
PLAN=ROOT/f'bench/rhythm-v2/{ID}.json';ART=ROOT/f'artifacts/{ID}'


def prepare():
    if PLAN.exists():raise FileExistsError('Keep registered probe')
    parent=json.loads((ROOT/'bench/rhythm-v2/c11-native-cell-chain-v1.json').read_text())
    files=closure('scripts/run_context_shape_probe.py')+['tests/test_static_context_cell_chain.py']
    plan=dict(id=ID,parent_plan='bench/rhythm-v2/c11-native-cell-chain-v1.json',parent_sha256=sha(ROOT/'bench/rhythm-v2/c11-native-cell-chain-v1.json'),
        seed=parent['seed'],model=parent['model'],density_reference=parent['density_reference'],
        requests=[s for batch in parent['sample_stream'][:8] for s in batch],
        arms=['dynamic','static_context'],intervention='Only post-SSD cross-attention padding toQ200,K2048 with exact masks;sameweights,law,loss,teacherinputs;noSSMpadding',
        optimizer_updates=0,quality_authorized=False,cloud_compute_calls=0,arm_wall_s=300,total_wall_s=720,
        physical_footprint_cap_bytes=10*GIB,new_disk_peak_bytes=GIB,reserve_bytes=15*GIB,
        compare='Separatecoldprocesses;completeatmost32requestsperarm;first-requestfullgradientandallrequestcountlogits/lossparity1e-4;recordallincomplete/resourcefailures.Noseedretries.',
        source_sha256={name:sha(ROOT/name) for name in sorted(set(files))})
    write(PLAN,plan)
    for name in plan['source_sha256']:
        dest=ART/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    shutil.copy2(PLAN,ART/'source/frozen-plan.json')


def arm(name):
    plan=json.loads(PLAN.read_text());started=time.monotonic();rows=[];failure=None;status='running'
    def stop(*_):raise TimeoutError('Probe arm wall/interrupt')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(plan['arm_wall_s'])
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('No fallback')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5)
    torch.manual_seed(plan['seed']);base=NativeCellChain(FullEventConfig(**plan['model']),density_reference=plan['density_reference'])
    initial=weight_digest(base)
    if name=='static_context':
        model=StaticContextCellChain(FullEventConfig(**plan['model']),density_reference=plan['density_reference']);model.load_state_dict(base.state_dict());del base
    else:model=base
    if weight_digest(model)!=initial:raise ValueError('Matched initialization')
    model=model.to('mps');panel,row=load_data();gc.collect()
    def resource():
        torch.mps.synchronize();memory=physical_footprint()
        if memory['physical_footprint_bytes']>plan['physical_footprint_cap_bytes']:raise MemoryError('Physical footprint hard cap')
        if torch.mps.driver_allocated_memory()>.8*torch.mps.recommended_max_memory():raise MemoryError('Driver cap')
        if admission(ROOT,plan['new_disk_peak_bytes'],plan['reserve_bytes'])['status']!='admitted':raise MemoryError('Disk reserve')
        return dict(**memory,driver_bytes=torch.mps.driver_allocated_memory(),allocated_bytes=torch.mps.current_allocated_memory(),wall_s=time.monotonic()-started)
    try:
        for i,s in enumerate(plan['requests']):
            model.train();model.zero_grad(set_to_none=True);rr=cell_row(panel,row(s['index'],s['view'],s['field_index']));x,t=cell_batch(rr,'mps')
            begin=time.monotonic();output=model(x);loss,_=cell_loss(model,output,x,rr['request'],t);before=resource()
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
            loss.backward();sample=resource();entry=dict(position=i,**s,loss=float(loss.detach()),step_wall_s=time.monotonic()-begin,before_backward=before,**sample)
            rows.append(entry);np.save(ART/f'{name}-count-{i:02d}.npy',output['count'].detach().cpu().numpy())
            if i==0:np.savez_compressed(ART/f'{name}-first-gradients.npz',**{key:p.grad.detach().cpu().numpy() for key,p in model.named_parameters() if p.grad is not None})
            write(ART/f'{name}.json',dict(status='running',initial_weight_sha256=initial,rows=rows,optimizer_updates=0))
            print(name,i+1,round(sample['physical_footprint_bytes']/GIB,3),round(entry['step_wall_s'],3),flush=True)
        status='completed'
    except BaseException as exc:
        status='resource_stopped' if isinstance(exc,(MemoryError,TimeoutError)) else 'failed'
        failure=dict(type=type(exc).__name__,error=str(exc),memory=physical_footprint())
    finally:
        signal.alarm(0);write(ART/f'{name}.json',dict(status=status,initial_weight_sha256=initial,rows=rows,failure=failure,wall_s=time.monotonic()-started,optimizer_updates=0,cloud_compute_calls=0))


def run():
    if (ART/'summary.json').exists():raise FileExistsError('No probe rerun')
    plan=json.loads(PLAN.read_text())
    for source,expected in plan['source_sha256'].items():
        if sha(ROOT/source)!=expected:raise ValueError('Probe source drift '+source)
    if admission(ROOT,GIB,15*GIB)['status']!='admitted':raise ValueError('Disk admission')
    start=time.monotonic()
    for name in plan['arms']:
        if (ART/f'{name}.json').exists():raise FileExistsError('No automatic arm retry')
        remaining=plan['total_wall_s']-(time.monotonic()-start)
        if remaining<plan['arm_wall_s']:raise TimeoutError('No fullarmbudget')
        with (ART/f'{name}.log').open('w') as log:
            subprocess.run([str(ROOT/'.venv/bin/python'),'-u','-m','scripts.run_context_shape_probe','arm',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=plan['arm_wall_s']+30)
    a=json.loads((ART/'dynamic.json').read_text());b=json.loads((ART/'static_context.json').read_text())
    if a['initial_weight_sha256']!=b['initial_weight_sha256']:raise ValueError('Arm state mismatch')
    common=min(len(a['rows']),len(b['rows']));count_diff=max((float(np.abs(np.load(ART/f'dynamic-count-{i:02d}.npy')-np.load(ART/f'static_context-count-{i:02d}.npy')).max()) for i in range(common)),default=None)
    loss_diff=max((abs(a['rows'][i]['loss']-b['rows'][i]['loss']) for i in range(common)),default=None)
    gradient=None
    if common:
        with np.load(ART/'dynamic-first-gradients.npz') as x,np.load(ART/'static_context-first-gradients.npz') as y:
            if set(x.files)!=set(y.files):raise ValueError('Gradient name mismatch')
            gradient=max(float(np.abs(x[k]-y[k]).max()) for k in x.files)
    summary=dict(id=ID,status='completed_bounded_probe',arm_status={n:r['status'] for n,r in [('dynamic',a),('static_context',b)]},
        common_requests=common,max_count_logit_difference=count_diff,max_loss_difference=loss_diff,max_first_gradient_difference=gradient,
        wall_s=time.monotonic()-start,optimizer_updates=0,cloud_compute_calls=0,quality_authorized=False)
    write(ART/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run','arm'));p.add_argument('name',nargs='?');a=p.parse_args()
    {'prepare':prepare,'run':run,'arm':lambda:arm(a.name)}[a.action]()
