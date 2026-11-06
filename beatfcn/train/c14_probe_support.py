"""Shared immutable-state checks for the two independent C14 probes."""
from collections import Counter
from pathlib import Path
import os

import numpy as np
import torch

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.native_cell_chain import NativeCellChain
from beatfcn.train.managed_mps import ManagedMPS
from scripts.run_local_portfolio_scale import sha,weight_digest


ROOT=Path(__file__).resolve().parents[2]


class ProbeGuard(ManagedMPS):
    def __init__(self,root,artifact,wall_s,disk_cap):
        self.disk_cap=disk_cap
        super().__init__(root,artifact,wall_s)

    def check(self,phase=None):
        sample=super().check(phase)
        size=sum(p.stat().st_size for p in self.artifact.rglob('*') if p.is_file())
        if size+16*2**20>self.disk_cap:raise MemoryError('Probe artifact cap including16MiB diagnostic reserve')
        return sample


def probability_difference(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Probability shape/finite identity')
    value=float(np.abs(a-b).max()) if a.size else 0.
    if value>1e-4:raise ValueError(f'Probability identity {value}>1e-4')
    return value


def prediction_identity(old,new):
    if not old['complete'] or not new['complete']:raise ValueError('Identity needs complete baseline')
    event=lambda e:(e['onset_us'],e['end_us'],e['type'],e.get('lane',-1))
    if old['count_decision']!=new['count_decision'] or Counter(map(event,old['events']))!=Counter(map(event,new['events'])):
        raise ValueError('Count/native multiset identity')
    a=old['steps'];b=new['steps']
    if len(a)!=len(b):raise ValueError('Step length identity')
    maximum=0.;tails=0
    for x,y in zip(a,b):
        if (x['cell'],x['rank'],event(x['event']))!=(y['cell'],y['rank'],event(y['event'])):raise ValueError('Member order identity')
        if x['kind_allowed']!=y['kind_allowed']:raise ValueError('Kind mask identity')
        for name in ('head_probability','type_probability'):maximum=max(maximum,probability_difference(x[name],y[name]))
        if len(x['tail'])!=len(y['tail']):raise ValueError('Endpoint layer length')
        for xt,yt in zip(x['tail'],y['tail']):
            if xt['level']!=yt['level'] or xt['labels']!=yt['labels']:raise ValueError('Endpoint support identity')
            maximum=max(maximum,probability_difference(xt['probability'],yt['probability']))
        tails+=bool(x['tail'])
    return dict(max_probability_diff=maximum,members=len(a),hold_paths=tails,integer_multiset_equal=True)


def load_frozen_model(spec):
    import json
    if torch.__version__!='2.14.0' or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('Registered runtime/fallback')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5)
    path=ROOT/spec['parent_plan']
    if sha(path)!=spec['parent_plan_sha256']:raise ValueError('Parent plan identity')
    parent=json.loads(path.read_text())
    for name,h in {**parent['source_sha256'],**parent['runtime_records'],**parent['input_payload_sha256']}.items():
        if sha(ROOT/name)!=h:raise ValueError('Frozen source/cache drift '+name)
    checkpoint=ROOT/spec['checkpoint']
    if sha(checkpoint)!=spec['checkpoint_sha256']:raise ValueError('Frozen quality checkpoint identity')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state['updates']!=1000 or not state['complete'] or state['accumulation_phase']!='applied' or state['plan_sha256']!=sha(path):
        raise ValueError('Complete applied1000 identity')
    model=NativeCellChain(FullEventConfig(**parent['model']),density_reference=parent['density_reference'])
    model.load_state_dict(state['model'],strict=True);del state
    if sum(p.numel() for p in model.parameters())!=10240592:raise ValueError('Model parameter identity')
    model.to('mps').eval()
    return parent,model,weight_digest(model)
