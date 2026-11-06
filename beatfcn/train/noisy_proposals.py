"""Proposal-only inputs and separate matching/soft-label targets."""
import math
import hashlib
import json
from dataclasses import asdict

import numpy as np
import torch

from beatfcn.model.local_typed_canvas import KINDS
from beatfcn.model.noisy_proposal_refiner import monotone_proposal_matching,REJECT


def proposal_cache_key(example,masked=False):
    digest=hashlib.sha256(json.dumps(dict(index=example['index'],view=example['view'],request=asdict(example['request']),masked=masked),sort_keys=True).encode())
    for key in ('controls','reference','reference_times','reference_valid'):
        value=np.asarray(example['inputs'][key])
        digest.update(str((key,value.shape,str(value.dtype))).encode());digest.update(value.tobytes())
    return digest.hexdigest()


def proposal_inputs(times,request,device='cpu'):
    times=sorted(int(v) for v in times)
    if not times:raise ValueError('Caller handles empty proposal sets without invoking the refiner')
    if len(times)>request.max_events or any(not request.start_us<=v<request.end_us for v in times):raise ValueError('Proposal input domain')
    n=len(times);heads=np.full((n,613),-1,np.int64);allowed=np.zeros((n,613),bool)
    relative=(np.asarray(times,np.int64)-request.start_us)/1e6
    gap=np.diff(np.r_[0,relative]);clock=np.stack((relative/8,np.sin(relative*math.pi),np.cos(relative*math.pi),np.log1p(np.maximum(gap,0))),-1).astype(np.float32)
    anchors=np.full(512,-1,np.int64)
    if len(request.copy_times)>512:raise ValueError('Refinement COPY capacity')
    anchors[:len(request.copy_times)]=request.copy_times
    for i,time in enumerate(times):
        heads[i,:101]=(time//1000+np.arange(-50,51))*1000
        heads[i,101:]=anchors
        allowed[i]=(heads[i]>=request.start_us)&(heads[i]<request.end_us)&(np.abs(heads[i]-time)<=50000)
        allowed[i,101:] &= (anchors%1000!=0)
    length=(request.end_us-request.start_us+19999)//20000
    blocks=(request.start_us+np.arange(200)*40000)//2000000
    blocks=np.minimum(blocks,(request.duration_us+1999999)//2000000-1)
    last_endpoint=max([request.duration_us//1000*1000]+[t for t in request.copy_times if 0<=t<=request.duration_us])
    values=dict(times_s=relative.astype(np.float32),clock=clock,head_times=heads,head_allowed=allowed,
        grid20_valid=np.arange(400)<length,body_blocks=blocks,duration_ms=np.asarray(request.duration_us//1000,np.int64),
        tail_times=anchors,tail_subms=(anchors>=0)&(anchors%1000!=0)&(anchors<=request.duration_us),
        kind_allowed=np.asarray([request.allowed_types is None or k in request.allowed_types for k in KINDS]+[True],bool),
        head_has_endpoint=allowed & (heads<last_endpoint))
    return {k:torch.from_numpy(np.asarray(v))[None].to(device) for k,v in values.items()}


def proposal_targets(times,events,request,*,sigma_ms=1.5,device='cpu'):
    times=sorted(int(t) for t in times)
    events=sorted(events,key=lambda e:(e.onset_us,KINDS.index(e.type),e.end_us))
    inputs=proposal_inputs(times,request)
    n=len(times);target=dict(kind=np.full(n,REJECT,np.int64),head_distribution=np.zeros((n,613),np.float32),
        head_us=np.zeros(n,np.int64),head_ms=np.zeros(n,np.int64),end_ms=np.zeros(n,np.int64),tail_copy=np.zeros(n,np.int64))
    pairs=monotone_proposal_matching(times,[e.onset_us for e in events])
    for i,j in pairs:
        e=events[j];target['kind'][i]=KINDS.index(e.type);target['head_us'][i]=e.onset_us
        target['head_ms'][i]=e.onset_us//1000;target['end_ms'][i]=e.end_us//1000
        if e.onset_us%1000:
            if e.onset_us not in request.copy_times:raise ValueError('Target microsecond onset lacks COPY')
            target['head_distribution'][i,101+request.copy_times.index(e.onset_us)]=1
        else:
            addresses=inputs['head_times'][0,i,:101].numpy()
            mask=inputs['head_has_endpoint'][0,i,:101] if e.type=='hold' else inputs['head_allowed'][0,i,:101]
            probability=np.exp(-.5*((addresses-e.onset_us)/(sigma_ms*1000))**2)*mask.numpy()
            if probability.sum()<=0:raise ValueError('Matched onset outside refinement native support')
            target['head_distribution'][i,:101]=probability/probability.sum()
        if e.type=='hold' and e.end_us%1000:
            if e.end_us not in request.copy_times:raise ValueError('Target microsecond tail lacks COPY')
            target['tail_copy'][i]=request.copy_times.index(e.end_us)+1
    return {k:torch.from_numpy(v)[None].to(device) for k,v in target.items()},dict(matched=len(pairs),proposals=n,targets=len(events),false_positive=n-len(pairs),unproposed=len(events)-len(pairs))


def jittered_training_proposals(events,request,seed):
    rng=np.random.default_rng(seed);times=[]
    for event in events:
        # Legal native proposals, even when a real target uses a COPY anchor.
        time=(event.onset_us//1000+int(rng.integers(-20,21)))*1000
        time=max(request.start_us,min(time,request.end_us-1000));times.append(time)
    count=int(rng.poisson(2*(request.end_us-request.start_us)/1e6))
    width=(request.end_us-request.start_us)//1000
    times.extend(request.start_us+int(v)*1000 for v in rng.integers(max(width,1),size=count))
    return sorted(times)
