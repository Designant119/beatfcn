"""Exact native/COPY multiset transform for the independent10ms lattice."""
import numpy as np
import torch

from beatfcn.model.local_lattice import FRAMES,SLOTS,COPIES
from beatfcn.model.local_typed_canvas import KINDS
from .canonical_probability import probability_batch


def lattice_layout(request):
    n=(request.end_us-request.start_us+9999)//10000
    if n>FRAMES or request.start_us%1000:raise ValueError('Lattice request clock/capacity')
    base=request.start_us+np.arange(FRAMES)*10000;valid=np.arange(FRAMES)<n
    maximum=np.clip((request.end_us-1-base)//1000,0,9)
    copy_us=np.full((FRAMES,COPIES),-1,np.int64);allowed=np.zeros((FRAMES,1+COPIES),bool);allowed[:,0]=True
    for f in range(n):
        copies=sorted({t for t in request.copy_times if base[f]<=t<min(base[f]+10000,request.end_us) and t%1000})
        if len(copies)>COPIES:raise ValueError('Lattice head COPY capacity')
        copy_us[f,:len(copies)]=copies;allowed[f,1:1+len(copies)]=True
    if len(request.copy_times)>512:raise ValueError('Lattice tail COPY capacity')
    tail=np.full(512,-1,np.int64);tail[:len(request.copy_times)]=request.copy_times
    return dict(lattice_valid=valid,lattice_native_max=maximum,lattice_copy_us=copy_us,
        lattice_copy_allowed=allowed,tail_copy_us=tail,duration_us=np.asarray(request.duration_us,np.int64),
        duration_ms=np.asarray(request.duration_us//1000,np.int64),tail_copy_is_subms=(tail>=0)&(tail%1000!=0),
        lattice_kind_allowed=np.asarray([request.allowed_types is None or k in request.allowed_types for k in KINDS],bool))


def lattice_target(events,request):
    layout=lattice_layout(request);shape=(FRAMES,SLOTS)
    target={key:np.zeros(shape,np.int64) for key in ('kind','head_copy','head_us','tail_us','tail_copy')}
    target.update(count=np.zeros(FRAMES,np.int64),active=np.zeros(shape,bool),offset_ms=np.zeros(shape,np.float32))
    for e in sorted(events,key=lambda e:(e.onset_us,e.type,e.end_us)):
        if not request.start_us<=e.onset_us<request.end_us or e.type not in KINDS or e.lane!=-1:raise ValueError('Lattice target domain')
        f=(e.onset_us-request.start_us)//10000;s=int(target['count'][f])
        if s>=SLOTS:raise ValueError('Lattice complete10ms multiset capacity overflow')
        if e.onset_us%1000:
            copies=np.flatnonzero(layout['lattice_copy_us'][f]==e.onset_us)
            if not len(copies):raise ValueError('Unauthorized lattice head COPY')
            target['head_copy'][f,s]=copies[0]+1
        target['count'][f]+=1;target['active'][f,s]=True;target['kind'][f,s]=KINDS.index(e.type)
        target['head_us'][f,s]=e.onset_us;target['tail_us'][f,s]=e.end_us
        target['offset_ms'][f,s]=(e.onset_us-request.start_us-f*10000)/1000
        if e.type=='hold':
            if not e.onset_us<e.end_us<=request.duration_us:raise ValueError('Invalid lattice hold')
            if e.end_us%1000:
                if e.end_us not in request.copy_times:raise ValueError('Unauthorized lattice tail COPY')
                target['tail_copy'][f,s]=request.copy_times.index(e.end_us)+1
        elif e.end_us!=e.onset_us:raise ValueError('Non-hold lattice duration')
    target['head_ms_native']=target['head_us']//1000
    target['tail_ms_native']=target['tail_us']//1000
    return target


def lattice_batch(example,device='mps',*,mask_reference=False):
    inputs,_=probability_batch([example],'cpu',mask_reference=mask_reference)
    inputs.update({k:torch.from_numpy(np.asarray(v))[None] for k,v in lattice_layout(example['request']).items()})
    target={k:torch.from_numpy(v)[None] for k,v in lattice_target(example['targets'],example['request']).items()}
    return {k:v.to(device) for k,v in inputs.items()},{k:v.to(device) for k,v in target.items()}
