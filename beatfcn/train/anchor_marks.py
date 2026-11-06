"""Target-free anchor features and explicitly separate teacher supervision."""
from collections import Counter
import numpy as np
import torch

from beatfcn.model.anchor_mark_chain import KINDS,TAIL_BLOCKS


def nearest(times,valid,queries):
    ids=np.flatnonzero(valid)
    if not len(ids):raise ValueError('No valid acoustic frames')
    t=times[ids];right=np.searchsorted(t,queries).clip(0,len(t)-1);left=(right-1).clip(0,len(t)-1)
    return ids[np.where(abs(queries-t[left])<=abs(t[right]-queries),left,right)]


def anchor_inputs(row,heads_us,*,audio=None,reference=None,mask_reference=False,roll_audio=False):
    request=row['request'];heads=list(map(int,heads_us))
    if heads!=sorted(heads) or any(not request.start_us<=h<request.end_us for h in heads):
        raise ValueError('Caller anchor multiset outside request')
    if any(h%1000 and h not in request.copy_times for h in heads):
        raise ValueError('Unauthorized fractional-ms anchor')
    if len(heads)>request.max_events:raise ValueError('Anchor event capacity')
    source=row['inputs'] if audio is None else audio
    local=np.asarray(source['local']);mel=np.asarray(source['mel'])
    if roll_audio:
        local=local.copy();mel=mel.copy();li=np.flatnonzero(source['local_valid']);mi=np.flatnonzero(source['mel_valid'])
        local[:,li]=np.roll(local[:,li],10,axis=1);mel[mi]=np.roll(mel[mi],40,axis=0)
    query=(np.array(heads,np.int64)-request.start_us)/1e6
    li=nearest(source['local_times'],source['local_valid'],query)
    mi=nearest(source['mel_times'],source['mel_valid'],query)
    nodes=local[:,li,0,:].transpose(1,0,2).copy()
    song=np.asarray(source['song'])[source['song_valid']].mean(0)
    freq=Counter(heads);seen=Counter();timing=[]
    for j,h in enumerate(heads):
        rank=seen[h];seen[h]+=1;m=freq[h]
        previous=(h-heads[j-1])/1e6 if j else 0.
        following=(heads[j+1]-h)/1e6 if j+1<len(heads) else 0.
        timing.append(((h-request.start_us)/max(request.end_us-request.start_us,1),np.log1p(previous),
            np.log1p(following),rank/m,np.log1p(m),np.log1p((request.duration_us-h)/1e6)))
        remaining=request.duration_us//1000-h//1000
        if not 0<=remaining<TAIL_BLOCKS*2000:raise ValueError('Full relative-tail support not representable')
    ref=row['inputs'] if reference is None else reference
    visible=ref['reference_valid'] & (ref['reference'][:,10]>0)
    present=bool(visible.any()) and not mask_reference
    mean=ref['reference'][visible].mean(0) if present else np.zeros(16,np.float32)
    data=dict(local=nodes,mel=mel[mi].copy(),song=song,controls=row['inputs']['controls'].reshape(-1).copy(),
        timing=np.asarray(timing,np.float32).reshape(len(heads),6),reference=mean)
    out={k:torch.from_numpy(np.asarray(v,np.float32).copy())[None] for k,v in data.items()}
    out['reference_present']=torch.tensor([present],dtype=torch.bool)
    out['remaining_ms']=torch.tensor([[request.duration_us//1000-h//1000 for h in heads]],dtype=torch.long)
    if not all(bool(torch.isfinite(v).all()) for v in out.values()):raise ValueError('Nonfinite mark inputs')
    return out


def teacher_example(row):
    events=sorted(row['targets'],key=lambda e:(e.onset_us,KINDS.index(e.type),e.end_us))
    heads=[e.onset_us for e in events]
    x=anchor_inputs(row,heads)
    kinds=[];delta=[];remaining=[]
    for e in events:
        kinds.append(KINDS.index(e.type));base=e.onset_us//1000
        d=e.end_us//1000-base
        if e.type=='hold' and (e.end_us%1000 or not 1<=d<=row['request'].duration_us//1000-base):
            raise ValueError('Teacher hold outside exact native-ms relative support')
        delta.append(d);remaining.append(row['request'].duration_us//1000-base)
    return x,dict(kind=torch.tensor([kinds],dtype=torch.long),delta_ms=torch.tensor(delta,dtype=torch.long),
                  remaining_ms=torch.tensor(remaining,dtype=torch.long)),heads
