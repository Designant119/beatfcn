"""Cell-base-relative endpoints, retaining native integer-ms tails and COPY heads."""
from dataclasses import asdict

import numpy as np
import torch

from beatfcn.model.count_readout import expected_tail_rank
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import encode_events
from beatfcn.model.local_typed_canvas import best_tail,KINDS
from beatfcn.train.local_typed_canvas import canvas_batch


def relative_targets(target,requests):
    target={k:v.clone() for k,v in target.items()}
    holds=target['slot_valid'][:,:,1]
    absolute=target['tail_block']*2000+target['tail_cell']*40+target['tail_ms']
    base=torch.tensor([r.start_us//1000 for r in requests],dtype=torch.int64)[:,None,None]
    base=base+torch.arange(target['count'].shape[1])[None,:,None]*40
    delta=torch.where(holds,absolute-base,0)
    if bool((delta[holds]<1).any()):raise ValueError('Relative endpoint before cell base')
    target['tail_block']=delta//2000;target['tail_cell']=(delta%2000)//40;target['tail_ms']=delta%40
    return target


def relative_batch(rows,device='cpu',*,mask_reference=False):
    x,y=canvas_batch(rows,mask_reference=mask_reference)
    y=relative_targets(y,[r['request'] for r in rows])
    return {k:v.to(device) for k,v in x.items()},{k:v.to(device) for k,v in y.items()}


@torch.no_grad()
def generate_relative(model,inputs,request,*,deadline=None):
    import time
    model.eval();out={k:v.float().cpu().numpy() for k,v in model(inputs).items()}
    valid=inputs['canvas_valid'][0].cpu().numpy();heads=inputs['canvas_head_us'][0].cpu().numpy()
    ids=np.flatnonzero(np.repeat(valid[:,None],4,axis=1).reshape(-1))
    logits=out['count'][0].astype(np.float64)
    prob=np.exp(logits-logits.max(-1,keepdims=True));prob/=prob.sum(-1,keepdims=True)
    events=[];tokens=[];failure=None;detail={}
    try:
        chosen,detail=expected_tail_rank(prob.reshape(-1,17)[ids],event_limit=request.max_events)
        counts=np.zeros(valid.size*4,np.int64);counts[ids]=chosen;counts=counts.reshape(valid.size,4)
        offset=out['offset'][0].argmax(-1)
        for frame,kind in np.argwhere(counts>0):
            base=request.start_us+int(frame)*40000
            for slot in range(int(counts[frame,kind])):
                head=int(heads[frame,offset[frame,kind,slot]])
                end=base+best_tail(out['tail_block'][0,frame,slot],out['tail_cell'][0,frame,slot],
                    out['tail_ms'][0,frame,slot],head-base,request.duration_us-base) if kind==1 else head
                events.append(RhythmEvent(head,end,KINDS[kind]))
        events.sort();tokens=encode_events(events,request)
        if len(tokens)>request.max_tokens:raise ValueError('Relative canvas serialization capacity')
        if deadline is not None and time.monotonic()>deadline:raise TimeoutError('Relative canvas deadline')
    except (ValueError,TimeoutError) as exc:failure=str(exc)
    return dict(complete=failure is None,failure=failure,events=[asdict(e) for e in events],tokens=tokens,count_decision=detail)
