"""Discrete self-prefix inputs with full hidden-state BPTT and unchanged mark loss.

Only sampled ids are stopped. This is an on-policy supervised surrogate, not
maximum likelihood of the original human type sequence.
"""
import numpy as np
import torch

from beatfcn.model.anchor_mark_chain import mark_loss


def prefix_outputs(model,inputs,uniforms=None,*,replay_ids=None,trace=False):
    """Return ordinary mark heads; replay_ids is only for numerical controls."""
    features,hidden=model.encode(inputs)
    if features.shape[0]!=1:raise ValueError('One request per prefix microbatch')
    n=features.shape[1]
    if replay_ids is not None:
        if replay_ids.shape!=(1,n):raise ValueError('Prefix replay shape')
    else:
        uniforms=np.asarray(uniforms,dtype=np.float64)
        if uniforms.shape!=(n,) or not np.isfinite(uniforms).all() or (uniforms<0).any() or (uniforms>=1).any():
            raise ValueError('One fixed unit-uniform per event')
    previous=torch.full((1,1),4,dtype=torch.long,device=features.device)
    kind=[];states=[];ids=[];previous_ids=[];embedded=[];hidden_states=[]
    for j in range(n):
        vector=model.types(previous)
        z,hidden=model.gru(torch.cat((features[:,j:j+1],vector),-1),hidden)
        logits=model.kind(z[:,0])
        invalid=(inputs['remaining_ms'][:,j]<=0)[:,None] & (torch.arange(4,device=features.device)==1)
        logits=logits.masked_fill(invalid,-1e9)
        if replay_ids is None:
            raw=logits.detach().cpu().numpy().astype(np.float64)[0]
            p=np.exp(raw-raw.max());p/=p.sum()
            chosen=int((uniforms[j]>=np.cumsum(p)).sum())
            if chosen>=4:raise ValueError('Prefix inverse-CDF overflow')
            current=torch.tensor([[chosen]],dtype=torch.long,device=features.device)
        else:
            current=replay_ids[:,j:j+1].to(features.device)
            if bool(((current<0)|(current>=4)).any()):raise ValueError('Invalid replay type')
        if trace:
            if logits.requires_grad:logits.retain_grad()
            if hidden.requires_grad:hidden.retain_grad()
            previous_ids.append(previous.detach().cpu());embedded.append(vector);hidden_states.append(hidden)
        kind.append(logits);states.append(z[:,0]);ids.append(current.detach().cpu())
        previous=current  # integer; no straight-through estimator, hidden is NOT detached
    h=torch.stack(states,1) if n else features
    out=dict(kind=torch.stack(kind,1) if n else model.kind(features),
        tail_block=model.tail_block(h),tail_cell=model.tail_cell(h),tail_ms=model.tail_ms(h))
    detail=dict(sampled_ids=torch.cat(ids,1) if n else torch.empty((1,0),dtype=torch.long))
    if trace:detail.update(step_logits=kind,previous_ids=previous_ids,previous_embeddings=embedded,hidden_states=hidden_states)
    return out,detail


class _PrefixForward:
    def __init__(self,model,uniforms,replay_ids=None):
        self.model,self.uniforms,self.replay_ids=model,uniforms,replay_ids

    def teacher(self,inputs,unused_current_labels):
        # The existing loss consumes the true labels after forward. Forward does not.
        return prefix_outputs(self.model,inputs,self.uniforms,replay_ids=self.replay_ids)[0]


def self_prefix_loss(model,inputs,target,uniforms):
    return mark_loss(_PrefixForward(model,uniforms),inputs,target)
