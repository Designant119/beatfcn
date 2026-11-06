"""Conditional event marks on an explicit caller-supplied onset multiset."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .canvas_diffusion import legal_tail_nll, _sample_tail


KINDS=('tap','hold','flick','drag')
TAIL_BLOCKS=601


class AnchorMarkChain(nn.Module):
    def __init__(self,d=128):
        super().__init__()
        self.projection=nn.Linear(2*1024+128+1024+48+6,d)
        self.reference=nn.Linear(16,d)
        self.empty_reference=nn.Parameter(torch.zeros(d))
        self.types=nn.Embedding(5,16)
        self.gru=nn.GRU(d+16,d,batch_first=True)
        self.kind=nn.Linear(d,4)
        self.tail_block=nn.Linear(d,TAIL_BLOCKS)
        self.tail_cell=nn.Linear(d,50)
        self.tail_ms=nn.Linear(d,40)

    def encode(self,x):
        n=x['local'].shape[1]
        parts=(F.layer_norm(x['local'],(1024,)).flatten(-2),
            F.layer_norm(x['mel'],(128,)),F.layer_norm(x['song'],(1024,))[:,None].expand(-1,n,-1),
            x['controls'][:,None].expand(-1,n,-1),x['timing'])
        features=torch.tanh(self.projection(torch.cat(parts,-1)))
        context=torch.where(x['reference_present'][:,None],self.reference(x['reference']),self.empty_reference[None])
        return features,context[None]

    def teacher(self,x,labels):
        features,hidden=self.encode(x)
        if not features.shape[1]:
            return dict(kind=self.kind(features),tail_block=self.tail_block(features),
                        tail_cell=self.tail_cell(features),tail_ms=self.tail_ms(features))
        previous=torch.cat((torch.full_like(labels[:,:1],4),labels[:,:-1]),1)
        h,_=self.gru(torch.cat((features,self.types(previous)),-1),hidden)
        kind=self.kind(h)
        kind=kind.masked_fill((x['remaining_ms']<=0)[...,None] &
            (torch.arange(4,device=kind.device)==1),-1e9)
        return dict(kind=kind,tail_block=self.tail_block(h),tail_cell=self.tail_cell(h),tail_ms=self.tail_ms(h))


def mark_loss(model,x,target):
    out=model.teacher(x,target['kind'])
    n=target['kind'].numel()
    if not n:
        z=sum(v.sum()*0 for v in out.values())
        return z,dict(type_sum=z,tail_sum=z,events=0,holds=0)
    type_sum=F.cross_entropy(out['kind'].flatten(0,1),target['kind'].flatten(),reduction='sum')
    tail_sum=type_sum*0;holds=0
    # Exact integer clock calculations are performed by the CPU data adapter.
    for j in (target['kind'][0]==1).nonzero().flatten().tolist():
        hi=int(target['remaining_ms'][j]);delta=int(target['delta_ms'][j])
        tail_sum=tail_sum+legal_tail_nll(out['tail_block'][0,j],out['tail_cell'][0,j],out['tail_ms'][0,j],1,hi,delta)
        holds+=1
    return (type_sum+tail_sum)/n,dict(type_sum=type_sum,tail_sum=tail_sum,events=n,holds=holds)


@torch.no_grad()
def sampled_marks(model,x,heads_us,duration_us,*,index,view_enum,seeds,allowed_types=None,deadline=None):
    import time
    model.eval();features,hidden=model.encode(x);draws=len(seeds)
    features=features.expand(draws,-1,-1);hidden=hidden.expand(-1,draws,-1).contiguous()
    previous=torch.full((draws,1),4,dtype=torch.long,device=features.device)
    type_rng=[np.random.default_rng(np.random.SeedSequence([s,index,view_enum,1])) for s in seeds]
    tail_rng=[np.random.default_rng(np.random.SeedSequence([s,index,view_enum,2])) for s in seeds]
    result=[[] for _ in seeds]
    for j,head in enumerate(heads_us):
        if deadline is not None and time.monotonic()>deadline:raise TimeoutError('Mark-chain evaluation cap')
        h,hidden=model.gru(torch.cat((features[:,j:j+1],model.types(previous)),-1),hidden)
        logits=model.kind(h[:,0]).cpu().numpy().astype(np.float64)
        if duration_us//1000-head//1000<1:logits[:,1]=-np.inf
        if allowed_types is not None:
            logits[:,[k for k,t in enumerate(KINDS) if t not in allowed_types]]=-np.inf
        p=np.exp(logits-np.max(logits,-1,keepdims=True));p/=p.sum(-1,keepdims=True)
        if not np.isfinite(p).all():raise ValueError('Empty/invalid type support')
        kinds=[int((type_rng[d].random()>=np.cumsum(p[d])).sum()) for d in range(draws)]
        block=model.tail_block(h[:,0]).cpu().numpy().astype(np.float64)
        cell=model.tail_cell(h[:,0]).cpu().numpy().astype(np.float64)
        ms=model.tail_ms(h[:,0]).cpu().numpy().astype(np.float64)
        for d,kind in enumerate(kinds):
            end=head
            if kind==1:
                base=head//1000;hi=duration_us//1000-base
                if not 1<=hi<TAIL_BLOCKS*2000:raise ValueError('Relative tail support overflow')
                end=(base+_sample_tail(block[d],cell[d],ms[d],1,hi,tail_rng[d]))*1000
            result[d].append(dict(onset_us=int(head),end_us=int(end),type=KINDS[kind],lane=-1))
        previous=torch.tensor(kinds,dtype=torch.long,device=features.device)[:,None]
    return result
