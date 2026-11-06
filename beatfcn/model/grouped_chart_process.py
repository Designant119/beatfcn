"""Strict-time grouped point process; complete same-onset multisets.

This is independent of the SSD and diffusion candidates. STOP is legal from
the first decision. Every physical native/COPY address competes with STOP;
type counts are conditioned on a genuinely nonempty group, never a quota.
"""
from itertools import groupby
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .canonical_event_package import CanonicalPackageCanvas
from .local_event_package import _allowed_kinds,_hurdle_count_log_probabilities
from .local_typed_canvas import FRAMES,SLOTS,KINDS


def nonempty_count_law(logp,counts):
    if not bool((counts>0).any()):raise ValueError('Empty onset group is represented by STOP')
    # log(1-exp(log p(allzero))) remains stable near zero.
    zero=logp[:,0].sum()
    logz=torch.log(-torch.expm1(zero))
    return -logp.gather(-1,counts[:,None]).sum()+logz


def decide_nonempty_counts(logp,*,rng=None):
    p=logp.detach().cpu().double().exp().numpy()
    if rng is None:
        count=p.argmax(-1)
        if not count.any():
            gain=np.full((len(p),p.shape[1]-1),-np.inf)
            np.log(p[:,1:],out=gain,where=p[:,1:]>0)
            gain=gain-np.log(p[:,:1])
            kind,positive=np.unravel_index(gain.argmax(),gain.shape)
            count[kind]=positive+1
        return count
    count=np.zeros(len(p),np.int64);seen=False
    for kind,row in enumerate(p):
        weight=row.copy()
        if not seen:weight[0]*=1-np.prod(p[kind+1:,0])
        if weight.sum()<=0:raise ValueError('No nonempty group support')
        count[kind]=int(rng.choice(len(row),p=weight/weight.sum()));seen=seen or count[kind]>0
    if not seen:raise AssertionError('Nonempty conditional count sampler produced empty')
    return count


class GroupedChartProcess(CanonicalPackageCanvas):
    def __init__(self,config,*,density_reference,history_dropout=.1):
        super().__init__(config,density_reference=density_reference)
        d=config.d
        self.history_dropout=history_dropout
        self.history_content=True
        self.chart_gru=nn.GRUCell(d,128)
        self.chart_start=nn.Parameter(torch.randn(128)*.02)
        self.chart_query=nn.Linear(128+d,128)
        self.address_key=nn.Linear(d,128)
        self.physical_phase=nn.Sequential(nn.Linear(4,128),nn.SiLU(),nn.Linear(128,128))
        self.stop_read=nn.Linear(128+d,1)
        self.history_to_group=nn.Linear(128,d)
        self.member_summary=nn.Linear(2*d,d)

    def context(self,inputs,request):
        output=CanonicalPackageCanvas.forward(self,inputs)
        x=output['x'][0]
        allowed=inputs['canvas_head_allowed'][0] & inputs['canvas_valid'][0,:,None]
        pairs=allowed.nonzero();frame,index=pairs.unbind(-1)
        head=inputs['canvas_head_us'][0,frame,index]
        # Request adapter has already removed address aliases, not events.
        order=head.argsort();frame,index,head=frame[order],index[order],head[order]
        if len(head)>1 and bool((head[1:]==head[:-1]).any()):raise ValueError('Duplicated candidate address alias')
        seconds=(head-request.start_us).float()/1e6
        # MPS2.6 integer remainder on large absolute microsecond values may
        # round via float32 (e.g.136001000%1000 becomes992). The native-ms
        # request origin makes this exact, mathematically identical remainder
        # use only the bounded0..8s relative integer range.
        phase=torch.stack((seconds/8,torch.sin(seconds*math.pi),torch.cos(seconds*math.pi),
            ((head-request.start_us)%1000).float()/1000),-1)
        keys=self.address_key(x[frame])+self.physical_phase(phase)
        pooled=(x*inputs['canvas_valid'][0,:,None]).sum(0)/inputs['canvas_valid'].sum().clamp_min(1)
        return dict(x=x,frame=frame,index=index,head=head,keys=keys,pooled=pooled)

    def head_law(self,context,hidden,request,previous_head=None):
        combined=torch.cat((hidden,context['pooled']),-1)
        query=self.chart_query(combined)
        scores=(context['keys']*query).sum(-1)/math.sqrt(len(query))
        allowed=torch.ones_like(context['head'],dtype=torch.bool)
        if previous_head is not None:allowed=allowed & (context['head']>previous_head)
        if _allowed_kinds(request)==(1,):
            last=max([request.duration_us//1000*1000]+[v for v in request.copy_times if v<=request.duration_us])
            allowed=allowed & (context['head']<last)
        scores=scores.masked_fill(~allowed,-torch.inf)
        # Last category is STOP, including at the beginning and after max head.
        return torch.cat((scores,self.stop_read(combined).reshape(1))).log_softmax(-1)

    def group_law(self,context,hidden,candidate,inputs,request):
        frame=int(context['frame'][candidate]);head=int(context['head'][candidate])
        x=context['x'][frame]+self.history_to_group(hidden)
        gate=self.count_gate(x);positive=self.count_positive(x).view(4,SLOTS)
        logp=_hurdle_count_log_probabilities(gate,positive)
        density=inputs['controls'][0,0];rho=density[:2].mean()*64
        tilt=torch.where(density[2]>.5,F.softplus(self.density_alpha)*(rho.clamp_min(1e-4)/self.density_reference).log(),0.)
        logp=logp+tilt*torch.arange(17,device=x.device);logp=logp-logp.logsumexp(-1,keepdim=True)
        allowed=list(_allowed_kinds(request))
        last=max([request.duration_us//1000*1000]+[v for v in request.copy_times if v<=request.duration_us])
        if head>=last and 1 in allowed:allowed.remove(1)
        kind_mask=torch.tensor([k in allowed for k in range(4)],device=x.device)
        zero=torch.full_like(logp,-torch.inf);zero[:,0]=0
        return torch.where(kind_mask[:,None],logp,zero),x

    def emit_group(self,context,hidden,candidate,counts,inputs,request,*,events=None,rng=None):
        frame=int(context['frame'][candidate]);index=int(context['index'][candidate]);head=int(context['head'][candidate])
        logp,feature=self.group_law(context,hidden,candidate,inputs,request)
        # Single physical head is already chosen by the global process.
        single=torch.arange(inputs['canvas_head_allowed'].shape[-1],device=feature.device)==index
        selected=torch.arange(FRAMES,device=feature.device)==frame
        addressed={**inputs,'canvas_head_allowed':torch.where(selected[None,:,None],single[None,None],inputs['canvas_head_allowed'])}
        x=context['x'].clone();x[frame]=feature
        result=[];tail_nll=x.sum()*0
        for kind,n in enumerate(counts):
            n=int(n)
            if not n:continue
            teacher=None
            if events is not None:
                target=sorted((e for e in events if e.type==KINDS[kind]),key=lambda e:e.end_us)
                teacher=dict(offset=torch.full((n,),index,device=x.device,dtype=torch.long),
                    tail_us=torch.tensor([e.end_us for e in target],device=x.device))
            members,_,tail=self.ordered_members(x,frame,kind,n,addressed,request,teacher=teacher,
                rng=rng if events is None else None,greedy=events is None and rng is None)
            result.extend(members);tail_nll=tail_nll+tail
        # Complete emitted multiset, including tail information, updates history.
        summaries=[]
        for e in result:
            kind=KINDS.index(e.type)
            endpoint=(e.end_us-head)/1e6
            temporal=self.clock_read(x.new_tensor([endpoint/1200,math.log1p(max(endpoint,0)),math.sin(endpoint*math.pi),math.cos(endpoint*math.pi)]))
            summaries.append(self.member_summary(torch.cat((self.kind_ids.weight[kind]+self.onset_ids.weight[index],temporal),-1)))
        count_values=torch.as_tensor(counts,device=feature.device,dtype=torch.long)
        summary=torch.stack(summaries).mean(0)+self.count_ids(count_values).sum(0)
        if not self.history_content:summary=summary*0
        if self.training and self.history_dropout:
            summary=F.dropout(summary,p=self.history_dropout,training=True)
        hidden=self.chart_gru(summary[None],hidden[None])[0]
        return sorted(result),hidden,tail_nll,logp


def grouped_process_loss(model,inputs,events,request):
    context=model.context(inputs,request);hidden=model.chart_start;previous=None
    onset_sum=count_sum=tail_sum=hidden.sum()*0
    for head,iterator in groupby(sorted(events,key=lambda e:e.onset_us),key=lambda e:e.onset_us):
        members=list(iterator);candidate=(context['head']==head).nonzero().flatten()
        if len(candidate)!=1:raise ValueError('Target group lacks unique authorized physical address')
        candidate=int(candidate[0]);head_logp=model.head_law(context,hidden,request,previous)
        onset_sum=onset_sum-head_logp[candidate]
        counts=torch.tensor([sum(e.type==k for e in members) for k in KINDS],device=hidden.device)
        if bool((counts>SLOTS).any()):raise ValueError('Same-onset type capacity overflow')
        _,hidden,tail,logp=model.emit_group(context,hidden,candidate,counts,inputs,request,events=members)
        count_sum=count_sum+nonempty_count_law(logp,counts);tail_sum=tail_sum+tail;previous=head
    onset_sum=onset_sum-model.head_law(context,hidden,request,previous)[-1]
    denominator=inputs['canvas_valid'].sum().clamp_min(1).float()*len(_allowed_kinds(request))
    parts={k:v/denominator for k,v in [('group_head_stop',onset_sum),('counts',count_sum),('tail',tail_sum)]}
    return sum(parts.values()),parts


@torch.no_grad()
def generate_grouped_process(model,inputs,request,*,seed=20261055,greedy=False,deadline=None):
    model.eval();context=model.context(inputs,request);hidden=model.chart_start
    rng=np.random.default_rng(seed);previous=None;events=[];trace=[]
    for step in range(request.max_events+1):
        if deadline is not None and time.monotonic()>deadline:raise TimeoutError('Grouped process deadline')
        logp=model.head_law(context,hidden,request,previous)
        p=logp.cpu().double().exp().numpy()
        candidate=int(p.argmax()) if greedy else int(rng.choice(len(p),p=p/p.sum()))
        if candidate==len(p)-1:return sorted(events),dict(complete=True,groups=trace,stop_step=step)
        law,_=model.group_law(context,hidden,candidate,inputs,request)
        counts=decide_nonempty_counts(law,rng=None if greedy else rng)
        if len(events)+int(counts.sum())>request.max_events:raise ValueError('Grouped process event resource overflow')
        members,hidden,_,_=model.emit_group(context,hidden,candidate,counts,inputs,request,rng=None if greedy else rng)
        events.extend(members);previous=int(context['head'][candidate]);trace.append(dict(head_us=previous,counts=counts.tolist()))
    raise ValueError('Grouped process reached event budget without STOP')
