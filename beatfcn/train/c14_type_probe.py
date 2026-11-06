"""Type-only source intervention and exact-rank oracle-prefix diagnostics."""
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
import math
import types

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.local_bundle import read_grid
from beatfcn.train.field_integrity import coverage,coverage_summary,cpu
from beatfcn.train.native_cell_chain import run_members,canonical_head_mask,kind_mask,endpoint_decision,SUPPORT
from beatfcn.train.native_field_replay import machine_tape
from beatfcn.train.native_diagnostic_priors import KINDS,type_probability,duration_probability


@contextmanager
def type_arm(model,output,inputs,arm):
    if arm not in ('B','Z','T'):raise ValueError('Unknown type arm')
    original=model.kind_logits
    def changed(self,out,state,feature,cells):
        song=out['song'].clone()
        if arm=='Z':song[0,cells]=0
        else:
            at=inputs['joint_frame_times'][:,cells]+.02+.733
            song[0,cells]=read_grid(out['song_values'],out['song_times'],out['song_valid'],at,.5)[0]
        logits=original({**out,'song':song},state,feature,cells)
        if not bool(torch.isfinite(logits).all()):raise FloatingPointError('Nonfinite type intervention logits')
        return logits
    if arm!='B':model.kind_logits=types.MethodType(changed,model)
    try:yield
    finally:
        if arm!='B':model.kind_logits=original


def type_source_coverage(output,inputs,arm):
    at=cpu(inputs['joint_frame_times'])[0]+.02+(.733 if arm=='T' else 0.)
    return dict(**coverage_summary(coverage(cpu(output['song_times']),cpu(output['song_valid']),at,.5)),zeroed=arm=='Z',
        scope='Only kind_logits direct-song query range;other song paths remain')


def _item(prior,request,cell,rank,truth,previous,kind_allowed,probability,head_supported,head_nll,endpoint_nll,endpoint_supported):
    k=truth.kind;type_supported=bool(kind_allowed[k]);supported=head_supported and type_supported
    forced=bool(sum(kind_allowed)==1 and supported)
    previous_kind=KINDS.index(previous.type) if previous else -1
    p=float(probability[k]);numerical=supported and (p<=0 or head_nll is None or not math.isfinite(head_nll) or (k==1 and endpoint_supported and (endpoint_nll is None or not math.isfinite(endpoint_nll))))
    global_prior=type_probability(prior,rank,previous_kind,kind_allowed,history=False)
    history_prior=type_probability(prior,rank,previous_kind,kind_allowed)
    duration=None
    if k==1 and head_supported and endpoint_supported:
        minimum=previous.end_us if previous and previous.onset_us==truth.head and previous.type=='hold' else None
        value=duration_probability(prior,truth.head,truth.end_us,request.duration_us,request.copy_times,minimum)
        if value<=0:raise ValueError('Diagnostic prior endpoint support')
        duration=-math.log(value)
    return dict(cell=cell,rank=rank,true_type=k,true_head_us=truth.head,true_end_us=truth.end_us,
        previous=asdict(previous) if previous else None,kind_allowed=list(map(bool,kind_allowed)),type_probability=list(map(float,probability)),
        forced=forced,head_support_failure=not head_supported,type_support_failure=not type_supported,
        endpoint_support_failure=k==1 and not endpoint_supported,numerical_underflow=numerical,
        type_nll=-math.log(p) if supported and p>0 else None,
        type_global_prior_nll=-math.log(global_prior[k]) if supported else None,
        type_history_prior_nll=-math.log(history_prior[k]) if supported else None,
        head_nll=head_nll if head_supported else None,
        endpoint_nll=endpoint_nll if head_supported and endpoint_supported else None,
        endpoint_duration_prior_nll=duration,
        score_semantics='infinite' if not supported or (k==1 and not endpoint_supported) else 'finite' if not numerical else 'probability_underflow',
        auxiliary_type_nll=-math.log(p) if p>0 and type_supported else None)


def summarize_conditionals(items,exact_type_sum=None,*,split=True):
    decisions=[r for r in items if not r['forced']];failures=sum(r['score_semantics']=='infinite' for r in items)
    type_fail=sum(r['head_support_failure'] or r['type_support_failure'] for r in decisions)
    tail=[r for r in items if r['true_type']==1];tail_fail=sum(r['head_support_failure'] or r['endpoint_support_failure'] for r in tail)
    numeric=sum(r['numerical_underflow'] for r in items)
    result=dict(events=len(items),type_decisions=len(decisions),forced=len(items)-len(decisions),support_failures=failures,
        type_support_failures=type_fail,head_support_failures=sum(r['head_support_failure'] for r in items),endpoint_support_failures=tail_fail,numerical_underflow=numeric,
        type_nll_sum=(exact_type_sum if exact_type_sum is not None else sum(r['type_nll'] for r in decisions)) if not type_fail and not numeric else None,
        global_prior_nll_sum=sum(r['type_global_prior_nll'] for r in decisions) if not type_fail else None,
        history_prior_nll_sum=sum(r['type_history_prior_nll'] for r in decisions) if not type_fail else None,
        hold_decisions=len(tail),endpoint_nll_sum=sum(r['endpoint_nll'] for r in tail) if not tail_fail else None,
        duration_prior_nll_sum=sum(r['endpoint_duration_prior_nll'] for r in tail) if not tail_fail else None,
        aggregate_semantics='infinite' if failures else 'numerical_underflow' if numeric else 'finite')
    if split:result['by_rank']={name:summarize_conditionals([r for r in items if (r['rank']==0)==first],split=False) for name,first in [('rank0',True),('rank_ge1',False)]}
    return result


@torch.no_grad()
def tt_diagnostic(model,output,inputs,request,target,prior,deadline):
    _,parts,steps,_=run_members(model,output,inputs,request,target,target['counts'],teacher=True,trace=True,deadline=deadline)
    previous={};items=[]
    from types import SimpleNamespace
    for s in steps:
        c=s['cell'];m=target['members'][c][s['rank']];head=int(target['heads'][c,m.address]);prev=previous.get(c)
        valid=canonical_head_mask(target['heads'][c],target['allowed'][c],prev,request);hp=s['head_probability'][m.address]
        end_nll=0.;end_support=True
        if m.kind==1:
            from beatfcn.train.exact_cell_support import ExactEndpointTree
            minimum=prev.end_us if prev and prev.onset_us==head and prev.type=='hold' else None
            path=ExactEndpointTree(request,head,minimum).path(m.end_us)
            for level,trace in enumerate(s['tail']):
                value=trace['probability'][trace['labels'].index(path[level])]
                if value<=0:raise FloatingPointError('TT endpoint probability underflow')
                end_nll-=math.log(value)
        item=_item(prior,request,c,s['rank'],SimpleNamespace(kind=m.kind,head=head,end_us=m.end_us),prev,s['kind_allowed'],s['type_probability'],bool(valid[m.address]),-math.log(hp) if hp>0 else None,end_nll,end_support)
        items.append(item);previous[c]=RhythmEvent(**s['event'])
    return dict(items=items,summary=summarize_conditionals(items,float(parts['kind'])),exact_factor_nll={k:float(v) for k,v in parts.items()},
        oracle='True count/current head/true canonical history;not free quality')


@torch.no_grad()
def self_diagnostic(model,output,inputs,request,target,prior,deadline,checkpoint=lambda:None):
    events,_,steps,_=run_members(model,output,inputs,request,target,target['counts'],teacher=False,trace=True,deadline=deadline)
    SUPPORT.encode(events,request)
    rollout=dict(complete=True,events=[asdict(e) for e in sorted(events)],count_decision=target['counts'].tolist(),steps=steps)
    tape,certificate=machine_tape(model,output,inputs,request,{k:target[k] for k in ('heads','allowed')},rollout)
    grouped=defaultdict(list)
    for r in tape:grouped[r['rank']].append(r)
    items=[];device=output['x'].device
    from types import SimpleNamespace
    for rank,records in grouped.items():
        cells=torch.tensor([r['cell'] for r in records],device=device);states=torch.tensor([r['state'] for r in records],device=device)
        queries=torch.tensor([r['query'] for r in records],device=device)
        addresses=target['heads'][[r['cell'] for r in records]]
        features=model.address_features(output,inputs,addresses,cells,request.start_us)
        ids=torch.tensor([target['members'][r['cell']][rank].address for r in records],device=device)
        selected=features[torch.arange(len(records),device=device),ids]
        previous=[RhythmEvent(**r['previous']) if r['previous'] else None for r in records]
        kinds=np.stack([kind_mask(request,int(addresses[j,int(ids[j])]),previous[j]) for j in range(len(records))])
        logits=model.kind_logits(output,states,selected,cells).masked_fill(~torch.tensor(kinds,device=device),-1e9)
        probability=logits.softmax(-1).cpu().numpy()
        head_masks=np.stack([canonical_head_mask(addresses[j],target['allowed'][r['cell']],previous[j],request) for j,r in enumerate(records)])
        head_logits=model.head_logits(queries,features).masked_fill(~torch.tensor(head_masks,device=device),-1e9).log_softmax(-1)
        for j,r in enumerate(records):
            c=r['cell'];m=target['members'][c][rank];head=int(addresses[j,m.address]);end_support=True;end_nll=0.
            if m.kind==1:
                try:_,value,_=endpoint_decision(model,output,inputs,states[j],selected[j],request,head,previous[j],target=m.end_us,trace=False);end_nll=float(value)
                except ValueError:end_support=False;end_nll=None
            items.append(_item(prior,request,c,rank,SimpleNamespace(kind=m.kind,head=head,end_us=m.end_us),previous[j],kinds[j],probability[j],bool(head_masks[j,m.address]),float(-head_logits[j,m.address]),end_nll,end_support))
        checkpoint()
    if len(items)!=sum(target['counts']):raise ValueError('SELF fixed-rank denominator')
    return dict(items=items,summary=summarize_conditionals(items),rollout=rollout,certificate=certificate,
        oracle='True count rollout;score current true head/type/end under own preceding events;never insert scored truth into rollout')
