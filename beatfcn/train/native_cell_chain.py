"""Natural count/member NLL and the same canonical conditional decode path."""
from dataclasses import asdict
import time

import numpy as np
import torch
from torch.nn import functional as F

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import encode_events
from beatfcn.train.canonical_probability import explicit_row
from beatfcn.train.exact_cell_support import ExactCellSupport,ExactEndpointTree
from beatfcn.train.joint_native_set import adapted_row
from beatfcn.train.local_bundle import batch_rows
from beatfcn.train.native_joint_support import KINDS


SUPPORT=ExactCellSupport()


def cell_row(panel,row,*,mask_reference=False):
    # Complete known intervals in the original six views lie outside the new
    # request core; random removal is positive-only. Caller-supplied intervals
    # can additionally remove addresses inside a core, without hidden labels.
    adapted=adapted_row(panel,row,mask_reference=mask_reference)
    return {**adapted,'known_regions':tuple(row.get('known_regions',()))}


def cell_batch(row,device='cpu'):
    inputs,_=batch_rows([explicit_row(row)]);r=row['request'];heads,allowed=SUPPORT.addresses(r)
    for lo,hi in row.get('known_regions',()):allowed&=~((heads>=lo)&(heads<hi))
    counts,members=SUPPORT.encode(row['targets'],r)
    by_cell=[[] for _ in counts]
    for m in members:
        if not allowed[m.cell,m.address]:raise ValueError('Target enters caller-protected known region')
        by_cell[m.cell].append(m)
    inputs.update(joint_frame_times=torch.arange(len(counts),dtype=torch.float32)[None]*.04,
        cell_has_address=torch.tensor(allowed.any(-1))[None])
    target=dict(counts=counts,members=by_cell,heads=heads,allowed=allowed)
    return {k:v.to(device) for k,v in inputs.items()},target


def previous_vectors(model,events,origin_us,inputs):
    device=inputs['mode'].device
    heads=np.array([e.onset_us if e else origin_us for e in events],np.int64)
    ends=np.array([e.end_us if e else origin_us for e in events],np.int64)
    ht=torch.tensor((heads-origin_us)/1e6,dtype=torch.float32,device=device)
    et=torch.tensor((ends-origin_us)/1e6,dtype=torch.float32,device=device)
    ids=torch.tensor([KINDS.index(e.type) if e else 0 for e in events],dtype=torch.long,device=device)
    kinds=F.one_hot(ids,4).float()
    bos=torch.tensor([e is None for e in events],dtype=torch.bool,device=device)
    hi=inputs['mel_times'][0].masked_fill(~inputs['mel_valid'][0],-1e9).max()+.005
    coverage=((et>=inputs['mel_times'][0,0]-.005)&(et<=hi)&~bos).float()
    features=torch.cat((model.time_features(ht),kinds,model.time_features(et),coverage[:,None],bos.float()[:,None]),-1)
    # BOS does not pretend a zero-second event was observed.
    return torch.where(bos[:,None],F.pad(bos.float()[:,None],(71,0)),features)


def endpoint_decision(model,output,inputs,state,head_feature,request,head,previous,*,target=None,trace=False):
    minimum=previous.end_us if previous is not None and previous.onset_us==head and previous.type=='hold' else None
    tree=ExactEndpointTree(request,head,minimum)
    if not tree.blocks():raise ValueError('Hold has no legal endpoint')
    device=state.device;onehot=state.new_tensor([0.,1.,0.,0.])
    query=model.tail_query(torch.cat((state,head_feature,onehot),-1))
    true_path=tree.path(target) if target is not None else None
    selected=[];loss=query.sum()*0;diagnostic=[]
    for level in range(3):
        if level==0:
            labels=tree.blocks();times=[min(tree.hi,max(tree.lo,tree.base+b*2000000+1000000)) for b in labels]
        elif level==1:
            labels=tree.cells(selected[0]);times=[min(tree.hi,max(tree.lo,tree.base+selected[0]*2000000+c*40000+20000)) for c in labels]
        else:
            times=list(tree.leaves(selected[0],selected[1]));labels=tuple(range(len(times)))
        keys=model.endpoint_keys(output,inputs,times,request.start_us)
        logits=model.endpoint_scores(query,keys,level)
        choice=labels.index(true_path[level]) if true_path is not None else int(logits.argmax())
        if true_path is not None:loss=loss-logits.log_softmax(0)[choice]
        selected.append(int(labels[choice]))
        if trace:diagnostic.append(dict(level=level,labels=list(labels),probability=logits.softmax(0).detach().cpu().tolist()))
        if level<2:query=model.endpoint_condition(keys[choice][None],query[None])[0]
    endpoint=int(times[selected[-1]])
    return endpoint,loss,diagnostic


def canonical_head_mask(heads,allowed,previous,request):
    valid=allowed.copy()
    if previous is not None:valid&=heads>=previous.onset_us
    if request.allowed_types==('hold',):
        # A final fractional head may lack any legal native/COPY end.
        possible=((heads//1000+1)*1000<=request.duration_us)
        for u in request.copy_times:possible|=(heads<u)&(u<=request.duration_us)
        valid&=possible
    return valid


def kind_mask(request,head,previous):
    valid=np.array([request.allowed_types is None or k in request.allowed_types for k in KINDS],bool)
    if previous is not None and previous.onset_us==head:valid[:KINDS.index(previous.type)]=False
    if valid[1] and not ExactEndpointTree(request,head).blocks():valid[1]=False
    return valid


def run_members(model,output,inputs,request,target,counts,*,teacher,trace=False,deadline=None,teacher_prefix=False):
    device=output['count'].device;active=np.where(counts>0)[0]
    total=int(counts.sum())
    if (counts>64).any() or total>min(256,request.max_events):raise ValueError('Predicted count OVERFLOW')
    zero=output['count'].sum()*0
    sums=dict(head=zero,kind=zero,tail=zero);steps=[];events=[];hold_count=0
    if not len(active):return events,sums,steps,hold_count
    cells=torch.tensor(active,dtype=torch.long,device=device)
    number=torch.tensor(counts[active],dtype=torch.long,device=device)
    state,embedding=model.initial_members(output,cells,number)
    previous=[None]*len(active)
    for rank in range(int(counts.max())):
        if deadline is not None and time.monotonic()>deadline:raise TimeoutError('Cell member deadline')
        state,query=model.member_step(output,cells,embedding,state,previous_vectors(model,previous,request.start_us,inputs))
        address_values=target['heads'][active]
        features=model.address_features(output,inputs,address_values,cells,request.start_us)
        valid=np.stack([canonical_head_mask(target['heads'][c],target['allowed'][c],p,request) for c,p in zip(active,previous)])
        logits=model.head_logits(query,features).masked_fill(~torch.tensor(valid,device=device),-1e9)
        if not valid.any(-1).all():raise ValueError('Canonical member has no legal head')
        if teacher:chosen=np.array([target['members'][c][rank].address for c in active],np.int64)
        else:chosen=logits.argmax(-1).cpu().numpy()
        ids=torch.tensor(chosen,dtype=torch.long,device=device);positions=torch.arange(len(active),device=device)
        selected_features=features[positions,ids]
        chosen_heads=address_values[np.arange(len(active)),chosen]
        if teacher:sums['head']=sums['head']+F.cross_entropy(logits,ids,reduction='sum')
        allowed_kinds=np.stack([kind_mask(request,int(h),p) for h,p in zip(chosen_heads,previous)])
        kind_logits=model.kind_logits(output,state,selected_features,cells).masked_fill(~torch.tensor(allowed_kinds,device=device),-1e9)
        if teacher:kinds=np.array([target['members'][c][rank].kind for c in active],np.int64)
        else:kinds=kind_logits.argmax(-1).cpu().numpy()
        if teacher:sums['kind']=sums['kind']+F.cross_entropy(kind_logits,torch.tensor(kinds,device=device),reduction='sum')
        current=[]
        for j,(cell,head,kind) in enumerate(zip(active,chosen_heads,kinds)):
            head,kind=int(head),int(kind);end=head;tail_trace=[]
            if kind==1:
                true_end=target['members'][cell][rank].end_us if teacher else None
                end,nll,tail_trace=endpoint_decision(model,output,inputs,state[j],selected_features[j],request,head,previous[j],target=true_end,trace=trace)
                sums['tail']=sums['tail']+nll;hold_count+=1
            event=RhythmEvent(head,int(end),KINDS[kind]);current.append(event);events.append(event)
            if trace:
                steps.append(dict(cell=int(cell),rank=rank,event=asdict(event),head_probability=logits[j].softmax(-1).detach().cpu().tolist(),
                    type_probability=kind_logits[j].softmax(-1).detach().cpu().tolist(),kind_allowed=allowed_kinds[j].tolist(),tail=tail_trace))
        keep=np.where(counts[active]>rank+1)[0]
        if not len(keep):break
        previous_next=[]
        for j in keep:
            cell=active[j]
            if teacher_prefix and rank<len(target['members'][cell]):
                m=target['members'][cell][rank]
                previous_next.append(RhythmEvent(int(target['heads'][cell,m.address]),m.end_us,KINDS[m.kind]))
            else:previous_next.append(current[j])
        active=active[keep];cells=cells[keep];state=state[keep];embedding=embedding[keep];previous=previous_next
    return events,sums,steps,hold_count


def cell_loss(model,output,inputs,request,target):
    count=torch.tensor(target['counts'],dtype=torch.long,device=output['count'].device)
    count_loss=F.cross_entropy(output['count'][0],count,reduction='sum')
    _,parts,_,holds=run_members(model,output,inputs,request,target,target['counts'],teacher=True)
    denominator=len(count)
    total=(count_loss+parts['head']+parts['kind']+parts['tail'])/denominator
    return total,dict(count=count_loss/denominator,head=parts['head']/denominator,kind=parts['kind']/denominator,
        tail=parts['tail']/denominator,events=int(count.sum()),holds=holds)


@torch.no_grad()
def decode_cells(model,inputs,request,layout,*,deadline=None,trace=False,output=None):
    if set(layout)!={'heads','allowed'}:raise ValueError('Decode accepts request-only support,not labels')
    model.eval();output=model(inputs) if output is None else output
    counts=output['count'][0].argmax(-1).cpu().numpy();events=[];steps=[];failure=None
    try:
        events,_,steps,_=run_members(model,output,inputs,request,layout,counts,teacher=False,trace=trace,deadline=deadline)
        encoded=SUPPORT.encode(events,request)
        if 1+3*len(events)>request.max_tokens:raise ValueError('Exact field serialization overflow')
        # Native subdomain can additionally prove compatibility with the old
        # grammar; extended COPY tails have a versioned exact integer field tape.
        legacy=encode_events(events,request) if all(e.type!='hold' or not e.end_us%1000 for e in events) else None
        if legacy is not None and len(legacy)>request.max_tokens:raise ValueError('Native grammar token overflow')
    except (ValueError,TimeoutError) as exc:failure=f'{type(exc).__name__}: {exc}';legacy=None
    return dict(complete=failure is None,failure=failure,events=[asdict(e) for e in sorted(events)],
        count_decision=counts.tolist(),field_tape_version='exact-us-events-v1',legacy_tokens=legacy,steps=steps)
