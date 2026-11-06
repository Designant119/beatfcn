"""Exact tuple support, conditional denoising score, and finite resampling chain."""
from bisect import bisect_left
from collections import Counter
from dataclasses import asdict
import hashlib
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from beatfcn.model.event_canvas import RhythmEvent
from .exact_cell_support import ExactEndpointTree
from .local_bundle import batch_rows
from .native_diagnostic_priors import legal_duration_counts
from .native_joint_support import KINDS
from .typed_inventory import inventory_row


SLOTS = 256


def public_layout(row):
    r = row['request']
    heads = set(range((r.start_us+999)//1000*1000, r.end_us, 1000))
    heads.update(t for t in r.copy_times if r.start_us <= t < r.end_us)
    heads = sorted(h for h in heads if not any(lo <= h < hi for lo, hi in row.get('known_regions', ())))
    heads=np.asarray(heads,np.int64)
    kinds=np.tile([r.allowed_types is None or k in r.allowed_types for k in KINDS],(len(heads),1))
    end_possible=(heads//1000+1)*1000<=r.duration_us
    for value in r.copy_times:end_possible|=(heads<value)&(value<=r.duration_us)
    kinds[:,1]&=end_possible
    keep=kinds.any(-1)
    return dict(heads=heads[keep],kinds=kinds[keep])


def empty_state():
    return dict(status=np.zeros(SLOTS, np.int64), head=np.zeros(SLOTS, np.int64),
                kind=np.zeros(SLOTS, np.int64), end=np.zeros(SLOTS, np.int64))


def target_state(events, request, layout):
    if len(events) > min(SLOTS, request.max_events) or 1+3*len(events) > request.max_tokens:
        raise ValueError('Native tuple capacity OVERFLOW')
    state = empty_state()
    state['status'][:] = 1
    lookup = {int(h):i for i,h in enumerate(layout['heads'])}
    for slot, event in enumerate(sorted(events, key=lambda e:(e.onset_us,KINDS.index(e.type),e.end_us))):
        if event.lane != -1 or event.onset_us not in lookup or event.type not in KINDS:
            raise ValueError('Unsupported native tuple head')
        kind = KINDS.index(event.type)
        if not layout['kinds'][lookup[event.onset_us], kind]:
            raise ValueError('Unsupported native tuple kind')
        if kind == 1:
            ExactEndpointTree(request,event.onset_us).path(event.end_us)
        elif event.end_us != event.onset_us:
            raise ValueError('Non-hold duration')
        for key, value in [('status',2),('head',event.onset_us),('kind',kind),('end',event.end_us)]:
            state[key][slot] = value
    return state


def state_events(state):
    return sorted([RhythmEvent(int(state['head'][i]),int(state['end'][i]),KINDS[int(state['kind'][i])])
                   for i in range(SLOTS) if state['status'][i] == 2])


def masked_batch(row, device='cpu'):
    inputs, _ = batch_rows([row])
    layout = public_layout(row)
    target = target_state(row['targets'], row['request'], layout)
    if Counter(state_events(target)) != Counter(row['targets']):
        raise ValueError('Native tuple multiset roundtrip')
    frames=(row['request'].end_us-row['request'].start_us+39999)//40000
    inputs['joint_frame_times']=torch.arange(frames,dtype=torch.float32)[None]*.04
    return {k:v.to(device) for k,v in inputs.items()}, dict(layout=layout,state=target)


def stable_rng(*parts):
    digest = hashlib.sha256(repr(parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'little'))


def corruption(clean, rng, rate=None):
    if rate is None:
        u = rng.random()
        rate = 1. if u < .5 else .5 if u < .75 else .25
    selected = rng.random(SLOTS) < rate
    state = {k:v.copy() for k,v in clean.items()}
    state['status'][selected] = 0
    # Values at MASK are zeroed, not merely hidden behind a type embedding.
    for k in ('head','kind','end'):
        state[k][selected] = 0
    return state, selected, float(rate)


def sample_logits(logits, u):
    values = logits.detach().cpu().double().numpy()
    if not np.isfinite(values).any() or np.isnan(values).any():
        raise FloatingPointError('Invalid categorical support')
    weights = np.exp(values-np.max(values))
    cumulative = np.cumsum(weights/weights.sum())
    return min(int(np.searchsorted(cumulative, float(u), side='right')), len(values)-1)


def duration_interval_mass(prior, head, song_end, copies, lo, hi):
    """Exact native/COPY leaf prior mass in an inclusive endpoint interval."""
    numbers = legal_duration_counts(head,song_end,copies)
    normalizer = sum(w for w,n in zip(prior['duration_counts'],numbers) if n)
    mass = 0.
    lower = 0
    for weight, upper, number in zip(prior['duration_counts'],prior['duration_upper_us'],numbers):
        start, stop = max(head+1,lo,head+lower+1), min(song_end,hi,head+upper)
        if number and start <= stop:
            native = max(0,stop//1000-(start+999)//1000+1)
            fractional = sum(start<=t<=stop and t%1000!=0 for t in set(copies))
            mass += weight/normalizer*(native+fractional)/number
        lower = upper
    return mass


def endpoint_factor(model, output, slot, address, request, prior, *, target=None, uniforms=None, trace=False):
    head = int(output['memory']['layout']['heads'][address])
    tree = ExactEndpointTree(request,head)
    if not tree.blocks():
        raise ValueError('Hold without complete legal endpoint')
    path = tree.path(target) if target is not None else None
    selected = []
    loss = output['states'].sum()*0
    records = []
    for level in range(3):
        if level == 0:
            labels = tree.blocks()
            intervals = [(tree.base+b*2000000,tree.base+(b+1)*2000000-1) for b in labels]
        elif level == 1:
            base = tree.base+selected[0]*2000000
            labels = tree.cells(selected[0])
            intervals = [(base+c*40000,base+(c+1)*40000-1) for c in labels]
        else:
            leaves = tree.leaves(selected[0],selected[1])
            labels = tuple(range(len(leaves)))
            intervals = [(v,v) for v in leaves]
        centers = [min(tree.hi,max(tree.lo,(lo+hi)//2)) for lo,hi in intervals]
        masses = [duration_interval_mass(prior,head,request.duration_us,request.copy_times,lo,hi) for lo,hi in intervals]
        if not masses or min(masses) <= 0:
            raise ValueError('Incomplete prior endpoint support')
        residual = model.tail_scores(output,slot,address,centers,level)
        logits = residual+residual.new_tensor(np.log(masses))
        choice = labels.index(path[level]) if path is not None else sample_logits(logits,uniforms[level])
        logp = logits.log_softmax(0)[choice]
        loss = loss-logp
        selected.append(int(labels[choice]))
        if trace:records.append(dict(level=level,choice=int(choice),label=int(labels[choice]),log_probability=float(logp.detach().cpu()),support=len(labels)))
    end = int(intervals[selected[-1]][0])
    if target is not None and end != target:
        raise ValueError('Endpoint path identity')
    return end,loss,records


def tuple_loss(model, output, request, target, selected, rate, prior, *, details=False):
    device=output['states'].device;clean=target['state'];ids=np.flatnonzero(selected)
    active=clean['status']==2
    label=torch.tensor(active,dtype=torch.float32,device=device)
    binary=F.binary_cross_entropy_with_logits(output['active'][0],label,reduction='none')
    if not len(target['layout']['heads']):
        if active.any():raise ValueError('Target active in forced-NULL request')
        binary=output['active'][0]*0
    total=binary[torch.tensor(ids,device=device)].sum()
    parts=dict(occupancy=total,head=total*0,kind=total*0,tail=total*0)
    active_ids=np.flatnonzero(selected&active)
    type_probs=[];head_values=[];tail_values=[];binary_types=[]
    if len(active_ids):
        head_index={int(h):i for i,h in enumerate(target['layout']['heads'])}
        addresses=np.array([head_index[int(clean['head'][s])] for s in active_ids],np.int64)
        hlogits=model.head_logits(output,active_ids)
        head_nll=F.cross_entropy(hlogits,torch.tensor(addresses,device=device),reduction='none')
        klogits=model.type_logits(output,active_ids,addresses).masked_fill(~torch.tensor(target['layout']['kinds'][addresses],device=device),-torch.inf)
        kind_nll=F.cross_entropy(klogits,torch.tensor(clean['kind'][active_ids],device=device),reduction='none')
        parts['head']=head_nll.sum();parts['kind']=kind_nll.sum()
        for j,s in enumerate(active_ids):
            if clean['kind'][s]==1:
                _,value,_=endpoint_factor(model,output,int(s),int(addresses[j]),request,prior,target=int(clean['end'][s]))
                parts['tail']=parts['tail']+value
                if details:tail_values.append(dict(slot=int(s),nll=float(value.detach().cpu()),head=int(clean['head'][s]),end=int(clean['end'][s])))
        if details:
            type_probs=klogits.softmax(-1).detach().cpu().tolist()
            head_values=head_nll.detach().cpu().tolist()
            all_z=klogits.logsumexp(-1)
            for j,s in enumerate(active_ids):
                row=[]
                for k in range(4):
                    if not target['layout']['kinds'][addresses[j],k]:row.append(None)
                    elif clean['kind'][s]==k:row.append(float((all_z[j]-klogits[j,k]).detach().cpu()))
                    else:
                        other=klogits[j].clone();other[k]=-torch.inf
                        row.append(float((all_z[j]-other.logsumexp(-1)).detach().cpu()))
                binary_types.append(row)
    loss=sum(parts.values())/(SLOTS*rate)
    if not details:return loss,{k:v/(SLOTS*rate) for k,v in parts.items()}
    return loss,dict(parts={k:float(v.detach().cpu()) for k,v in parts.items()},selected=ids.tolist(),rate=rate,
        binary_nll=binary.detach().cpu().tolist(),active_label=active.tolist(),active_slots=active_ids.tolist(),
        type_probability=type_probs,type_binary_nll=binary_types,head_nll=head_values,tail_nll=tail_values,
        support=[target['layout']['kinds'][int(np.searchsorted(target['layout']['heads'],clean['head'][s]))].tolist() for s in active_ids],
        type_labels=clean['kind'][active_ids].tolist(),head_uniform_nll=math.log(len(target['layout']['heads'])) if active_ids.size else None,
        state={k:v.tolist() for k,v in clean.items()})


@torch.no_grad()
def decode_masked(model, inputs, request, layout, prior, *, seed, public_id, deadline=None, clear_callback=None):
    model.eval()
    if set(layout)!={'heads','kinds'}:raise ValueError('Public layout only')
    state=empty_state();traces=[];failure=None
    try:
        memory=model.encode(inputs,layout,request.start_us)
        for sweep in range(2):
            rng=stable_rng(seed,public_id,sweep,'generation')
            order=rng.permutation(SLOTS);uniforms=rng.random((SLOTS,6))
            for block in range(8):
                if deadline is not None and time.monotonic()>deadline:raise TimeoutError('Masked generation deadline')
                slots=order[block*32:(block+1)*32]
                state['status'][slots]=0
                for key in ('head','kind','end'):state[key][slots]=0
                output=model(memory,state)
                active_probability=output['active'][0,slots].sigmoid().cpu().numpy()
                if not len(layout['heads']):active_probability[:]=0
                chosen_slots=[]
                for j,s in enumerate(slots):
                    active=uniforms[s,0]<active_probability[j]
                    state['status'][s]=1  # publish active only after a complete valid tuple
                    if active:chosen_slots.append(int(s))
                    traces.append(dict(sweep=sweep,block=block,slot=int(s),active=bool(active),uniforms=uniforms[s].tolist(),
                                       active_probability=float(active_probability[j])))
                if chosen_slots:
                    hlogits=model.head_logits(output,chosen_slots)
                    addresses=np.array([sample_logits(hlogits[j],uniforms[s,1]) for j,s in enumerate(chosen_slots)],np.int64)
                    klogits=model.type_logits(output,chosen_slots,addresses).masked_fill(~torch.tensor(layout['kinds'][addresses],device=hlogits.device),-torch.inf)
                    for j,s in enumerate(chosen_slots):
                        address=int(addresses[j]);head=int(layout['heads'][address]);kind=sample_logits(klogits[j],uniforms[s,2]);end=head
                        if kind==1:end,_,_=endpoint_factor(model,output,s,address,request,prior,uniforms=uniforms[s,3:])
                        state['head'][s]=head;state['kind'][s]=kind;state['end'][s]=end;state['status'][s]=2
                        traces[-len(slots)+int(np.flatnonzero(slots==s)[0])].update(head=head,kind=kind,end=end)
                        model._coverage(memory,[head],'selected_head')
                        if kind==1:model._coverage(memory,[end],'selected_endpoint')
                    del hlogits,klogits
                del output
                if clear_callback is not None:clear_callback()
        events=state_events(state)
        target_state(events,request,layout)
    except (ValueError,TimeoutError,FloatingPointError) as exc:
        failure=f'{type(exc).__name__}: {exc}';events=state_events(state)
    return dict(complete=failure is None,failure=failure,events=[asdict(e) for e in events],
                trace=traces,field_tape_version='masked-native-exact-us-v1',legacy_tokens=None)


def visible_history(state, slot):
    ids=np.flatnonzero((state['status'][:slot]==2))
    return int(state['kind'][ids[-1]]) if len(ids) else -1


def conditional_type_prior(priors, state, slot, allowed, history):
    global_p=np.asarray(priors['global_type_probability'],np.float64)
    counts=np.asarray(priors['history_counts'].get(f'{slot//16}:{visible_history(state,slot)}',[0]*4),np.float64)
    values=(counts+4*global_p) if history else global_p
    values=values*np.asarray(allowed,bool)
    if values.sum()<=0:raise ValueError('Prior has no legal type')
    return values/values.sum()


def proper_record(model, output, request, target, state, selected, rate, priors):
    from .native_diagnostic_priors import duration_probability
    loss,detail=tuple_loss(model,output,request,target,selected,rate,priors['duration'],details=True)
    types={k:[] for k in KINDS}
    for j,slot in enumerate(detail['active_slots']):
        allowed=detail['support'][j];truth=detail['type_labels'][j]
        if sum(allowed)==1:continue
        frequency=conditional_type_prior(priors,state,slot,allowed,False)
        history=conditional_type_prior(priors,state,slot,allowed,True)
        for k,name in enumerate(KINDS):
            if not allowed[k]:continue
            positive=k==truth
            def binary(prob):return -math.log(float(prob) if positive else max(1-float(prob),1e-300))
            types[name].append(dict(slot=slot,positive=positive,model_nll=detail['type_binary_nll'][j][k],
                frequency_nll=binary(frequency[k]),history_nll=binary(history[k])))
    tails=[]
    for value in detail['tail_nll']:
        p=duration_probability(priors['duration'],value['head'],value['end'],request.duration_us,request.copy_times)
        if p<=0:raise ValueError('True endpoint outside prior support')
        tails.append(dict(**value,prior_nll=-math.log(p)))
    null_prior=np.asarray(priors['slot_active_probability'])
    occupancy=[]
    for slot in detail['selected']:
        pos=detail['active_label'][slot]
        occupancy.append(dict(slot=slot,positive=pos,nll=detail['binary_nll'][slot],
                              prior_nll=-math.log(null_prior[slot] if pos else 1-null_prior[slot])))
    return dict(loss=float(loss.detach().cpu()),parts=detail['parts'],rate=rate,
        selected_slots=detail['selected'],active_slots=detail['active_slots'],occupancy=occupancy,
        types=types,head_nll=detail['head_nll'],head_uniform_nll=detail['head_uniform_nll'],tail=tails,
        corruption_provenance='training_state_only_not_public_reference',
        corruption_sha256=hashlib.sha256(b''.join(state[k].tobytes() for k in ('status','head','kind','end'))).hexdigest())
