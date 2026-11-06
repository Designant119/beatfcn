"""Read-only replay of a completed machine tape; never changes generation.

Every counterfactual restarts from the baseline state/head/history. Teacher
scores and unique-head matched scores are explicitly separate from free output.
"""
from collections import Counter
from dataclasses import asdict
import math

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.local_bundle import read_grid
from beatfcn.train.exact_cell_support import ExactEndpointTree
from beatfcn.train.native_cell_chain import previous_vectors, canonical_head_mask, kind_mask, endpoint_decision


ARMS=('base','head_mel_zero','head_mel_shift','song_zero','song_shift',
      'upstream_zero','upstream_shift','end_mel_zero','end_mel_shift',
      'end_song_zero','end_song_shift','clock')


def array(tensor):
    result=tensor.detach().cpu().numpy()
    if not np.isfinite(result).all():raise FloatingPointError('Nonfinite diagnostic tensor')
    return result


def direct_read(values,times,valid,query,spacing,mode):
    at=query+(.733 if mode=='shift' else 0.)
    result=read_grid(values,times,valid,at.reshape(1,-1),spacing)[0]
    if mode=='zero':result=torch.zeros_like(result)
    low=times[0,0]-spacing/2
    high=times[0].masked_fill(~valid[0],-1e9).max()+spacing/2
    coverage=((at>=low)&(at<=high)).float()
    return result,coverage


def controlled_heads(model,output,inputs,heads,cells,origin,mel_mode='base',clock=False):
    values=np.asarray(heads,np.int64);device=output['x'].device
    times=torch.tensor((values-origin)/1e6,dtype=torch.float32,device=device)
    raw,changed_coverage=direct_read(inputs['mel'],inputs['mel_times'],inputs['mel_valid'],times,.01,mel_mode)
    raw=raw.reshape(*times.shape,128)
    lo=inputs['mel_times'][0,0]-.005
    hi=inputs['mel_times'][0].masked_fill(~inputs['mel_valid'][0],-1e9).max()+.005
    coverage=((times>=lo)&(times<=hi)).float()
    phase=np.stack((((values-origin)%40000)/40000,(values%1000)/1000,(values%1000!=0).astype(float)),-1)
    phase=torch.tensor(phase,dtype=torch.float32,device=device)
    frames=output['x'][0,cells]
    while frames.ndim<raw.ndim:frames=frames.unsqueeze(-2)
    frames=frames.expand(*raw.shape[:-1],frames.shape[-1])
    features=model.address_feature(torch.cat((frames,raw,model.time_features(times+(.733 if clock else 0.)),coverage[...,None],phase),-1))
    return features,changed_coverage


def controlled_end_keys(model,output,inputs,values,origin,mel_mode='base',song_mode='base',clock=False):
    values=np.asarray(values,np.int64);device=output['x'].device
    times=torch.tensor((values-origin)/1e6,dtype=torch.float32,device=device)
    song,sc=direct_read(output['song_values'],output['song_times'],output['song_valid'],times,.5,song_mode)
    raw,mc=direct_read(inputs['mel'],inputs['mel_times'],inputs['mel_valid'],times,.01,mel_mode)
    lo=inputs['mel_times'][0,0]-.005;hi=inputs['mel_times'][0].masked_fill(~inputs['mel_valid'][0],-1e9).max()+.005
    coverage=((times>=lo)&(times<=hi)).float()
    phase=np.stack((((values-origin)%40000)/40000,(values%1000)/1000,(values%1000!=0).astype(float)),-1)
    phase=torch.tensor(phase,dtype=torch.float32,device=device)
    keys=model.endpoint_key(torch.cat((song,raw,model.time_features(times+(.733 if clock else 0.)),coverage[:,None],phase),-1))
    return keys,dict(mel=float(mc.mean()),song=float(sc.mean()))


def endpoint_replay(model,output,inputs,root,request,head,previous,arm,target=None):
    minimum=previous.end_us if previous is not None and previous.onset_us==head and previous.type=='hold' else None
    tree=ExactEndpointTree(request,head,minimum)
    if not tree.blocks():return dict(support_failure=True)
    try:path=tree.path(target) if target is not None else None
    except ValueError:return dict(support_failure=True)
    selected=[];query=root;levels=[];nll=0.;prior=0.
    for level in range(3):
        if level==0:
            labels=tree.blocks();times=[min(tree.hi,max(tree.lo,tree.base+b*2000000+1000000)) for b in labels]
        elif level==1:
            labels=tree.cells(selected[0]);times=[min(tree.hi,max(tree.lo,tree.base+selected[0]*2000000+c*40000+20000)) for c in labels]
        else:
            times=list(tree.leaves(*selected));labels=tuple(range(len(times)))
        keys,coverage=controlled_end_keys(model,output,inputs,times,request.start_us,
            mel_mode='zero' if arm=='end_mel_zero' else 'shift' if arm=='end_mel_shift' else 'base',
            song_mode='zero' if arm=='end_song_zero' else 'shift' if arm=='end_song_shift' else 'base',clock=arm=='clock')
        logits=model.endpoint_scores(query,keys,level)
        choice=labels.index(path[level]) if path else int(logits.argmax())
        prob=logits.softmax(0);selected.append(int(labels[choice]))
        if path:nll-=float(logits.log_softmax(0)[choice]);prior+=math.log(len(labels))
        levels.append(dict(level=level,labels=list(labels),probability=array(prob).tolist(),choice=selected[-1],coverage=coverage))
        # Each arm conditions on its own keys along the same true branch path.
        if level<2:query=model.endpoint_condition(keys[choice][None],query[None])[0]
    return dict(support_failure=False,end_us=int(times[selected[-1]]),levels=levels,
                nll=nll if path else None,uniform_path_nll=prior if path else None,
                teacher_endpoint=path is not None)


@torch.no_grad()
def machine_tape(model,output,inputs,request,layout,prediction):
    if not prediction['complete']:raise ValueError('Cannot certify incomplete baseline tape')
    counts=np.asarray(prediction['count_decision'],np.int64)
    events={(s['cell'],s['rank']):s for s in prediction['steps']}
    active=np.where(counts>0)[0];records=[];max_difference=0.
    if not len(active):return records,dict(max_probability_diff=0.,integer_events_equal=True,coverage_members=0)
    cells=torch.tensor(active,dtype=torch.long,device=output['x'].device)
    state,embedding=model.initial_members(output,cells,torch.tensor(counts[active],device=cells.device))
    previous=[None]*len(active);history={int(c):[] for c in active}
    for rank in range(int(counts.max())):
        pv=previous_vectors(model,previous,request.start_us,inputs)
        state,query=model.member_step(output,cells,embedding,state,pv)
        features=model.address_features(output,inputs,layout['heads'][active],cells,request.start_us)
        allowed=np.stack([canonical_head_mask(layout['heads'][c],layout['allowed'][c],p,request) for c,p in zip(active,previous)])
        logits=model.head_logits(query,features).masked_fill(~torch.tensor(allowed,device=cells.device),-1e9)
        chosen=logits.argmax(-1).cpu().numpy();ids=torch.tensor(chosen,device=cells.device)
        selected=features[torch.arange(len(active),device=cells.device),ids]
        heads=layout['heads'][active,chosen]
        legal=np.stack([kind_mask(request,int(h),p) for h,p in zip(heads,previous)])
        kind=model.kind_logits(output,state,selected,cells).masked_fill(~torch.tensor(legal,device=cells.device),-1e9)
        current=[]
        for j,(c,h) in enumerate(zip(active,heads)):
            base=events[(int(c),rank)];event=RhythmEvent(**base['event'])
            if int(h)!=event.onset_us or ('tap','hold','flick','drag')[int(kind[j].argmax())]!=event.type:
                raise ValueError('Machine replay changed baseline integer head/type')
            for actual,expected in ((logits[j].softmax(-1),base['head_probability']),(kind[j].softmax(-1),base['type_probability'])):
                max_difference=max(max_difference,float(np.abs(array(actual)-np.asarray(expected)).max()))
            if event.type=='hold':
                end,_,tail=endpoint_decision(model,output,inputs,state[j],selected[j],request,int(h),previous[j],trace=True)
                if end!=event.end_us:raise ValueError('Machine replay changed baseline integer endpoint')
                for a,b in zip(tail,base['tail']):
                    if a['labels']!=b['labels']:raise ValueError('Replay tree support changed')
                    max_difference=max(max_difference,float(np.abs(np.asarray(a['probability'])-b['probability']).max()))
            records.append(dict(cell=int(c),rank=rank,event=asdict(event),previous=asdict(previous[j]) if previous[j] else None,
                history=list(history[int(c)]),state=array(state[j]).tolist(),query=array(query[j]).tolist(),
                count_embedding=array(embedding[j]).tolist(),previous_vector=array(pv[j]).tolist(),
                head_feature=array(selected[j]).tolist(),head_allowed=allowed[j].tolist(),kind_allowed=legal[j].tolist(),
                head_probability=base['head_probability'],type_probability=base['type_probability'],tail=base['tail']))
            history[int(c)].append(asdict(event));current.append(event)
        keep=np.where(counts[active]>rank+1)[0]
        if not len(keep):break
        previous=[current[j] for j in keep];active=active[keep];cells=cells[keep];state=state[keep];embedding=embedding[keep]
    if max_difference>1e-4:raise ValueError('Machine tape probability replay failed')
    return records,dict(max_probability_diff=max_difference,integer_events_equal=True,coverage_members=len(records))


def unique_head_targets(tape,targets):
    links=[[j for j,t in enumerate(targets) if abs(t.onset_us-r['event']['onset_us'])<=10000] for r in tape]
    degree=Counter(j for values in links for j in values)
    return {i:targets[values[0]] for i,values in enumerate(links) if len(values)==1 and degree[values[0]]==1}


@torch.no_grad()
def replay_fields(model,output,inputs,request,layout,prediction,targets,upstream,*,health=False,checkpoint=lambda:None):
    tape,certificate=machine_tape(model,output,inputs,request,layout,prediction)
    paired=unique_head_targets(tape,targets);results={}
    arms=('base','head_mel_zero','head_mel_shift','end_mel_zero','end_mel_shift','end_song_zero','end_song_shift','clock') if health else ARMS
    max_identity=0.
    for arm in arms:
        rows=[]
        for pos,r in enumerate(tape):
            c=r['cell'];h=r['event']['onset_us'];device=output['x'].device
            cell=torch.tensor([c],device=device);state=torch.tensor(r['state'],device=device)
            query=torch.tensor(r['query'],device=device);previous=RhythmEvent(**r['previous']) if r['previous'] else None
            context,source=(upstream[arm.split('_')[1]] if arm.startswith('upstream_') else (output,inputs))
            head_values=layout['heads'][[c]]
            features,coverage=controlled_heads(model,context,source,head_values,cell,request.start_us,
                mel_mode='zero' if arm=='head_mel_zero' else 'shift' if arm=='head_mel_shift' else 'base',clock=arm=='clock')
            selected_index=int(np.flatnonzero(layout['allowed'][c]&(layout['heads'][c]==h))[0])
            feature=features[0,selected_index]
            if arm.startswith('song_'):
                context=dict(context);song=context['song'].clone()
                at=inputs['joint_frame_times'][0,c]+.02
                new,_=direct_read(output['song_values'],output['song_times'],output['song_valid'],at.reshape(1),.5,arm.split('_')[1])
                song[0,c]=new[0];context['song']=song
            legal=torch.tensor(r['kind_allowed'],device=device)
            logits=model.kind_logits(context,state[None],feature[None],cell)[0].masked_fill(~legal,-1e9)
            head_logits=model.head_logits(query[None],features)[0].masked_fill(~torch.tensor(r['head_allowed'],device=device),-1e9)
            probability=logits.softmax(0);row=dict(cell=c,rank=r['rank'],head_us=h,
                type_probability=array(probability).tolist(),head_probability=array(head_logits.softmax(0)).tolist(),
                type_argmax=int(logits.argmax()),
                head_argmax_us=int(head_values[0,int(head_logits.argmax())]),coverage_head=float(coverage.reshape(-1)[selected_index]),
                coverage_legal_candidates=float(coverage.reshape(-1)[torch.tensor(r['head_allowed'],device=device)].mean()),
                forced=sum(r['kind_allowed'])==1,matched_unique_head=pos in paired)
            target=paired.get(pos)
            if target is not None:
                k=('tap','hold','flick','drag').index(target.type);row['true_type']=k
                exact=np.flatnonzero((head_values[0]==target.onset_us)&np.asarray(r['head_allowed'],bool))
                row['head_support_failure']=len(exact)!=1
                row['head_nll']=float(-head_logits.log_softmax(0)[int(exact[0])]) if len(exact)==1 else None
                row['true_head_us']=target.onset_us
                row['type_support_failure']=not r['kind_allowed'][k]
                row['type_nll']=float(-logits.log_softmax(0)[k]) if r['kind_allowed'][k] else None
                row['type_brier']=float(((probability-torch.nn.functional.one_hot(torch.tensor(k,device=device),4))**2).sum())
            if r['event']['type']=='hold' or (target is not None and target.type=='hold'):
                # Freeze the baseline head feature/root for endpoint-only effects.
                hf=torch.tensor(r['head_feature'],device=device)
                root=model.tail_query(torch.cat((state,hf,hf.new_tensor([0.,1.,0.,0.])),0))
                end_arm=arm if arm.startswith('end_') or arm=='clock' else 'base'
                row['endpoint']=endpoint_replay(model,output,inputs,root,request,h,previous,end_arm)
                if target is not None and target.type=='hold':
                    row['endpoint_true_path']=endpoint_replay(model,output,inputs,root,request,h,previous,end_arm,target.end_us)
            if arm=='base':
                if row['head_argmax_us']!=h or ('tap','hold','flick','drag')[row['type_argmax']]!=r['event']['type']:
                    raise ValueError('Field baseline changed integer head/type')
                max_identity=max(max_identity,float(np.abs(array(probability)-np.asarray(r['type_probability'])).max()),
                    float(np.abs(array(head_logits.softmax(0))-np.asarray(r['head_probability'])).max()))
                if r['event']['type']=='hold' and row['endpoint'].get('end_us')!=r['event']['end_us']:
                    raise ValueError('Endpoint-only baseline identity failed')
            rows.append(row)
            if pos%8==7:checkpoint()
        results[arm]=rows;checkpoint()
    if max_identity>1e-4:raise ValueError('Field intervention identity failed')
    return dict(tape=tape,machine_count=prediction['count_decision'],certificate=certificate,
        field_identity_max_diff=max_identity,unique_head_pairs=len(paired),ambiguous_or_unmatched=len(tape)-len(paired),
        type_support_failures=sum(r.get('type_support_failure',False) for r in results['base']),
        arms=results,scope='Fixed baseline count/state/query/head/history;conditional diagnostics,not full free quality')
