"""Same fixed-machine field controls, preserving original active-rank batch shape."""
from collections import defaultdict

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from .native_field_replay import ARMS,array,controlled_heads,direct_read,endpoint_replay,machine_tape,unique_head_targets


@torch.no_grad()
def replay_fields(model,output,inputs,request,layout,prediction,targets,upstream,*,health=False,checkpoint=lambda:None):
    tape,certificate=machine_tape(model,output,inputs,request,layout,prediction)
    paired=unique_head_targets(tape,targets);groups=defaultdict(list)
    for pos,r in enumerate(tape):groups[r['rank']].append((pos,r))
    arms=('base','head_mel_zero','head_mel_shift','end_mel_zero','end_mel_shift','end_song_zero','end_song_shift','clock') if health else ARMS
    results={};maximum=0.;device=output['x'].device
    for arm in arms:
        rows=[]
        for group in groups.values():
            cells=torch.tensor([r['cell'] for _,r in group],device=device)
            states=torch.tensor([r['state'] for _,r in group],device=device)
            queries=torch.tensor([r['query'] for _,r in group],device=device)
            heads=layout['heads'][[r['cell'] for _,r in group]]
            context,source=(upstream[arm.split('_')[1]] if arm.startswith('upstream_') else (output,inputs))
            features,coverage=controlled_heads(model,context,source,heads,cells,request.start_us,
                mel_mode='zero' if arm=='head_mel_zero' else 'shift' if arm=='head_mel_shift' else 'base',clock=arm=='clock')
            ids=torch.tensor([int(np.flatnonzero(layout['allowed'][r['cell']]&(layout['heads'][r['cell']]==r['event']['onset_us']))[0]) for _,r in group],device=device)
            selected=features[torch.arange(len(group),device=device),ids]
            if arm.startswith('song_'):
                context=dict(context);song=context['song'].clone();at=inputs['joint_frame_times'][0,cells]+.02
                new,_=direct_read(output['song_values'],output['song_times'],output['song_valid'],at,.5,arm.split('_')[1])
                song[0,cells]=new;context['song']=song
            kind_mask=torch.tensor([r['kind_allowed'] for _,r in group],device=device)
            head_mask=torch.tensor([r['head_allowed'] for _,r in group],device=device)
            logits=model.kind_logits(context,states,selected,cells).masked_fill(~kind_mask,-1e9)
            head_logits=model.head_logits(queries,features).masked_fill(~head_mask,-1e9)
            probability=logits.softmax(-1);head_probability=head_logits.softmax(-1)
            for j,(pos,r) in enumerate(group):
                c=r['cell'];h=r['event']['onset_us'];previous=RhythmEvent(**r['previous']) if r['previous'] else None
                row=dict(cell=c,rank=r['rank'],head_us=h,type_probability=array(probability[j]).tolist(),
                    type_argmax=int(logits[j].argmax()),head_probability=array(head_probability[j]).tolist(),
                    head_argmax_us=int(heads[j,int(head_logits[j].argmax())]),coverage_head=float(coverage[j,ids[j]]),
                    coverage_legal_candidates=float(coverage[j,head_mask[j]].mean()),
                    forced=sum(r['kind_allowed'])==1,matched_unique_head=pos in paired)
                target=paired.get(pos)
                if target is not None:
                    k=('tap','hold','flick','drag').index(target.type);row['true_type']=k
                    exact=np.flatnonzero((heads[j]==target.onset_us)&np.asarray(r['head_allowed'],bool))
                    row['head_support_failure']=len(exact)!=1;row['true_head_us']=target.onset_us
                    row['head_nll']=float(-head_logits[j].log_softmax(0)[int(exact[0])]) if len(exact)==1 else None
                    row['type_support_failure']=not r['kind_allowed'][k]
                    row['type_nll']=float(-logits[j].log_softmax(0)[k]) if r['kind_allowed'][k] else None
                    row['type_brier']=float(((probability[j]-torch.nn.functional.one_hot(torch.tensor(k,device=device),4))**2).sum())
                if r['event']['type']=='hold' or (target is not None and target.type=='hold'):
                    hf=torch.tensor(r['head_feature'],device=device)
                    root=model.tail_query(torch.cat((states[j],hf,hf.new_tensor([0.,1.,0.,0.]))))
                    endpoint_arm=arm if arm.startswith('end_') or arm=='clock' else 'base'
                    row['endpoint']=endpoint_replay(model,output,inputs,root,request,h,previous,endpoint_arm)
                    if target is not None and target.type=='hold':row['endpoint_true_path']=endpoint_replay(model,output,inputs,root,request,h,previous,endpoint_arm,target.end_us)
                if arm=='base':
                    if row['head_argmax_us']!=h or ('tap','hold','flick','drag')[row['type_argmax']]!=r['event']['type']:
                        raise ValueError('Batched BASE integer identity failed')
                    maximum=max(maximum,float(np.abs(array(probability[j])-r['type_probability']).max()),
                        float(np.abs(array(head_probability[j])-r['head_probability']).max()))
                    if r['event']['type']=='hold' and row['endpoint'].get('end_us')!=r['event']['end_us']:raise ValueError('BASE endpoint changed')
                rows.append(row)
            checkpoint()
        results[arm]=rows
    if maximum>1e-4:raise ValueError('Batched field probability identity failed')
    return dict(tape=tape,machine_count=prediction['count_decision'],certificate=certificate,field_identity_max_diff=maximum,
        unique_head_pairs=len(paired),ambiguous_or_unmatched=len(tape)-len(paired),
        type_support_failures=sum(r.get('type_support_failure',False) for r in results['base']),arms=results,
        scope='Original active-rank batch shape;fixed count/state/query/head/history;diagnostic only')
