"""Exploratory component metrics and explicitly teacher-conditioned scores."""
from collections import defaultdict
import math

import numpy as np

from .exact_cell_support import ExactEndpointTree
from .native_joint_support import KINDS


def teacher_scores(steps,target,request,count_probability,prior):
    count_nll=-sum(math.log(max(float(p[int(n)]),1e-300)) for p,n in zip(count_probability,target['counts']))
    result=dict(count_nll=count_nll,count_cells=len(target['counts']),type_nll=[],type_prior_nll=[],
                endpoint_nll=[],endpoint_prior_nll=[],type_binary={k:[] for k in KINDS},forced=0,support_failures=0)
    previous={}
    for row in steps:
        cell,rank=row['cell'],row['rank'];truth=target['members'][cell][rank];k=truth.kind
        valid=np.asarray(row['kind_allowed'],bool);prob=np.asarray(row['type_probability'],float)
        pp=np.asarray(prior,float)*valid;pp/=pp.sum()
        if valid.sum()==1:result['forced']+=1
        else:
            result['type_nll'].append(-math.log(max(prob[k],1e-300)))
            result['type_prior_nll'].append(-math.log(max(pp[k],1e-300)))
            for j,name in enumerate(KINDS):
                if not valid[j]:continue
                pos=j==k;p=max(min(prob[j],1-1e-15),1e-300);p0=max(min(pp[j],1-1e-15),1e-300)
                result['type_binary'][name].append(dict(positive=pos,nll=-math.log(p if pos else 1-p),prior_nll=-math.log(p0 if pos else 1-p0)))
        if k==1:
            h=row['event']['onset_us'];prev=previous.get(cell)
            minimum=prev['end_us'] if prev and prev['onset_us']==h and prev['type']=='hold' else None
            path=ExactEndpointTree(request,h,minimum).path(truth.end_us)
            nll=prior_nll=0.
            for level,trace in enumerate(row['tail']):
                choice=trace['labels'].index(path[level]);nll-=math.log(max(trace['probability'][choice],1e-300));prior_nll+=math.log(len(trace['labels']))
            result['endpoint_nll'].append(nll);result['endpoint_prior_nll'].append(prior_nll)
        previous[cell]=row['event']
    return result


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['component']].append(r)
    components={c:dict(full50=float(np.mean([r['scores']['full_note_f1'] for r in rr])),
                       onset10=float(np.mean([r['scores']['onset_10ms']['f1'] or 0 for r in rr])),
                       null10=float(np.mean([np.mean(r['output_null_onset10']) for r in rr]))) for c,rr in groups.items()}
    types={}
    for kind in KINDS:
        tp=sum(r['scores']['per_type'][kind]['tp'] for r in rows)
        pred=sum(r['scores']['per_type'][kind]['pred'] for r in rows)
        true=sum(r['scores']['per_type'][kind]['true'] for r in rows)
        types[kind]=dict(tp=tp,pred=pred,true=true,f1=2*tp/max(pred+true,1),
                        matched_components=len({r['component'] for r in rows if r['scores']['per_type'][kind]['tp']>0}))
    return dict(requests=len(rows),components=components,complete=sum(r['prediction']['complete'] for r in rows),
        full50_component_macro=float(np.mean([r['full50'] for r in components.values()])),
        onset10_component_macro=float(np.mean([r['onset10'] for r in components.values()])),
        null10_component_macro=float(np.mean([r['null10'] for r in components.values()])),
        pred=sum(len(r['prediction']['events']) for r in rows),true=sum(len(r['target']) for r in rows),types=types)
