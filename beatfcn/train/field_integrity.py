"""Full-support, full-probability replay checks and read-only coverage metadata."""
from collections import Counter
import math

import numpy as np

from .exact_cell_support import ExactEndpointTree
from .native_field_replay_batched import replay_fields as batched_replay


def probability(values,mask=None):
    a=np.asarray(values,np.float64)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all() or (a<0).any() or (a>1.000001).any():raise ValueError('Invalid probability vector')
    if abs(float(a.sum())-1)>1e-4:raise ValueError('Probability normalization')
    if mask is not None:
        m=np.asarray(mask,bool)
        if len(m)!=len(a) or np.any(a[~m]>1e-12):raise ValueError('Probability/mask support')
    return a


def validate(result,prediction=None):
    tape=result['tape'];base=result['arms']['base'];keys=[(r['cell'],r['rank'],r['event']['onset_us']) for r in tape]
    if len(keys)!=len(set(keys)):raise ValueError('Repeated tape member identity')
    counts=Counter(r['cell'] for r in tape)
    if any(counts.get(c,0)!=n for c,n in enumerate(result['machine_count'])):raise ValueError('Count/tape multiplicity')
    if prediction is not None:
        if result['machine_count']!=prediction['count_decision']:raise ValueError('Changed machine counts')
        get=lambda e:(e['onset_us'],e['end_us'],e['type'],e.get('lane',-1))
        if Counter(get(r['event']) for r in tape)!=Counter(get(e) for e in prediction['events']):raise ValueError('Changed integer multiset')
    maximum=0.;hold_paths=0
    for arm,rows in result['arms'].items():
        if [(r['cell'],r['rank'],r['head_us']) for r in rows]!=keys:raise ValueError('Arm member order/count changed')
        for i,r in enumerate(rows):
            original=tape[i]
            for name,mask in [('head_probability','head_allowed'),('type_probability','kind_allowed')]:
                p=probability(r[name],original[mask]);q=probability(original[name],original[mask])
                if len(p)!=len(q):raise ValueError('Probability length changed')
                if arm=='base':maximum=max(maximum,float(np.abs(p-q).max()))
            if arm=='base' and (r['head_argmax_us']!=original['event']['onset_us'] or ('tap','hold','flick','drag')[r['type_argmax']]!=original['event']['type']):raise ValueError('BASE integer head/type')
            for endpoint_name in ('endpoint','endpoint_true_path'):
                end=r.get(endpoint_name)
                if end is None or end.get('support_failure'):continue
                if len(end['levels'])!=3:raise ValueError('Endpoint must contain3levels')
                nll=0.;underflow=False
                for level,trace in enumerate(end['levels']):
                    if trace['level']!=level or len(trace['labels'])!=len(trace['probability']):raise ValueError('Endpoint support lengths')
                    p=probability(trace['probability'])
                    if trace['choice'] not in trace['labels']:raise ValueError('Endpoint choice outside support')
                    if end.get('teacher_endpoint'):
                        value=float(p[trace['labels'].index(trace['choice'])]);underflow|=value==0
                        nll-=math.log(max(value,1e-300))
                if end.get('teacher_endpoint') and (not math.isfinite(end['nll']) or (not underflow and abs(nll-end['nll'])>1e-4)):
                    raise ValueError('Truepath probability/NLL mismatch')
            if arm=='base' and original['event']['type']=='hold':
                end=r.get('endpoint',{})
                if end.get('end_us')!=original['event']['end_us'] or len(original['tail'])!=3:raise ValueError('BASE integer endpoint')
                if len(end['levels'])!=len(original['tail']):raise ValueError('BASE endpoint layer count')
                for j in range(3):
                    a=end['levels'][j];b=original['tail'][j]
                    if a['labels']!=b['labels']:raise ValueError('BASE endpoint support changed')
                    p=probability(a['probability']);q=probability(b['probability'])
                    if len(p)!=len(q):raise ValueError('BASE endpoint probability length')
                    maximum=max(maximum,float(np.abs(p-q).max()))
                hold_paths+=1
    if maximum>1e-4:raise ValueError('Full BASE probability identity')
    return dict(status='passed',maximum_probability_diff=maximum,member_count=len(tape),machine_hold_paths=hold_paths,
                arms=len(result['arms']),empty_tape=len(tape)==0,scope='All lengths/supports/head/type/3endpoint probabilities and integers;empty is zero coverage')


def cpu(t):return t.detach().cpu().numpy()


def coverage(times,valid,query,spacing,shift=0.):
    t=np.asarray(times).reshape(-1);v=np.asarray(valid,bool).reshape(-1);q=np.asarray(query,np.float32)+np.float32(shift)
    if not v.any():return np.zeros(q.shape,bool)
    lo=np.float32(t[v].min()-spacing/2);hi=np.float32(t[v].max()+spacing/2)
    return (q>=lo)&(q<=hi)


def coverage_summary(flags,selected=None):
    a=np.asarray(flags,bool).reshape(-1)
    return dict(total=len(a),in_range=int(a.sum()),out_of_range=int((~a).sum()),fraction=float(a.mean()) if len(a) else None,
                selected_in_range=bool(a[selected]) if selected is not None else None)


def endpoint_coverage(end,request,record,inputs,output,arm):
    if not end or end.get('support_failure'):return dict(status='unsupported_or_missing')
    h=record['event']['onset_us'];previous=record['previous']
    minimum=previous['end_us'] if previous and previous['onset_us']==h and previous['type']=='hold' else None
    tree=ExactEndpointTree(request,h,minimum);selected=[];rows=[]
    for level,trace in enumerate(end['levels']):
        labels=trace['labels']
        if level==0:times=[min(tree.hi,max(tree.lo,tree.base+b*2000000+1000000)) for b in labels]
        elif level==1:times=[min(tree.hi,max(tree.lo,tree.base+selected[0]*2000000+c*40000+20000)) for c in labels]
        else:times=list(tree.leaves(selected[0],selected[1]))
        if len(times)!=len(labels):raise ValueError('Coverage endpoint support length')
        at=np.asarray((np.asarray(times,np.int64)-request.start_us)/1e6,np.float32);choice=labels.index(trace['choice'])
        rows.append(dict(level=level,selected_query_s=float(at[choice]),
            mel=coverage_summary(coverage(inputs['mel_times'],inputs['mel_valid'],at,.01,.733 if arm=='end_mel_shift' else 0.),choice),
            song=coverage_summary(coverage(output['song_times'],output['song_valid'],at,.5,.733 if arm=='end_song_shift' else 0.),choice)))
        selected.append(trace['choice'])
    return dict(status='measured',levels=rows,end_us=end['end_us'],teacher_path=end.get('teacher_endpoint',False),
                zero_mel=arm=='end_mel_zero',zero_song=arm=='end_song_zero')


def add_coverage(result,inputs,output,request,layout):
    x={k:cpu(v) for k,v in inputs.items() if k.endswith(('_times','_valid')) or k=='joint_frame_times'}
    out={k:cpu(output[k]) for k in ('song_times','song_valid')};tape=result['tape'];global_cov={}
    for arm,rows in result['arms'].items():
        shift=.733 if arm=='upstream_shift' else 0.
        global_cov[arm]={}
        for name,step in [('local',.04),('mel',.01),('song',.5)]:
            q=x[name+'_times'][x[name+'_valid']]
            global_cov[arm][name]=coverage_summary(coverage(x[name+'_times'],x[name+'_valid'],q,step,shift))
        global_cov[arm]['zero_feature_intervention']=arm=='upstream_zero'
        for j,row in enumerate(rows):
            r=tape[j];cell=r['cell'];heads=(layout['heads'][cell]-request.start_us)/1e6;mask=np.asarray(r['head_allowed'],bool)
            chosen=int(np.flatnonzero(layout['allowed'][cell]&(layout['heads'][cell]==row['head_us']))[0])
            flags=coverage(x['mel_times'],x['mel_valid'],heads,.01,.733 if arm in ('head_mel_shift','upstream_shift') else 0.)
            song_at=np.array([x['joint_frame_times'][0,cell]+.02],np.float32)
            row['coverage_paths']=dict(head_mel_legal_candidates=coverage_summary(flags[mask]),head_mel_selected=bool(flags[chosen]),
                type_song_read=coverage_summary(coverage(out['song_times'],out['song_valid'],song_at,.5,.733 if arm=='song_shift' else 0.)),
                endpoint_free=endpoint_coverage(row.get('endpoint'),request,r,x,out,arm),
                endpoint_true=endpoint_coverage(row.get('endpoint_true_path'),request,r,x,out,arm),
                acoustic_zero=arm in ('head_mel_zero','song_zero','end_mel_zero','end_song_zero','upstream_zero'),
                clock_feature_delta_s=.733 if arm=='clock' else 0.)
    result['source_coverage']=global_cov
    result['coverage_scope']='Clock-interval geometry and query clipping,not waveform silence or all encoder receptive-field support;full outcome denominator retained'


def replay_fields(*args,**kwargs):
    result=batched_replay(*args,**kwargs)
    result['full_probability_certificate']=validate(result,args[5])
    add_coverage(result,args[2],args[1],args[3],args[4])
    return result
