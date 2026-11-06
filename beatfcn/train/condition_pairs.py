"""Controlled real-excerpt diagnostics, isolated from the production sampler.

Both requests start on the same blank chart and edit the same eight seconds.
Source events outside the excerpt are not supplied as history or certified as
the human chart's silence. Boundary-crossing notes reject the entire excerpt.
"""
from dataclasses import replace

import numpy as np
import torch

from ..events import controls as C
from ..events.canonical import MODE_TYPES,build_model_view,quantize_us
from ..events.requests import TrainingSample,assert_no_leak
from ..model import features as F
from ..model.batch import collate_meta


def blank_context_excerpt(chart, c0, duration, audio_id, seed):
    end=c0+F.CORE_US
    if not 0<=c0<end<=duration:raise ValueError('requires a full core inside audio')
    for e in chart.events:
        if e.is_hold and (e.onset_us<c0<e.end_us or e.onset_us<end<e.end_us):
            raise ValueError('boundary hold: reject excerpt rather than truncate or reveal it')
    targets=tuple(e for e in chart.events if c0<=e.onset_us<end)
    if not targets:raise ValueError('empty excerpt')
    if any(not c0<=quantize_us(e.onset_us)<end or (e.is_hold and quantize_us(e.end_us)>end) for e in targets):
        raise ValueError('quantized target crosses excerpt boundary')
    view=build_model_view(chart.mode,[],targets)
    if not view.ok:raise ValueError('excerpt model view quarantined')
    quantized=tuple(e for g in view.target_groups for e in g.events)
    bands,raw=C.derive_controls(chart.mode,targets,[(c0,end)])
    quantized_bands,_=C.derive_controls(chart.mode,quantized,[(c0,end)])
    if bands!=quantized_bands:raise ValueError('control bands change after model quantization')
    if set(bands)!=set(C.MODE_FIELDS[chart.mode]):raise ValueError('all applicable controls must have support')
    regions=([(0,c0,'complete')] if c0 else [])+[(c0,end,'editable')]
    if end<duration:regions.append((end,duration,'complete'))
    request=dict(schema_version='beatfcn.intent.v1',request_id=f'paired-excerpt:{audio_id}:{c0}',
        audio_id=audio_id,audio_duration_us=duration,mode=chart.mode,observed_events=[],
        regions=[dict(region_id=f'excerpt-{i}',start_us=a,end_us=b,state=s) for i,(a,b,s) in enumerate(regions)],
        references=[],controls={'allowed_types':list(MODE_TYPES[chart.mode]),**bands},
        timing_map_ref=None,scene_ref='diagnostic:rhythm-only-empty-scene' if chart.mode=='phira' else None,
        hard_profile_ref=None,seed=seed)
    recipe=dict(kind='blank_context_real_excerpt',scope_us=[c0,end],condition_source='training_human_excerpt',
        outside_context='diagnostic blank canvas; not an assertion that the source chart is silent',
        source_events_outside_excerpt=len(chart.events)-len(targets),target_notes_thinned=0)
    sample=TrainingSample(request,targets,(),(),recipe,view,raw)
    assert_no_leak(request,targets)
    return sample


def without_conditions(ex):
    return replace(ex,controls=np.full_like(ex.controls,-1))


def assert_same_generation_inputs(left, right, info, taps, top, tap_dim, summary_dim):
    """Compare all initial generation metadata, not teacher-forced future rows."""
    def initial(ex):
        ex=replace(ex,groups=[],right=ex.right[:1].copy())
        return collate_meta([ex],[info],taps,top,tap_dim,summary_dim,False)
    a,b=initial(left),initial(right)
    if a.keys()!=b.keys():raise ValueError('generation input keys differ')
    def equal(x,y):
        if torch.is_tensor(x):return torch.is_tensor(y) and torch.equal(x,y)
        if isinstance(x,np.ndarray):return isinstance(y,np.ndarray) and np.array_equal(x,y)
        if isinstance(x,dict):return isinstance(y,dict) and x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
        if isinstance(x,(list,tuple)):return type(x)==type(y) and len(x)==len(y) and all(equal(a,b) for a,b in zip(x,y))
        return x==y
    differences=[]
    for key in a:
        if key=='controls':continue
        if not equal(a[key],b[key]):differences.append(key)
    if differences:raise ValueError(f'non-condition input differs: {differences}')
    if torch.equal(a['controls'],b['controls']):raise ValueError('pair has identical conditions')
    return dict(non_condition_inputs_equal=True,compared_keys=sorted(k for k in a if k!='controls'),
                changed_fields=[name for i,name in enumerate(F.CONTROL_FIELDS) if left.controls[i]!=right.controls[i]])
