"""Prospective matched-control addendum and one supervised C17 lifecycle.

Core model/source/plan remain immutable. This entry includes both extra controls
inside the same evaluation and overall resource caps, before any actual update.
"""
from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import torch

from scripts import run_masked_native_events as core
from scripts.run_native_cell_chain import closure
from scripts.run_local_portfolio_scale import sha,write,weight_digest
from scripts.check_experiment_storage import admission,GIB
from beatfcn.train.managed_mps import ManagedMPS
from beatfcn.train.masked_native_events import masked_batch,inventory_row,empty_state,proper_record,decode_masked
from beatfcn.train.native_trial_evaluation import score_outcome
from beatfcn.model.event_canvas import RhythmEvent


ROOT=core.ROOT;ART=core.ART;EXTRA=ART/'matched-controls-plan.json'


def same_song_local_roll(inputs):
    result=dict(inputs)
    for name,dim,shift in [('local',2,18),('mel',1,72)]:
        valid=np.flatnonzero(inputs[name+'_valid'][0].detach().cpu().numpy())
        index=np.arange(inputs[name+'_times'].shape[1])
        index[valid]=np.roll(valid,-shift)
        result[name]=inputs[name].index_select(dim,torch.tensor(index,device=inputs[name].device))
    return result


def full_supported(target,source):
    result={}
    for name,spacing in [('local',.04),('mel',.01),('song',.5)]:
        q=target[name+'_times'][target[name+'_valid']].detach().cpu().numpy()
        t=source[name+'_times'][source[name+'_valid']].detach().cpu().numpy()
        fraction=float(np.mean((q>=t.min()-spacing/2)&(q<=t.max()+spacing/2))) if len(q) and len(t) else 0.
        result[name]=fraction
    return result


def prepare():
    if EXTRA.exists():raise FileExistsError('Preserve matched-control freeze')
    core.runtime();plan=core.read(core.PLAN);prep=core.read(ART/'preparation.json')
    cap=600-prep['wall_s']
    if cap<=2:raise TimeoutError('Preparation budget exhausted')
    guard=ManagedMPS(ROOT,ART,cap);started=time.monotonic()
    def stop(*_):raise TimeoutError('Matched-control preparation cap')
    signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(cap)-2))
    try:
        panel,row=core.load_data();pool={};records=[]
        for index in plan['indices']['dev']:
            guard.check('matched_control_geometry')
            rr=inventory_row(panel,row(index,'draft'));x,_=masked_batch(rr,'cpu');pool[index]=(rr,x)
        for position,index in enumerate(plan['indices']['dev']):
            guard.check('matched_control_selection');rr,x=pool[index]
            zero=core.original.input_digest(core.original.zero_audio(x))
            rolled=same_song_local_roll(x)
            if core.original.input_digest(core.original.zero_audio(rolled))!=zero:raise ValueError('Roll changes zero-audio input')
            records.append(dict(task='matched_local_roll',index=index,component=panel.records[index]['component'],eligible=True,
                public_input_sha256=core.original.input_digest(rolled),core_zero_sha256=zero,source_full_coverage={k:1. for k in ('local','mel','song')},
                intervention='local18frames=.72s and mel72frames=.72s circular roll on valid values; song unchanged; clocks/validity unchanged; wraps disclosed'))
            donor=None;coverage=None;transformed=None
            for offset in range(1,len(plan['indices']['dev'])):
                candidate=plan['indices']['dev'][(position+offset)%len(plan['indices']['dev'])]
                if panel.records[candidate]['component']==panel.records[index]['component']:continue
                _,dx=pool[candidate];fractions=full_supported(x,dx)
                if not all(v==1. for v in fractions.values()):continue
                value,cov=core.original.audio_variant(x,dx,scale=1.)
                if not all(float(v)>=1.-1e-7 for v in cov.values()):continue
                donor=candidate;coverage={k:float(v) for k,v in cov.items()};transformed=value;break
            record=dict(task='matched_foreign',index=index,component=panel.records[index]['component'],eligible=donor is not None,donor=donor,
                status='registered' if donor is not None else 'no_full_support_donor',source_full_coverage=coverage,
                selection='First different-component donor in fixed cyclic dev order covering every valid target input timestamp in all3streams;scale1;no target labels or scores used')
            if donor is not None:
                if core.original.input_digest(core.original.zero_audio(transformed))!=zero:raise ValueError('Matched foreign changes zero-audio input')
                record.update(public_input_sha256=core.original.input_digest(transformed),core_zero_sha256=zero)
            records.append(record)
        duration=time.monotonic()-started
        meta=dict(id='c17-masked-native-matched-controls-v1',core_plan_sha256=sha(core.PLAN),records=records,
            eligible=sum(r['eligible'] for r in records),all_target_records=len(records),wall_s=duration,
            global_wall_s=9000,evaluation_core_plus_addendum_wall_s=1200,optimizer_updates=0,cloud_calls=0,
            ordering='Core340free before core1020proper;fixed extra controls afterwards,all extra free before extra all-mask proper;no adaptive configuration',
            C='Only inherit core C-draft outcomes after bitwise-zero-input identity+same sampler key;not new inference',
            unsupported_policy='All32foreign records retained;ineligible explicitly missing,never substituted or silently dropped',
            extra_proper='One all-mask panel per eligible A extra;core1020 panels unchanged',
            source_sha256={name:sha(ROOT/name) for name in set(closure('scripts/run_masked_native_controls.py'))|{'tests/test_masked_native_controls.py','tests/test_masked_native_events.py'}})
        write(EXTRA,meta)
        for name in meta['source_sha256']:
            dest=ART/'source/matched-control-addendum'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    finally:
        signal.alarm(0);write(ART/'matched-control-preparation-runtime.json',dict(telemetry=guard.close(),wall_s=time.monotonic()-started))
    print(dict(status='matched_controls_frozen',records=len(records),eligible=meta['eligible'],wall_s=duration),flush=True)


def extra_inputs(record,panel,row):
    rr=inventory_row(panel,row(record['index'],'draft'));x,t=masked_batch(rr,'mps')
    if record['task']=='matched_local_roll':x=same_song_local_roll(x)
    else:
        dr=inventory_row(panel,row(record['donor'],'draft'));dx,_=masked_batch(dr,'mps')
        x,cov=core.original.audio_variant(x,dx,scale=1.)
        if not all(float(v)>=1.-1e-7 for v in cov.values()):raise ValueError('Matched support drift')
    if core.original.input_digest(x)!=record['public_input_sha256']:raise ValueError('Registered extra input drift')
    if core.original.input_digest(core.original.zero_audio(x))!=record['core_zero_sha256']:raise ValueError('Zero-input certificate drift')
    return rr,x,t


@torch.no_grad()
def extra_evaluate(cap):
    core.runtime();plan=core.read(core.PLAN);extra=core.read(EXTRA)
    for name,h in extra['source_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Addendum source drift')
    guard=ManagedMPS(ROOT,ART,cap);report=dict(status='running',updates=0,optimizer_updates=0,cloud_calls=0,free_completed=0,conditional_completed=0)
    def stop(*_):raise TimeoutError('Matched-control evaluation cap')
    signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(cap)-2))
    try:
        panel,row=core.load_data();priors=core.read(ROOT/plan['priors_path']);model=core.load_quality(plan,'A');before=weight_digest(model)
        original_C={r['index']:r for r in core.read(ART/'C-draft.json')};outcomes=deepcopy(extra['records'])
        for rec in outcomes:
            if not rec['eligible']:continue
            guard.check('matched_free');rr,x,t=extra_inputs(rec,panel,row);model.query_coverage={}
            prediction=decode_masked(model,x,rr['request'],t['layout'],priors['duration'],seed=plan['seed'],public_id=(rec['index'],rr['request'].start_us,rr['request'].end_us),deadline=guard.start+guard.wall_s,clear_callback=guard.clear)
            events=[RhythmEvent(**v) for v in prediction['events']];scores,partial=score_outcome(core.ENGINE,events,rr['targets'],prediction['complete'])
            source=original_C[rec['index']]
            if source['input_sha256']!=rec['core_zero_sha256']:raise ValueError('Inherited C source input mismatch')
            rec.update(status='completed' if prediction['complete'] else 'decode_failed',prediction=prediction,target=[asdict(e) for e in rr['targets']],request=asdict(rr['request']),scores=scores,raw_partial_scores=partial,
                query_coverage_free=deepcopy(model.query_coverage),C_inherited=dict(source_path='artifacts/'+core.ID+'/C-draft.json',source_sha256=sha(ART/'C-draft.json'),source_index=rec['index'],prediction=source['prediction'],scores=source['scores'],input_identity=rec['core_zero_sha256']))
            report['free_completed']+=1;write(ART/'matched-controls-outcomes.json',outcomes);del rr,x,t,prediction,events;guard.clear()
        for rec in outcomes:
            if not rec['eligible']:continue
            guard.check('matched_proper');rr,x,t=extra_inputs(rec,panel,row);model.query_coverage={};memory=model.encode(x,t['layout'],rr['request'].start_us);out=model(memory,empty_state())
            rec.update(proper_panel=proper_record(model,out,rr['request'],t,empty_state(),np.ones(256,bool),1.,priors),query_coverage_teacher=deepcopy(model.query_coverage))
            report['conditional_completed']+=1;write(ART/'matched-controls-outcomes.json',outcomes);del rr,x,t,memory,out;guard.clear()
        if weight_digest(model)!=before:raise ValueError('Addendum changed weights')
        report.update(status='matched_controls_completed',eligible=extra['eligible'],all_target_records=extra['all_target_records'],ineligible=sum(not r['eligible'] for r in outcomes),new_C_forwards=0,
            scope='Additional fixed support controls;all64recorddenominators retained,foreignineligible explicit;main340/1020unchanged')
    except BaseException as exc:report.update(status='execution_failed_pending_mechanism_review',failure=dict(type=type(exc).__name__,error=str(exc),phase=guard.phase))
    finally:
        signal.alarm(0);report.update(telemetry=guard.close(),wall_s=time.monotonic()-guard.start);write(ART/'matched-controls-evaluation.json',report)
    print({k:report.get(k) for k in ('status','free_completed','conditional_completed','wall_s','failure')},flush=True)


def run():
    if core.REPORT.exists():raise FileExistsError('No duplicate lifecycle')
    plan=core.read(core.PLAN);extra=core.read(EXTRA);prep=core.read(ART/'preparation.json')
    if sha(core.PLAN)!=extra['core_plan_sha256']:raise ValueError('Core plan identity')
    for name,h in extra['source_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Frozen complete entry source drift')
    prep_s=prep['wall_s']+extra['wall_s'];started=time.monotonic();health_s=0;eval_s=0
    env={**os.environ,'PYTHONPATH':str(ROOT/plan['runtime_target'])+os.pathsep+str(ROOT)}
    report=dict(id=core.ID,status='registered',stages={},preparation_wall_s=prep_s,optimizer_updates=0,cloud_calls=0,cloud_spend_usd=0,confirmation_scored=False,scaling_authorized=False,matched_control_plan_sha256=sha(EXTRA))
    try:
        for stage in ('admission','health-A','health-C','quality-A','quality-C','evaluation','matched-controls'):
            remaining=9000-prep_s-(time.monotonic()-started)
            cap=min(1200-health_s if stage=='admission' or stage.startswith('health') else 1200-eval_s if stage in ('evaluation','matched-controls') else 3000,remaining)
            if cap<=2:raise TimeoutError('Complete lifecycle wall cap')
            if admission(ROOT,GIB,15*GIB)['status']!='admitted':raise MemoryError('Storage admission')
            report.update(status=stage+'_running',active_stage=stage);publish(report)
            module='scripts.run_masked_native_controls' if stage=='matched-controls' else 'scripts.run_masked_native_events'
            args=['extra',str(int(cap))] if stage=='matched-controls' else ['worker',stage,str(int(cap))]
            with (ART/f'{stage}.log').open('w') as log:subprocess.run([str(ROOT/'.venv/bin/python'),'-u','-m',module,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=cap,check=True)
            r=core.read(ART/('matched-controls-evaluation.json' if stage=='matched-controls' else stage+'.json'));report['stages'][stage]={k:r.get(k) for k in ('status','updates','wall_s','failure')};report['optimizer_updates']+=r.get('updates',0)
            if stage=='admission' or stage.startswith('health'):health_s+=r['wall_s']
            if stage in ('evaluation','matched-controls'):eval_s+=r['wall_s']
            expected='admission_passed' if stage=='admission' else 'health_passed' if stage.startswith('health') else 'matched_controls_completed' if stage=='matched-controls' else 'terminal_pending_mechanism_review' if stage=='evaluation' else 'quality_completed'
            if r['status']!=expected:report.update(status=r['status'],failed_stage=stage);break
            if stage=='admission':
                costs=r['actual_cost_probe'];estimate=costs['full_panel_linear_estimate_s']+extra['eligible']*(costs['free_s']+costs['proper_three_rates_s']/3)
                report['complete_evaluation_cost_estimate_s']=estimate
                if estimate>1200:raise TimeoutError('Main+matched controls estimate exceeds1200s')
        else:report.update(status='terminal_pending_mechanism_review',research=core.read(ART/'evaluation.json')['research'],matched_controls=core.read(ART/'matched-controls-evaluation.json'))
    except BaseException as exc:report.update(status='supervisor_failed_pending_mechanism_review',failure=dict(type=type(exc).__name__,error=str(exc)))
    finally:report['wall_s']=prep_s+(time.monotonic()-started);publish(report)
    print({k:report.get(k) for k in ('status','failed_stage','optimizer_updates','wall_s','failure')},flush=True)


def publish(report):
    core.publish(report)
    if report.get('active_stage')=='matched-controls':
        path=ROOT/'reports/local-integration-v1/state.json';state=core.read(path)
        state['progress_file']=str((ART/'matched-controls-evaluation.json').relative_to(ROOT));write(path,state)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='extra':extra_evaluate(int(sys.argv[2]))
    else:globals()[sys.argv[1]]()
