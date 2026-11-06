"""Same frozen v1 tensors/stream/recipe,strict1s ALL-event density intervention.

Reuse the v1 engine unchanged; configure paths/filter and a prospectively matched
evaluation only in this process. Historical v1 source and artifacts stay intact.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import time

import numpy as np
import torch

from beatfcn.train.event_density_drag_filter import EventDensityFilteredPanel,RULE,filtered_request_row
from beatfcn.train.masked_native_events import empty_state,target_state,decode_masked
from scripts import run_drag_data_probe as core
from scripts.run_local_evolution import load_data
from scripts.run_native_cell_chain import closure

ROOT=core.ROOT
ID='data-drag-density-probe-v2'
ART=ROOT/'artifacts'/ID
PLAN=ROOT/'bench/rhythm-v2'/(ID+'.json')
REPORT=ROOT/'reports/local-integration-v1'/(ID+'.json')


def configure():
    core.ID=ID;core.ART=ART;core.PLAN=PLAN;core.REPORT=REPORT
    core.DragFilteredPanel=EventDensityFilteredPanel
    core.filtered_request_row=filtered_request_row
    core.evaluate=evaluate


def prepare():
    if PLAN.exists():raise FileExistsError('Preserve v2 freeze')
    ART.mkdir(parents=True,exist_ok=True);started=time.monotonic();core.runtime()
    previous_path=ROOT/'bench/rhythm-v2/data-drag-density-probe-v1.json';previous=core.read(previous_path)
    for name,h in previous['source_sha256'].items():
        if core.sha(ROOT/name)!=h:raise ValueError('V1 scientific source drift '+name)
    panel,row=load_data();filtered=EventDensityFilteredPanel(panel)
    p=deepcopy(previous);p.update(id=ID,parent_plan=str(previous_path.relative_to(ROOT)),parent_plan_sha256=core.sha(previous_path),
        rule=RULE,predecessors=['data-drag-density-probe-v1'],
        sampling='Exact v1 train128/fit4/stream/seed reused;newaffected outside128 audit only;no resampling based on new threshold or dev results',
        intervention='Only density predicate changed:ALL onset objects>12 in half-open1s;remove drag only. Same raw baseline recipe and initial tensors as v1.',
        controls='PRIMARY both arms on identical v2filtereddev target. SECONDARY both on original dev. Proper commonfiltered anchors plus original anchor allmask;raw-arm machine context shared;true context and audiozero predeclared.',
        evaluation_protocol=dict(primary='common v2filtered target for both arms',secondary='common original target for both arms',
            generation='No outputs filtered;public requests and head supports unchanged',
            proper='Allmask/commonrawmachine/trueodd/audiozero on commonv2filteredanchors;allmask also on commonoriginalanchors',
            changed_development_labels='Allowed by Owner data steer;never compare own-regime scores as training benefit'),
        source_audit='reports/local-integration-v1/data-drag-density-audit-v2.json')
    p['priors']['raw']=deepcopy(previous['priors']['raw'])
    counts=np.full(4,.5);seen=set();events=Counter()
    for i in p['train_indices']:
        occurrence=Counter()
        for e in filtered.targets(i):
            sig=(panel.records[i]['chart_id'],e.onset_us,e.end_us,e.type,e.lane);ordinal=occurrence[sig];occurrence[sig]+=1
            if (*sig,ordinal) not in seen:counts[core.KINDS.index(e.type)]+=1;seen.add((*sig,ordinal))
            events[e.type]+=1
    p['priors']['filtered']=dict(probability=(counts/counts.sum()).tolist(),counts=counts.tolist(),events=dict(events),unique_events=len(seen))
    used=sorted(set(p['train_indices']+p['fit_indices']+p['eval_indices']+p['empty_indices']));inputs={};targets={};views=[]
    for i in used:
        if time.monotonic()-started>300:raise TimeoutError('Prepare300s')
        rr,x,t=core.batch(panel,row,filtered,i,'raw','cpu');fr,fx,ft=core.batch(panel,row,filtered,i,'filtered','cpu')
        if any(not torch.equal(x[k],fx[k]) for k in x):raise ValueError('Public inputs changed')
        if core.original.input_digest(x)!=previous['input_digest'][str(i)]:raise ValueError('V1 public input drift')
        if not np.array_equal(t['layout']['heads'],ft['layout']['heads']):raise ValueError('Head support drift')
        if Counter(e for e in rr['targets'] if e.type!='drag')!=Counter(e for e in fr['targets'] if e.type!='drag'):raise ValueError('Non-drag changed')
        inputs[str(i)]=core.original.input_digest(x)
        targets[str(i)]={a:hashlib.sha256(repr([(e.onset_us,e.end_us,e.type,e.lane) for e in r['targets']]).encode()).hexdigest() for a,r in [('raw',rr),('filtered',fr)]}
        del rr,x,t,fr,fx,ft
    historical=core.read(ROOT/'bench/rhythm-v2/c17-masked-native-events-v1.json')
    for s in [s for b in historical['sample_stream'][:16] for s in b]:
        rr=row(s['index'],s['view'],s['field_index']);fr=filtered_request_row(filtered,rr)
        if Counter(e for e in rr['targets'] if e.type!='drag')!=Counter(e for e in fr['targets'] if e.type!='drag'):raise ValueError('Partition changed non-drag')
        views.append(dict(index=s['index'],view=s['view'],request_equal=rr['request']==fr['request'],before=len(rr['targets']),after=len(fr['targets'])))
    core.write(ART/'pipeline-preflight.json',dict(status='completed',public_inputs_exact=len(used),v1_raw_inputs_exact=True,
        non_drag_targets_exact=True,public_support_exact=True,original_partitions=True,views=views,optimizer_updates=0))
    if core.sha(ROOT/p['initial_path'])!=previous['initial_sha256']:raise ValueError('V1 initial tensor file changed')
    p.update(input_digest=inputs,target_digest=targets,pretraining_probe_receipts=[
        dict(status='completed',evidence='artifacts/data-drag-density-v2-tests.log',scope='Strict12/13,alltypes,halfopen,duplicate,bruteforceunion regression'),
        dict(status='completed',evidence=str((ART/'pipeline-preflight.json').relative_to(ROOT)),scope='Actual newfilter/V1publicinputs/support/originalpartitions')])
    p['source_sha256']={name:core.sha(ROOT/name) for name in set(closure('scripts/run_drag_data_probe_v2.py'))|{'tests/test_event_density_drag_filter.py'}}
    core.write(PLAN,p)
    for name in p['source_sha256']:
        dest=ART/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    core.write(ART/'preparation.json',dict(status='completed',wall_s=time.monotonic()-started,parameters=p['parameters'],
        initial_tensor_digest=p['initial_tensor_digest'],same_exact_v1_stream=True,optimizer_updates=0))
    print(dict(status='prepared',parameters=p['parameters'],wall_s=time.monotonic()-started,filtered_prior=p['priors']['filtered']),flush=True)


def mask_even(state):
    selected=np.arange(256)%2==0
    result={k:v.copy() for k,v in state.items()};result['status'][selected]=0
    for k in ('head','kind','end'):result[k][selected]=0
    return result,selected


def evaluate(net,plan,panel,row,filtered,arm,guard):
    records=[];raw={r['index']:r for r in core.read(ART/'raw-outcomes.json')} if arm=='filtered' else {}
    for i in plan['eval_indices']+plan['empty_indices']:
        guard.check('free-'+arm);rr,x,t=core.batch(panel,row,filtered,i,arm)
        prediction=decode_masked(net,x,rr['request'],t['layout'],plan['duration_prior'],seed=plan['seed'],
            public_id=(i,rr['request'].start_us,rr['request'].end_us),deadline=guard.start+guard.wall_s,clear_callback=guard.clear)
        if not prediction['complete']:raise ValueError('Sampler failed '+str(prediction['failure']))
        events=[core.RhythmEvent(**e) for e in prediction['events']];truth=panel.targets(i);clean=filtered.targets(i)
        records.append(dict(index=i,component=panel.records[i]['component'],empty=i in plan['empty_indices'],prediction=prediction,
            target_raw=[asdict(e) for e in truth],target_filtered=[asdict(e) for e in clean],scores_raw=core.score(events,truth),
            scores_filtered=core.score(events,clean),hold_raw=core.hold_errors(events,truth),hold_filtered=core.hold_errors(events,clean)))
        core.write(ART/f'{arm}-outcomes.json',records);del rr,x,t,events;guard.clear()
    for record in records:
        i=record['index'];guard.check('matched-proper-'+arm)
        # Supervision regime is COMMON across arms,independent of training arm.
        rr,x,t=core.batch(panel,row,filtered,i,'filtered')
        record['all_mask']=core.proper(net,x,rr,t,plan,empty_state(),np.ones(256,bool));guard.clear()
        baseline=record if arm=='raw' else raw[i]
        machine=[core.RhythmEvent(**e) for e in baseline['prediction']['events']]
        context,selected=mask_even(target_state(machine,rr['request'],t['layout']))
        record['common_machine_context']=core.proper(net,x,rr,t,plan,context,selected);guard.clear()
        teacher,selected=mask_even(t['state']);record['true_context']=core.proper(net,x,rr,t,plan,teacher,selected);guard.clear()
        record['audio_zero']=core.proper(net,core.original.zero_audio(x),rr,t,plan,empty_state(),np.ones(256,bool));guard.clear()
        raw_rr,raw_x,raw_t=core.batch(panel,row,filtered,i,'raw')
        record['raw_anchor_all_mask']=core.proper(net,raw_x,raw_rr,raw_t,plan,empty_state(),np.ones(256,bool))
        record['proper_target_regime']='common v2filtered for both arms;raw_anchor_all_mask is secondary'
        core.write(ART/f'{arm}-outcomes.json',records);del rr,x,t,raw_rr,raw_x,raw_t;guard.clear()
    return records


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run'));args=parser.parse_args()
    configure();prepare() if args.action=='prepare' else core.run()
