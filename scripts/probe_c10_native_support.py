"""Bounded real-data audit for shared C10 representation, no neural calls."""
from collections import Counter
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np

from beatfcn.train.native_joint_support import NativeJointSupport,KINDS
from beatfcn.train.joint_reference import joint_reference_inputs
from beatfcn.train.reference_completion import partition
from beatfcn.train.local_request_panel import event_request
from scripts.run_c07_count_ancestral import specs
from scripts.run_local_evolution import load_data
from scripts.run_local_portfolio_scale import sha,write
from scripts.check_experiment_storage import admission,GIB


ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'bench/rhythm-v2/c10-native-support-probe-v1.json'
FOLDER=ROOT/'reports/local-integration-v1'
REPORT=FOLDER/'native-support-probe.json'
ART=ROOT/'artifacts/c10-native-support-probe-v1'


def read(path):return json.loads((ROOT/path).read_text())


def prepare():
    if PLAN.exists():raise FileExistsError('Keep original registered probe')
    files=('scripts/probe_c10_native_support.py','beatfcn/train/native_joint_support.py',
        'beatfcn/train/joint_reference.py','tests/test_native_joint_support.py','tests/test_joint_reference.py',
        'scripts/run_local_evolution.py','scripts/run_local_portfolio_scale.py',
        'scripts/run_frontend_canvas_comparison.py','beatfcn/train/local_bundle.py',
        'beatfcn/train/reference_completion.py','beatfcn/train/event_reference.py',
        'beatfcn/model/full_event_codec.py','beatfcn/model/local_song_rollout.py')
    p=dict(id='c10-native-support-probe-v1',scope='Common representation/reference preparation before C10 synthesis;not a selected neural architecture or quality trial',
        training_plan='bench/rhythm-v2/c08-no-mark-history-v1.json',training_plan_sha256=sha(ROOT/'bench/rhythm-v2/c08-no-mark-history-v1.json'),
        support=dict(cell_us=40000,max_cells=200,max_count=64,fractional_copies=16),
        wall_s=180,new_disk_peak_bytes=32*1024**2,reserve_bytes=15*GIB,
        sources={f:sha(ROOT/f) for f in files},gate='All2000original training specs and original74evaluation requests preserve exact native multiset;capacity and legal grammar pass;known empty and sparse reference semantics verified separately.',
        optimizer_updates=0,neural_calls=0,cloud_calls=0,execution_ready_for_training=False)
    write(PLAN,p)
    for f in files:
        dest=ART/'source'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
    shutil.copy2(PLAN,ART/'source/frozen-plan.json')


def signature(events):return Counter((e.onset_us,e.end_us,e.type,e.lane) for e in events)


def run():
    if REPORT.exists():raise FileExistsError('No automatic probe rerun')
    started=time.monotonic();p=read(str(PLAN.relative_to(ROOT)));checks=[];failures=[]
    disk=admission(ROOT,p['new_disk_peak_bytes'],p['reserve_bytes']);assert disk['status']=='admitted'
    for name,h in p['sources'].items():
        if sha(ROOT/name)!=h:raise ValueError('Probe source drift '+name)
    if sha(ROOT/p['training_plan'])!=p['training_plan_sha256']:raise ValueError('Training stream identity drift')
    base=read('bench/rhythm-v2/local-frontend-comparison-v2.json');cache_start=time.monotonic()
    for name,h in base['mert_cache_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Feature cache drift '+name)
    panel,row=load_data();train=read(p['training_plan']);codec=NativeJointSupport(**p['support'])
    exposures=Counter();count_hist=Counter();known_empty=[];cross_hold=0;extra_duplicates=0;small_gaps=0;positive_gaps=0
    max_count=max_copies=max_reference=0;minimum_positive=None
    def check(index,view,field,split,bands=None):
        nonlocal cross_hold,extra_duplicates,small_gaps,positive_gaps,max_count,max_copies,max_reference,minimum_positive
        if time.monotonic()-started>p['wall_s']:raise TimeoutError('Data audit wall cap')
        rr=row(index,view,field,bands);request=rr['request'];events=rr['targets']
        encoded=codec.encode(events,request);decoded=codec.decode(encoded,request)
        if signature(events)!=signature(decoded):raise ValueError('Native multiset mismatch')
        addresses,valid=codec.addresses(request);max_count=max(max_count,int(encoded.counts.max(initial=0)))
        max_copies=max(max_copies,int(valid[:,40:].sum(-1).max(initial=0)))
        if split=='train':exposures.update(e.type for e in events);count_hist.update(map(int,encoded.counts))
        cross_hold+=sum(e.type=='hold' and e.end_us>request.end_us for e in events)
        extra_duplicates+=len(events)-len(signature(events))
        gaps=np.diff(sorted({e.onset_us for e in events}));gaps=gaps[gaps>0]
        positive_gaps+=len(gaps);small_gaps+=int((gaps<10000).sum())
        if len(gaps):minimum_positive=min(minimum_positive or int(gaps.min()),int(gaps.min()))
        regions=()
        if view in ('middle','continuation','random_groups'):
            record=panel.records[index];orig=event_request(record['request'])
            req,visible,hidden,regions=partition(panel.targets(index),orig,view,f"{record['chart_id']}:{orig.start_us}",20261027)
            if signature(hidden)!=signature(events) or signature(visible)!=signature(rr['visible']) or (req.start_us,req.end_us)!=(request.start_us,request.end_us):raise ValueError('Reference partition replay mismatch')
        ref,provenance=joint_reference_inputs(request,panel.records[index]['audio_sha256'],rr['visible'],regions,
            source_id=f'joint-preflight:{index}:{view}:{field}',role=rr['source_role'])
        known=int(ref['reference'][:,14].sum());positive=int(ref['reference'][:,10].sum())
        if positive!=len(rr['visible']):raise ValueError('Visible multiplicity changed')
        if regions and known!=len(regions):raise ValueError('Complete known regions lost')
        if not regions and known:raise ValueError('Sparse positive reference invents silence')
        if regions and not rr['visible']:
            known_empty.append(dict(index=index,view=view,split=split,regions=regions,
                old_known_rows=int(rr['inputs']['reference'][:,14].sum()),new_known_rows=known))
        max_reference=max(max_reference,len(ref['reference']))
        payload=dict(counts=encoded.counts.tolist(),members=[asdict(v) for v in encoded.members])
        checks.append(dict(index=index,view=view,field=field,split=split,events=len(events),
            cells=len(encoded.counts),max_count=int(encoded.counts.max(initial=0)),reference_rows=len(ref['reference']),
            source_target_sha256=hashlib.sha256(json.dumps(sorted(signature(events).items()),separators=(',',':')).encode()).hexdigest(),
            encoded_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()))
    try:
        for batch in train['sample_stream']:
            for s in batch:check(s['index'],s['view'],s['field_index'],'train')
        for task,items in specs(train).items():
            for i,v,bands,masked in items:check(i,v,0,'evaluation:'+task,bands)
        expected=read('artifacts/c07-anchor-conditioned-mark-chain-v1/data-identity.json')['exposure_type_counts']
        if [exposures[k] for k in KINDS]!=expected:raise ValueError('Original training event exposure mismatch')
    except BaseException as exc:
        failures.append(dict(completed_requests=len(checks),error=f'{type(exc).__name__}: {exc}'))
    passed=not failures
    result=dict(id=p['id'],status='completed' if passed else 'failed',passed=passed,plan_sha256=sha(PLAN),
        requests=len(checks),training_requests=sum(v['split']=='train' for v in checks),
        exposure_type_counts={k:exposures[k] for k in KINDS},training_count_histogram=dict(sorted(count_hist.items())),
        max_cell_count=max_count,max_fractional_copies_per_cell=max_copies,max_reference_rows=max_reference,
        cross_core_hold_exposures=cross_hold,extra_identical_multiset_exposures=extra_duplicates,
        minimum_positive_interval_us=minimum_positive,positive_intervals=positive_gaps,positive_intervals_below10ms=small_gaps,
        known_empty_reference_cases=known_empty,checks=checks,failures=failures,
        disk_admission=disk,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        wall_s=time.monotonic()-started,cache_and_load_wall_s=cache_start-started,
        optimizer_updates=0,neural_calls=0,cloud_calls=0,learning_health_passed=None,
        boundaries='Supports exact representation/reference semantics only;no count/type/tail network,likelihood,CPU/MPS gradients,resource health or free-generation quality certified.')
    write(REPORT,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','known_empty_reference_cases')},ensure_ascii=False,indent=2))
    if not passed:raise SystemExit(2)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run'));args=parser.parse_args()
    {'prepare':prepare,'run':run}[args.action]()
