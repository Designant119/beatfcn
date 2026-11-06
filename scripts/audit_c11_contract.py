"""Zero-update C11 contract checks, independent of any chosen successor model."""
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

from beatfcn.train.local_request_panel import event_request
from beatfcn.train.native_joint_support import NativeJointSupport, KINDS
from beatfcn.train.reference_completion import partition
from scripts.run_local_evolution import load_data
from scripts.run_local_portfolio_scale import sha, write


ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'reports/council-integration-c11'
REPORT=FOLDER/'parent-contract-audit.json'


def main():
    if REPORT.exists():raise FileExistsError('Preserve actual audit receipt')
    started=time.monotonic();manifest=json.loads((FOLDER/'manifest.json').read_text())
    for name,info in manifest.items():
        b=(ROOT/info['frozen']).read_bytes()
        if len(b)!=info['bytes'] or hashlib.sha256(b).hexdigest()!=info['sha256']:
            raise ValueError('Frozen object drift '+name)
    raw='artifacts/joint-native-set-v1a/health-predictions.json'
    rows=json.loads((ROOT/raw).read_text());collisions=[]
    for r in rows:
        pred=r['prediction'];active=[a for a,k in zip(pred['selected_addresses'],pred['query_classes']) if k>0]
        collisions.append(dict(index=r['index'],active_queries=len(active),unique_selected_addresses=len(set(active)),
            max_query_address_multiplicity=max(Counter(active).values(),default=0),
            max_emitted_head_multiplicity=max(Counter(e['onset_us'] for e in pred['events']).values(),default=0)))
    # A valid labeled-tuple mode need not be an unordered-multiset mode.
    pA,pB=.4,.35
    modes=dict(labelled_AA=pA*pA,multiset_AA=pA*pA,multiset_AB=2*pA*pB)
    assert modes['multiset_AB']>modes['multiset_AA']
    # Count marginal MAP followed by tail conditional MAP is not whole-event MAP.
    hold_mode=dict(p_count0=.4,p_count1=.6,largest_tail_probability=.01,
        decoded_count_marginal_mode=1,largest_single_hold_event_probability=.006,empty_event_probability=.4)
    assert hold_mode['largest_single_hold_event_probability']<hold_mode['empty_event_probability']
    # Balanced CE changes natural class frequencies even for an audio-free model.
    rho=.01
    ce=lambda p,rate: -rate*math.log(p)-(1-rate)*math.log1p(-p)
    balanced=dict(true_positive_rate=rho,natural_optimum_p1=rho,balanced_optimum_p1=.5,
        natural_ce_at_natural_optimum=ce(rho,rho),natural_ce_at_balanced_optimum=ce(.5,rho),
        balanced_ce_at_natural_optimum=ce(rho,.5),balanced_ce_at_balanced_optimum=ce(.5,.5),
        valid_addresses_example=8000,hard_count_if_p1_below_half=0,hard_count_if_p1_above_half=8000)
    assert balanced['balanced_ce_at_balanced_optimum']<balanced['balanced_ce_at_natural_optimum']
    # Actual target replay avoids feature/model calls. Check against every frozen
    # training target SHA rather than trusting equivalent-looking partition code.
    panel,_=load_data();plan=json.loads((ROOT/'bench/rhythm-v2/joint-native-set-v1a.json').read_text())
    data=json.loads((ROOT/'artifacts/joint-native-set-v1a/data-identity.json').read_text())
    support=NativeJointSupport();hist={k:Counter() for k in KINDS};max_by={k:0 for k in KINDS}
    positive_counts=Counter();all_addresses=0;different_tails=0;different_tail_examples=[];checked=0
    for batch in plan['sample_stream']:
        for spec in batch:
            rec=panel.records[spec['index']];events=panel.targets(spec['index']);request=event_request(rec['request'])
            if spec['view'] in ('middle','continuation','random_groups'):
                request,_,events,_=partition(events,request,spec['view'],f"{rec['chart_id']}:{request.start_us}",20261027)
            digest=hashlib.sha256(json.dumps([asdict(e) for e in sorted(events)],sort_keys=True).encode()).hexdigest()
            prior=data['checks'][checked]
            if any(prior[k]!=spec[k] for k in ('index','view','field_index')) or prior['target_sha256']!=digest:
                raise ValueError('Target replay identity mismatch')
            _,allowed=support.addresses(request);n=int(allowed.sum());all_addresses+=n
            counts=Counter((e.onset_us,e.type) for e in events);tails=defaultdict(list)
            for (head,kind),count in counts.items():
                hist[kind][count]+=1;positive_counts[kind]+=1;max_by[kind]=max(max_by[kind],count)
            for kind in KINDS:hist[kind][0]+=n-sum(k==kind for _,k in counts)
            for e in events:
                if e.type=='hold':tails[e.onset_us].append(e.end_us//1000-e.onset_us//1000)
            for head,values in tails.items():
                if len(set(values))>1:
                    different_tails+=1
                    if len(different_tail_examples)<8:different_tail_examples.append(dict(**spec,head_us=head,deltas_ms=values))
            checked+=1
            if time.monotonic()-started>180:raise TimeoutError('Shared label audit cap')
    result=dict(id='c11-parent-contract-audit-v1',status='completed',frozen_objects=len(manifest),
        frozen_manifest_sha256=sha(FOLDER/'manifest.json'),source_sha256=sha(Path(__file__)),
        health_query_collision_checks=collisions,labelled_vs_multiset_mode_counterexample=modes,
        count_then_tail_not_joint_mode_counterexample=hold_mode,balanced_count_calibration_counterexample=balanced,
        absent_type_negative_gradient=dict(softmax_classes=5,uniform_p=.2,target='absent',hold_logit_gradient=.2,
            interpretation='Unmatched absent CE does penalize a false hold logit. Absence of head supervision is separate.'),
        training_replay_requests=checked,target_sha_checks_passed=True,max_per_address_kind=max_by,
        all_valid_addresses=all_addresses,count_histograms={k:dict(sorted(v.items())) for k,v in hist.items()},
        positive_address_fraction={k:positive_counts[k]/all_addresses for k in KINDS},
        same_head_hold_distinct_tail_exposures=different_tails,different_tail_examples=different_tail_examples,
        wall_s=time.monotonic()-started,neural_calls=0,optimizer_updates=0,cloud_calls=0,
        scope='Counterexamples and exact actual target support only;not a tested successor or diagnosis of unique neural cause')
    write(REPORT,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('count_histograms','different_tail_examples')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
