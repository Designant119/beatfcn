"""Terminal evidence extraction; mechanism judgment is a separate parent review."""
from collections import Counter,defaultdict
import json
import math
from pathlib import Path

import numpy as np

from scripts.run_local_portfolio_scale import sha,write
from beatfcn.train.field_integrity import validate
from beatfcn.train.native_trial_evaluation import ORDER,KINDS,summarize_teacher


ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/c13-native-quality-v1'
DEST=ROOT/'reports/local-integration-v1/c13-native-quality-terminal-facts.json'


def read(path):return json.loads(path.read_text())


def average(values):return float(np.mean(values)) if values else None


def field_facts(path,prior):
    r=read(path);certificate=validate(r);base=r['arms']['base'];arms={}
    for name,rows in r['arms'].items():
        type_effect=[];head_effect=[];tail_effect=[];binary={k:[] for k in KINDS};support=Counter();coverage=[]
        for i,a in enumerate(rows):
            b=base[i];record=r['tape'][i]
            if a.get('head_support_failure'):support['head']+=1
            if a.get('type_support_failure'):support['type']+=1
            end=a.get('endpoint_true_path',{});original_end=b.get('endpoint_true_path',{})
            if end.get('support_failure'):support['endpoint']+=1
            if a.get('head_nll') is not None and b.get('head_nll') is not None:head_effect.append(a['head_nll']-b['head_nll'])
            if a.get('type_nll') is not None and b.get('type_nll') is not None and not a['forced']:type_effect.append(a['type_nll']-b['type_nll'])
            if end.get('nll') is not None and original_end.get('nll') is not None:tail_effect.append(end['nll']-original_end['nll'])
            if a.get('true_type') is not None and not a['forced']:
                mask=np.asarray(record['kind_allowed'],bool);p0=np.asarray(prior,float)*mask;p0/=p0.sum()
                for k,kind in enumerate(KINDS):
                    if not mask[k]:continue
                    truth=a['true_type']==k;value=a['type_probability'][k]
                    if value in (0,1) and ((value==0 and truth) or (value==1 and not truth)):
                        binary[kind].append(dict(positive=truth,status='probability_underflow_or_saturation',gain=None));continue
                    nll=-math.log(max(value if truth else 1-value,1e-300));prior_nll=-math.log(max(p0[k] if truth else 1-p0[k],1e-300))
                    binary[kind].append(dict(positive=truth,status='finite',gain=prior_nll-nll))
            coverage.append(a.get('coverage_paths',{}))
        arms[name]=dict(members=len(rows),unique_matched=sum(a['matched_unique_head'] for a in rows),
            forced=sum(a['forced'] for a in rows),support_failures=dict(support),head_pairs=len(head_effect),head_nll_minus_base=average(head_effect),
            type_pairs=len(type_effect),type_nll_minus_base=average(type_effect),endpoint_pairs=len(tail_effect),endpoint_nll_minus_base=average(tail_effect),
            type_binary={kind:dict(decisions=len(values),positive=sum(v['positive'] for v in values),negative=sum(not v['positive'] for v in values),
                underflow=sum(v['gain'] is None for v in values),gain=average([v['gain'] for v in values if v['gain'] is not None]),
                positive_gain=average([v['gain'] for v in values if v['positive'] and v['gain'] is not None]),
                negative_gain=average([v['gain'] for v in values if not v['positive'] and v['gain'] is not None])) for kind,values in binary.items()},
            coverage_paths=coverage)
    return dict(index=r['index'],certificate=certificate,source_coverage=r.get('source_coverage'),arms=arms,
        unique_head_pairs=r['unique_head_pairs'],ambiguous_or_unmatched=r['ambiguous_or_unmatched'],
        scope='Unique machine-head subset/forced and unsupported counted explicitly;not all-target or free quality')


def main():
    if DEST.exists():raise FileExistsError('Preserve terminal facts;new version for a new analysis')
    top_path=ROOT/'reports/local-integration-v1/native-quality-repaired.json';top=read(top_path)
    if top['status'].endswith('_running') or top['status'] in ('running','registered'):raise ValueError('Terminal only;no interim dev selection')
    identity={str(top_path.relative_to(ROOT)):sha(top_path)}
    result=dict(status='facts_ready_pending_mechanism_review',candidate='c13-native-quality-v1',terminal=top,stages={},tasks={},fields=[],
        diagnosis_scope='Evidence extraction only;no automatic failure cause,closure,retry,seed or scaling decision')
    for stage in ('preflight','quality','evaluation'):
        path=ART/f'{stage}.json'
        if not path.exists():continue
        r=read(path);identity[str(path.relative_to(ROOT))]=sha(path)
        result['stages'][stage]={k:r.get(k) for k in ('status','updates','failure','wall_s','runtime_prefix','accumulation_phase','applied_exposure','seen_types',
            'initial_weight_sha256','checkpoint_sha256','research_checks','research_qualified','quality_gate_status','quality_gate_checks','non_tap_evidence','acoustic_pairs')}
        tp=ART/f'{stage}-telemetry.json'
        if tp.exists():
            t=read(tp);result['stages'][stage]['telemetry']={k:t.get(k) for k in ('peak_sampled_physical_bytes','error')};result['stages'][stage]['telemetry']['samples']=len(t['samples']);identity[str(tp.relative_to(ROOT))]=sha(tp)
    for task in ORDER:
        path=ART/f'{task}.json'
        if not path.exists():continue
        rows=read(path);windows=[]
        for r in rows:
            events=r['prediction']['events'];heads=sorted({e['onset_us'] for e in events});gap=np.diff(heads)
            windows.append(dict(index=r['index'],component=r['component'],status=r['status'],complete=r['prediction']['complete'],failure=r['prediction'].get('failure'),
                full50=r['scores']['full_note_f1'],onset10=r['scores']['onset_10ms']['f1'],raw_partial_full50=r['raw_partial_scores']['full_note_f1'],
                pred=len(events),true=len(r['target']),types=dict(Counter(e['type'] for e in events)),target_types=dict(Counter(e['type'] for e in r['target'])),
                duplicate_extras=len(events)-len({(e['onset_us'],e['end_us'],e['type']) for e in events}),
                positive_gap_median_ms=average(gap/1000) if len(gap)==1 else float(np.median(gap)/1000) if len(gap) else None,
                positive_gap_under10_fraction=float(np.mean(gap<10000)) if len(gap) else None,
                control=r['control'],endpoint_min_cost=r.get('endpoint_min_cost'),teacher_status=r.get('teacher_status','not_attempted'),
                teacher_factor_nll=r.get('teacher_scores',{}).get('exact_factor_nll'),teacher_failures=r.get('teacher_scores',{}).get('numerical_or_support_failures')))
        result['tasks'][task]=dict(expected=len(rows),attempted=sum(r['status']!='not_attempted' for r in rows),complete=sum(r['prediction']['complete'] for r in rows),
            teacher_completed=sum(r.get('teacher_status')=='completed' for r in rows),windows=windows,
            teacher_component_proper=summarize_teacher(rows),scope='Fixed outcome denominator;raw partial quality is diagnostic only')
        identity[str(path.relative_to(ROOT))]=sha(path)
    prior=read(ART/'data-identity.json')['prior'] if (ART/'data-identity.json').exists() else None
    if prior:
        for path in sorted(ART.glob('quality-p2-*.json')):
            result['fields'].append(field_facts(path,prior));identity[str(path.relative_to(ROOT))]=sha(path)
    result['identity']=identity;write(DEST,result)
    print(json.dumps(dict(status=result['status'],terminal=top['status'],tasks=list(result['tasks']),fields=len(result['fields']))))


if __name__=='__main__':main()
