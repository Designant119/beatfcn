"""Read saved paired outputs and proper probabilities; no model execution."""
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ID='data-drag-density-probe-v1'
ART=ROOT/'artifacts'/ID
KINDS=('tap','hold','flick','drag')


def mean(values):return float(np.mean(values)) if values else None


def proper_summary(records,key,plan):
    probs=defaultdict(list);first=[];tails=[];heads=[];first_by_kind=defaultdict(list);negatives=defaultdict(list)
    for r in records:
        for value in r[key]['types']:
            k=value['truth'];p=value['probability'];support=np.array(value['support'],bool)
            prior={}
            for name in ('raw','filtered'):
                q=np.array(plan['priors'][name]['probability'])*support;q/=q.sum();prior[name]=q
            item=dict(model=-math.log(max(p[k],1e-300)),raw_prior=-math.log(prior['raw'][k]),
                filtered_prior=-math.log(prior['filtered'][k]),component=r['component'])
            for j,name in enumerate(KINDS):
                if j!=k and support[j]:negatives[name].append(dict(model=-math.log(max(1-p[j],1e-300)),
                    raw_prior=-math.log(max(1-prior['raw'][j],1e-300))))
            probs[KINDS[k]].append(item);heads.append(value['head_nll'])
            if value['slot']==0:first.append(item);first_by_kind[KINDS[k]].append(item)
        tails.extend(r[key]['tails'])
    by_type={}
    for k in KINDS:
        rows=probs[k];by_type[k]=dict(n=len(rows),model_nll=mean([v['model'] for v in rows]),
            raw_frequency_nll=mean([v['raw_prior'] for v in rows]),filtered_frequency_nll=mean([v['filtered_prior'] for v in rows]),
            excess_over_common_raw_prior=mean([v['model']-v['raw_prior'] for v in rows]),
            negative_n=len(negatives[k]),negative_model_nll=mean([v['model'] for v in negatives[k]]),
            negative_raw_prior_nll=mean([v['raw_prior'] for v in negatives[k]]))
    allrows=[v for rows in probs.values() for v in rows];components=defaultdict(list)
    for v in allrows:components[v['component']].append(v['model'])
    return dict(type_by_class=by_type,type_nll_micro=mean([v['model'] for v in allrows]),
        raw_frequency_nll_micro=mean([v['raw_prior'] for v in allrows]),
        type_nll_component_macro=mean([mean(v) for v in components.values()]),
        first_member=dict(n=len(first),model_nll=mean([v['model'] for v in first]),raw_frequency_nll=mean([v['raw_prior'] for v in first]),
            excess_over_common_raw_prior=mean([v['model']-v['raw_prior'] for v in first]),
            support_by_type={k:len(v) for k,v in first_by_kind.items()}),
        head_nll=mean(heads),hold_tail=dict(n=len(tails),model_nll=mean([v['nll'] for v in tails]),
            prior_nll=mean([v['prior_nll'] for v in tails]),excess_over_duration_prior=mean([v['nll']-v['prior_nll'] for v in tails])),
        support_failures=sum(r[key]['support_failures'] for r in records),
        scope='True-anchor conditional score only;first member all-mask has no event history;machine context uses raw-control generated history')


def summarize(records,plan):
    draft=[r for r in records if not r['empty']];empty=[r for r in records if r['empty']]
    per_type={k:dict(tp=sum(r['scores_raw']['per_type'][k]['tp'] for r in draft),
        pred=sum(r['scores_raw']['per_type'][k]['pred'] for r in draft),true=sum(r['scores_raw']['per_type'][k]['true'] for r in draft),
        components=sum(any(r['scores_raw']['per_type'][k]['tp']>0 for r in draft if r['component']==c) for c in {r['component'] for r in draft})) for k in KINDS}
    for v in per_type.values():v['f1']=2*v['tp']/max(v['pred']+v['true'],1)
    components=defaultdict(list)
    for r in draft:components[r['component']].append(r)
    tails=[v for r in draft for v in r['hold_raw']['tail_abs_ms']]
    allmask=proper_summary(draft,'all_mask',plan);machine=proper_summary(draft,'common_machine_context',plan)
    # Compare the same even-slot anchors;the other all-mask rows are excluded.
    even_records=[]
    for r in draft:
        ev={**r,'all_mask':{**r['all_mask'],
            'types':[v for v in r['all_mask']['types'] if v['slot']%2==0],
            'tails':[v for v in r['all_mask']['tails'] if v['slot']%2==0]}}
        even_records.append(ev)
    even=proper_summary(even_records,'all_mask',plan)
    occupancy=[r['all_mask']['occupancy'] for r in draft]
    return dict(draft_requests=len(draft),components=len(components),
        pred=sum(len(r['prediction']['events']) for r in draft),true=sum(len(r['target_raw']) for r in draft),
        free_empty_drafts=sum(not r['prediction']['events'] for r in draft),
        full50_component_macro=mean([mean([r['scores_raw']['full_note_f1'] for r in rr]) for rr in components.values()]),
        onset10_component_macro=mean([mean([r['scores_raw']['onset_10ms']['f1'] or 0 for r in rr]) for rr in components.values()]),
        per_type=per_type,empty_control=dict(requests=len(empty),pred=sum(len(r['prediction']['events']) for r in empty)),
        free_hold=dict(head_matches=len(tails),tail_median_ms=float(np.median(tails)) if tails else None,
            joint_50ms=sum(v<=50 for v in tails),tail_abs_ms=tails),
        all_mask=allmask,common_machine_context=machine,
        machine_context_penalty=dict(type_nll=machine['type_nll_micro']-even['type_nll_micro'],
            head_nll=machine['head_nll']-even['head_nll'],
            tail_nll=(machine['hold_tail']['model_nll']-even['hold_tail']['model_nll']) if machine['hold_tail']['n'] else None,
            matched_even_anchors=even['type_by_class'],support_failures=machine['support_failures']),
        readout=dict(allmask_expected_events=sum(v['expected'] for v in occupancy),
            marginal_NULL_mode_active=sum(v['marginal_mode_active'] for v in occupancy),
            marginal_NULL_mode_empty_windows=sum(v['marginal_mode_active']==0 for v in occupancy),
            true_slots_under_half=sum(v['true_slots_below_half'] for v in occupancy),
            expected_on_NULL_target_slots=sum(v['expected_on_NULL_target_slots'] for v in occupancy),
            marginal_mode_is_not_joint_chart_MAP=True),
        raw_filtered_target_scores_exact=all(r['scores_raw']==r['scores_filtered'] for r in records))


def main():
    path=ROOT/'reports/local-integration-v1/data-drag-density-probe-summary-v1.json'
    if path.exists():raise FileExistsError('Preserve analysis')
    plan=json.loads((ROOT/'bench/rhythm-v2'/f'{ID}.json').read_text())
    raw={a:json.loads((ART/f'{a}-outcomes.json').read_text()) for a in ('raw','filtered')}
    summary={a:summarize(v,plan) for a,v in raw.items()}
    a,b=summary['raw'],summary['filtered']
    delta=dict(type_NLL_reduction=a['all_mask']['type_nll_micro']-b['all_mask']['type_nll_micro'],
        first_member_NLL_reduction=a['all_mask']['first_member']['model_nll']-b['all_mask']['first_member']['model_nll'],
        type_positive_NLL_reduction={k:a['all_mask']['type_by_class'][k]['model_nll']-b['all_mask']['type_by_class'][k]['model_nll'] for k in KINDS},
        free_TP_gain={k:b['per_type'][k]['tp']-a['per_type'][k]['tp'] for k in KINDS},
        additional_empty_drafts=b['free_empty_drafts']-a['free_empty_drafts'],
        additional_empty_control_events=b['empty_control']['pred']-a['empty_control']['pred'],
        hold_tail_proper_NLL_reduction=a['all_mask']['hold_tail']['model_nll']-b['all_mask']['hold_tail']['model_nll'],
        onset10_gain=b['onset10_component_macro']-a['onset10_component_macro'],full50_gain=b['full50_component_macro']-a['full50_component_macro'])
    by_arm={name:{r['index']:r for r in records if not r['empty']} for name,records in raw.items()}
    per_component=defaultdict(lambda:dict(raw=0,filtered=0))
    for name,records in by_arm.items():
        for r in records.values():
            per_component[r['component']][name]+=sum(r['scores_raw']['per_type'][k]['tp'] for k in ('hold','flick','drag'))
    delta['non_tap_TP_gain_components']=sum(v['filtered']>v['raw'] for v in per_component.values())
    delta['non_tap_TP_component_details']=dict(per_component)
    result=dict(id=ID,arms=summary,filtered_minus_raw=delta,scope='One seed,640261parameters,128data-enriched windows,128freshupdates perarm;not full-scale efficacy',
        inference='Data intervention whole effect,including changed canonical slot assignments;no isolated component attribution',
        cloud_spend_usd=0,confirmation_scored=False)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(delta=delta,arms={k:{q:v[q] for q in ('pred','free_empty_drafts','full50_component_macro','onset10_component_macro','per_type','empty_control','free_hold','readout')} for k,v in summary.items()}),ensure_ascii=False))


if __name__=='__main__':main()
