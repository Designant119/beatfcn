"""Matched-cohort v2 analysis;separate training effect from relabeling effect."""
from collections import defaultdict
from copy import deepcopy
import json

from scripts import run_drag_data_probe_v2 as trial
from scripts.analyze_drag_data_probe import summarize,proper_summary,mean


def free_summary(records,regime):
    rows=[r for r in records if not r['empty']];groups=defaultdict(list)
    for r in rows:groups[r['component']].append(r)
    scores='scores_'+regime;target='target_'+regime
    pertype={}
    for kind in trial.core.KINDS:
        values=[r[scores]['per_type'][kind] for r in rows];tp=sum(v['tp'] for v in values);pred=sum(v['pred'] for v in values);truth=sum(v['true'] for v in values)
        pertype[kind]=dict(tp=tp,pred=pred,true=truth,f1=2*tp/max(pred+truth,1))
    return dict(true=sum(len(r[target]) for r in rows),pred=sum(len(r['prediction']['events']) for r in rows),per_type=pertype,
        full50_component_macro=mean([mean([r[scores]['full_note_f1'] for r in rr]) for rr in groups.values()]),
        onset10_component_macro=mean([mean([r[scores]['onset_10ms']['f1'] or 0 for r in rr]) for rr in groups.values()]),
        actual_empty_targets=sum(not r[target] for r in rows),missed_nonempty_windows=sum(bool(r[target]) and not r['prediction']['events'] for r in rows),
        predictions_in_new_empty_targets=sum(len(r['prediction']['events']) for r in rows if not r[target]))


def main():
    path=trial.ROOT/'reports/local-integration-v1/data-drag-density-probe-summary-v2.json'
    if path.exists():raise FileExistsError('Preserve v2 result')
    plan=trial.core.read(trial.PLAN);records={a:trial.core.read(trial.ART/f'{a}-outcomes.json') for a in ('raw','filtered')}
    primary={};secondary={};controls={}
    for arm,rr in records.items():
        normalized=deepcopy(rr)
        for r in normalized:
            r['target_raw']=r['target_filtered'];r['scores_raw']=r['scores_filtered'];r['hold_raw']=r['hold_filtered']
        primary[arm]=summarize(normalized,plan)
        primary[arm]['raw_filtered_target_scores_exact']=all(r['scores_raw']==r['scores_filtered'] for r in rr)
        primary[arm]['cohort_diagnostics']=free_summary(rr,'filtered')
        draft=[r for r in rr if not r['empty']]
        secondary[arm]=dict(free=free_summary(rr,'raw'),all_mask=proper_summary(draft,'raw_anchor_all_mask',plan))
        positive=proper_summary(draft,'true_context',plan);zero=proper_summary(draft,'audio_zero',plan)
        pp=primary[arm];even_type=pp['common_machine_context']['type_nll_micro']-pp['machine_context_penalty']['type_nll']
        controls[arm]=dict(true_context=positive,audio_zero=zero,
            true_context_type_gain=even_type-positive['type_nll_micro'],audio_zero_type_damage=zero['type_nll_micro']-pp['all_mask']['type_nll_micro'])
    a,b=primary['raw'],primary['filtered']
    training_effect=dict(type_NLL_reduction=a['all_mask']['type_nll_micro']-b['all_mask']['type_nll_micro'],
        first_member_NLL_reduction=a['all_mask']['first_member']['model_nll']-b['all_mask']['first_member']['model_nll'],
        type_positive_NLL_reduction={k:a['all_mask']['type_by_class'][k]['model_nll']-b['all_mask']['type_by_class'][k]['model_nll'] for k in trial.core.KINDS},
        free_TP_gain={k:b['per_type'][k]['tp']-a['per_type'][k]['tp'] for k in trial.core.KINDS},
        onset10_gain=b['onset10_component_macro']-a['onset10_component_macro'],full50_gain=b['full50_component_macro']-a['full50_component_macro'],
        original_target_full50_gain=secondary['filtered']['free']['full50_component_macro']-secondary['raw']['free']['full50_component_macro'],
        original_target_type_NLL_reduction=secondary['raw']['all_mask']['type_nll_micro']-secondary['filtered']['all_mask']['type_nll_micro'])
    relabel={arm:dict(full50_change=v['full50_component_macro']-secondary[arm]['free']['full50_component_macro'],
        onset10_change=v['onset10_component_macro']-secondary[arm]['free']['onset10_component_macro']) for arm,v in primary.items()}
    result=dict(id=trial.ID,primary_common_filtered=primary,secondary_common_original=secondary,controls=controls,
        training_effect_on_same_target=training_effect,score_change_from_relabeling_same_predictions=relabel,
        scope='One seed/same640261parameters/sameinitial128windowsandstream;primary1304events,secondary1389;not own-cohort comparison',
        cloud_spend_usd=0,confirmation_scored=False)
    trial.core.write(path,result)
    print(json.dumps(dict(training_effect=training_effect,relabel=relabel,arms={arm:dict(pred=v['pred'],true=v['true'],
        full50=v['full50_component_macro'],onset10=v['onset10_component_macro'],types=v['per_type'],hold=v['free_hold'],
        empty=v['empty_control'],readout=v['readout'],first=v['all_mask']['first_member']) for arm,v in primary.items()}),ensure_ascii=False))


if __name__=='__main__':main()
