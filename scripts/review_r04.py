"""Apply the preregistered R04 gates to independently recomputed event metrics."""
import hashlib
import json
from pathlib import Path
from statistics import mean

from scripts.review_r03 import analyze

ROOT = Path(__file__).resolve().parents[1]


def summarize(payload, panel):
    if payload.get('conditions_from_hidden_target') is not False:
        raise ValueError('product evaluation input provenance is missing or invalid')
    result = analyze(payload, panel)
    rows = {r['case_id']: r for r in payload['records'] if r['field'] is None}
    for case in result['factual']:
        row = rows[case['case_id']]
        predicted = sorted({e['onset_us'] for e in row['generated_events']})
        truth = sorted({e['onset_us'] for e in row['truth_events']})
        if not truth:
            raise ValueError('frozen first-onset panel unexpectedly has an empty target')
        case.update(first_onset_error_ms=(predicted[0]-truth[0])/1000 if predicted else None,
            empty=not predicted,
            longest_internal_gap_ms=max((b-a for a,b in zip(predicted,predicted[1:])), default=0)/1000,
            start_blank_ms=(predicted[0]-row['c0_us'])/1000 if predicted else 8000,
            prediction_count=sum(case['predicted_types'].values()), truth_count=sum(case['truth_types'].values()))
    result.update(first_onset_matches=sum(r['first_onset_error_ms'] is not None and abs(r['first_onset_error_ms'])<=20
                                         for r in result['factual']),
        empty_windows=sum(r['empty'] for r in result['factual']),
        aggregate_count_ratio=sum(r['prediction_count'] for r in result['factual'])/sum(r['truth_count'] for r in result['factual']),
        per_mode_full_note_f1={mode:mean(r['full_note_f1'] for r in result['factual'] if r['mode']==mode)
                              for mode in ('mania_4k','phira')})
    return result


def decide(control, candidate, gate):
    reasons=[]
    if not control['complete'] or not candidate['complete']: reasons.append('evaluation incomplete')
    full_gain=candidate['macro_full_note_f1']-control['macro_full_note_f1']
    timing_gain=candidate['macro_timing_point_f1']-control['macro_timing_point_f1']
    if candidate['first_onset_matches']<gate['first_onset_within_20ms_at_least']:
        reasons.append('first-onset match gate failed')
    if full_gain<gate['full_note_gain_over_control_at_least']: reasons.append('full-note gain below threshold')
    if timing_gain<gate['timing_point_gain_over_control_at_least']: reasons.append('timing-point gain below threshold')
    if candidate['empty_windows']>gate['maximum_empty_windows']: reasons.append('too many empty windows')
    if candidate['aggregate_count_ratio']>gate['maximum_aggregate_count_ratio']: reasons.append('note overproduction')
    if gate['all_requests_complete_legal'] and not candidate['all_requests_complete_legal']:
        reasons.append('incomplete or illegal request, including condition interventions')
    for mode,before in control['per_mode_full_note_f1'].items():
        if candidate['per_mode_full_note_f1'][mode]-before < -gate['max_per_mode_full_note_f1_drop']:
            reasons.append(f'{mode} full-note F1 regressed')
    types={}
    for key,before in control['types'].items():
        after=candidate['types'][key]
        delta=after['recall']-before['recall'] if before['recall'] is not None else None
        ratio=after['pred']/before['pred'] if before['pred'] else None
        types[key]=dict(recall_before=before['recall'],recall_after=after['recall'],recall_change=delta,prediction_count_ratio=ratio)
        if delta is not None and delta < -gate['max_per_mode_type_recall_drop']: reasons.append(f'{key} recall regressed')
        if not key.endswith(':tap') and ratio is not None and ratio < gate['min_existing_non_tap_prediction_ratio']:
            reasons.append(f'{key} predictions collapsed')
    return dict(short_window_gate_passed=not reasons,reasons=reasons,matched_control_full_note_gain=full_gain,
        matched_control_timing_point_gain=timing_gain,type_comparison=types,
        control_fields_passed=candidate['passed_fields'],product_promotion=False,
        next_action='Retain the objective only for further research; controls, whole songs and human utility still require evidence.'
                    if not reasons else 'Reject and revert only the R04 objective patch; audit the objective two-strike breaker.')


def main():
    plan_path=ROOT/'bench/rhythm-v2/r04-initial-loss-v1.json';plan=json.loads(plan_path.read_text())
    panel=json.loads((ROOT/'reports/r03_window_raw/panel.json').read_text())
    paths={'control':ROOT/'reports/r03_window_raw/r03-window-density.benchmark.json',
           'candidate':ROOT/'reports/r04_evaluate.json'}
    payloads={k:json.loads(p.read_text()) for k,p in paths.items()}
    if payloads['candidate']['plan_sha256']!=hashlib.sha256(plan_path.read_bytes()).hexdigest():
        raise ValueError('candidate plan identity differs')
    truths={k:{r['case_id']:r['truth_events'] for r in v['records'] if r['field'] is None} for k,v in payloads.items()}
    if truths['control']!=truths['candidate']: raise ValueError('paired targets differ')
    results={k:summarize(v,panel) for k,v in payloads.items()}
    comparison=decide(results['control'],results['candidate'],plan['quality_gate'])
    report=dict(schema='beatfcn.r04_initial_decision_review.v1',plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        source_files_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
        results=results,comparison=comparison,product_promotion=False)
    output=ROOT/'reports/r04_review.json';output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# R04 初始决策监督复盘','','指标由保存事件重新计算；短窗改善仍不等于整曲制谱可用。','',
           '| 模型 | 首点 ±20ms | 完整音符 F1 | 时间点 F1 | 空窗 | 数量比 | 条件通过 |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,row in results.items():
        lines.append(f"| {name} | {row['first_onset_matches']}/8 | {row['macro_full_note_f1']:.4f} | {row['macro_timing_point_f1']:.4f} | {row['empty_windows']}/8 | {row['aggregate_count_ratio']:.3f} | {row['passed_fields']}/13 |")
    lines+=['','裁决：'+('短窗门槛通过，继续验证实用性。' if comparison['short_window_gate_passed'] else '不通过，撤回本轮目标改动。'),'']
    lines+=['- '+r for r in comparison['reasons']]
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(comparison,ensure_ascii=False))


if __name__=='__main__': main()
