"""Recompute R03 metrics from events and apply the frozen, joint acceptance gates."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from beatfcn.events.canonical import Event, MODE_TYPES
from beatfcn.events import controls as C
from beatfcn.model.evaluation import score_events
from beatfcn.train.control_panel import response_gate, field_decision

ROOT = Path(__file__).resolve().parents[1]


def f1(counts):
    denominator = counts['pred'] + counts['true']
    return 2 * counts['matched'] / denominator if denominator else 1.


def analyze(payload, panel):
    expected_cases = {c['case_id']: c for c in panel['cases']}
    records = payload['records']
    seen = set(); factual = []; control_rows = defaultdict(list)
    by_type = defaultdict(Counter)
    for row in records:
        key = (row['case_id'], row['field'], tuple(row['requested'] or []))
        if key in seen or row['case_id'] not in expected_cases:
            raise ValueError('duplicate or unknown generation identity')
        seen.add(key)
        case = expected_cases[row['case_id']]
        if (row['chart_id'], row['c0_us'], row['mode']) != (case['row']['chart_id'], case['c0_us'], case['row']['mode']):
            raise ValueError('generation provenance differs from frozen panel')
        pred = [Event.from_json(e) for e in row['generated_events']]
        truth = [Event.from_json(e) for e in row['truth_events']]
        scores = score_events(pred, truth, row['mode'])
        if scores != row['scores']:
            raise ValueError('saved scores do not reproduce from the event lists')
        achieved = C.raw_controls(row['mode'], pred, [(row['c0_us'], row['c0_us'] + 8_000_000)])
        if achieved != row['achieved']:
            raise ValueError('saved controls do not reproduce from the event lists')
        if row['field'] is not None:
            control_rows[(row['mode'], row['field'], row['case_id'])].append(row)
            continue
        full = scores['edit_utility']['full_note']; timing = scores['edit_utility']['timing_point']
        factual.append(dict(case_id=row['case_id'], chart_id=row['chart_id'], mode=row['mode'],
            complete=row['complete'], legal=row['legal'], full_note_f1=f1(full), timing_point_f1=f1(timing),
            reference_edit_saved_fraction=full['saved_fraction'], timing_edit_saved_fraction=timing['saved_fraction'],
            predicted_types=dict(Counter(e.type for e in pred)), truth_types=dict(Counter(e.type for e in truth)),
            density_ratio=scores['density_ratio'], diagnostics=row['diagnostics']))
        for kind in MODE_TYPES[row['mode']]:
            counts = score_events([e for e in pred if e.type == kind], [e for e in truth if e.type == kind],
                                  row['mode'])['edit_utility']['full_note']
            by_type[(row['mode'], kind)].update({k: counts[k] for k in ('matched', 'pred', 'true')})
    if {r['case_id'] for r in factual} != set(expected_cases):
        raise ValueError('missing unspecified-condition quality cases')
    field_gates = {}
    for mode in ('mania_4k', 'phira'):
        for field in C.MODE_FIELDS[mode]:
            gates = []
            for case in [c for c in panel['cases'] if c['row']['mode'] == mode]:
                rows = control_rows.get((mode, field, case['case_id']))
                if rows is not None:
                    gates.append(dict(case_id=case['case_id'], **response_gate(rows, field)))
                elif not field_decision(gates)['failed']:
                    raise ValueError('unjustified missing control requests')
            key = f'{mode}:{field}'
            if gates != payload['field_gates'][key] or field_decision(gates) != payload['field_decisions'][key]:
                raise ValueError('saved control verdict does not reproduce')
            field_gates[key] = field_decision(gates)
    types = {f'{mode}:{kind}': dict(counts, recall=counts['matched'] / counts['true'] if counts['true'] else None)
             for (mode, kind), counts in by_type.items()}
    return dict(complete=payload['complete'], factual=factual, types=types, control_fields=field_gates,
                passed_fields=sum(r['passed'] for r in field_gates.values()),
                macro_full_note_f1=float(np.mean([r['full_note_f1'] for r in factual])),
                macro_timing_point_f1=float(np.mean([r['timing_point_f1'] for r in factual])),
                macro_reference_edit_saved_fraction=float(np.mean([r['reference_edit_saved_fraction'] for r in factual])),
                all_factual_complete_legal=all(r['complete'] and r['legal'] for r in factual),
                all_requests_complete_legal=all(r['complete'] and r['legal'] for r in records),
                event_recomputation_agreed=True)


def compare(initial, control, candidate, plan):
    gate = plan['quality_gate']; reasons = []
    for name, row in [('initial', initial), ('control', control), ('candidate', candidate)]:
        if not row['complete']:
            reasons.append(f'{name} evaluation incomplete')
    if candidate['passed_fields'] != plan['control_gate']['fields_required']:
        reasons.append(f"only {candidate['passed_fields']}/13 mode-field controls passed")
    initial_gain = candidate['macro_full_note_f1'] - initial['macro_full_note_f1']
    control_gain = candidate['macro_full_note_f1'] - control['macro_full_note_f1']
    if initial_gain < gate['macro_full_note_f1_gain_over_initial_at_least']:
        reasons.append('full-note gain over initial below threshold')
    if control_gain < gate['macro_full_note_f1_gain_over_matched_density_arm_at_least']:
        reasons.append('full-note gain over matched control below threshold')
    if not candidate['all_requests_complete_legal']:
        reasons.append('candidate generation incomplete or illegal, including control requests')
    type_comparison = {}
    for key, before in initial['types'].items():
        after = candidate['types'][key]
        change = None if before['recall'] is None else after['recall'] - before['recall']
        ratio = after['pred'] / before['pred'] if before['pred'] else None
        type_comparison[key] = dict(recall_before=before['recall'], recall_after=after['recall'],
                                   recall_change=change, prediction_count_ratio=ratio)
        if change is not None and change < -gate['max_per_mode_type_recall_drop_from_initial']:
            reasons.append(f'{key} recall regressed')
        if not key.endswith(':tap') and ratio is not None and ratio < gate['min_non_tap_prediction_count_ratio_to_initial_if_present']:
            reasons.append(f'{key} predictions collapsed')
    return dict(short_window_gate_passed=not reasons, reasons=reasons, initial_f1_gain=initial_gain,
                matched_control_f1_gain=control_gain, type_comparison=type_comparison,
                product_promotion=False,
                next_action='whole-song and actual-use audit still required' if not reasons else 'reject label-coverage intervention; audit two-strike route change')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'reports/r03_window_raw')
    args = parser.parse_args()
    plan_path = ROOT / 'bench/rhythm-v2/r03-window-conditions-v1.json'
    plan = json.loads(plan_path.read_text()); panel = json.loads((args.input / 'panel.json').read_text())
    names = ['baseline', 'r03-window-density.benchmark', 'r03-window-all.benchmark']
    results = {}; fingerprints = {}
    for name in names:
        path = args.input / f'{name}.json'
        if path.exists():
            payload = path.read_bytes()
            results[name] = analyze(json.loads(payload), panel)
            fingerprints[name] = hashlib.sha256(payload).hexdigest()
    if 'baseline' not in results:
        raise ValueError('baseline unavailable')
    report = dict(schema='beatfcn.r03_control_review.v1', plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                  source_files_sha256=fingerprints, results=results, product_promotion=False)
    if all(name in results for name in names):
        report['comparison'] = compare(*(results[name] for name in names), plan)
    else:
        report['status'] = 'baseline_only; no intervention outcome yet'
    out = ROOT / 'reports/r03_window_review.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    lines = ['# R03 完整窗口条件标签复盘', '', '逐事件重算指标与保存结果一致；短窗诊断不能代替整曲或人类实用性验收。', '',
             '| 模型 | 条件通过 | 完整音符 F1 | 时间点 F1 | 参考编辑节省代理 |', '| --- | --- | --- | --- | --- |']
    for name, row in results.items():
        lines.append(f"| {name} | {row['passed_fields']}/13 | {row['macro_full_note_f1']:.4f} | {row['macro_timing_point_f1']:.4f} | {row['macro_reference_edit_saved_fraction']:.4f} |")
    if 'comparison' in report:
        lines += ['', '裁决：' + ('短窗门槛通过，仍需整曲验证。' if report['comparison']['short_window_gate_passed'] else '不通过，不晋升。')]
        lines += ['- ' + reason for reason in report['comparison']['reasons']]
    else:
        lines += ['', '仅完成修复后基线；候选干预尚未评估。']
    out.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(path=str(out), models=list(results), comparison=report.get('comparison')), ensure_ascii=False))


if __name__ == '__main__':
    main()
