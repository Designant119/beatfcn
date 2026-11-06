"""R05: control gain is primary; preserve factual quality and all note types."""
import hashlib
import json
from pathlib import Path

from scripts.review_r04 import summarize, decide as quality_guardrails

ROOT=Path(__file__).resolve().parents[1]


def decide(control,candidate,gate):
    result=quality_guardrails(control,candidate,gate)
    reasons=result['reasons']
    fields=[k for k,v in candidate['control_fields'].items() if v['passed']]
    if len(fields)<gate['minimum_passed_fields']:reasons.append('insufficient strict condition gains')
    modes={field.split(':')[0] for field in fields}
    if gate['both_modes_have_passed_field'] and modes!={'mania_4k','phira'}:reasons.append('control gains do not cover both modes')
    kinds={field.split(':')[1] for field in fields}
    types={'hold_head_ratio','flick_ratio','drag_ratio'}
    structures={'density_notes_per_second','peak_notes_per_second','hold_occupancy_ratio','chord_ratio','jack_ratio'}
    if gate['require_type_and_structure_controls'] and not (kinds & types and kinds & structures):
        reasons.append('control gains must include type and structure')
    result.update(short_window_gate_passed=not reasons,passed_control_fields=fields,product_promotion=False,
        next_action='Retain architecture for further utility validation; this is not product acceptance.' if not reasons
                    else 'Reject and revert only R05 owned changes; preserve diagnostics and failed checkpoint evidence.')
    return result


def main():
    path=ROOT/'bench/rhythm-v2/r05-direct-condition-v1.json';plan=json.loads(path.read_text())
    panel=json.loads((ROOT/'reports/r03_window_raw/panel.json').read_text())
    paths={'control':ROOT/'reports/r03_window_raw/r03-window-all.benchmark.json','candidate':ROOT/'reports/r05_evaluate.json'}
    raw={k:json.loads(p.read_text()) for k,p in paths.items()}
    plan_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if raw['candidate']['plan_sha256']!=plan_sha:raise ValueError('candidate plan identity differs')
    targets={k:{r['case_id']:r['truth_events'] for r in v['records'] if r['field'] is None} for k,v in raw.items()}
    if targets['control']!=targets['candidate']:raise ValueError('paired targets differ')
    results={k:summarize(v,panel) for k,v in raw.items()}
    verdict=decide(results['control'],results['candidate'],plan['quality_gate'])
    report=dict(schema='beatfcn.r05_direct_condition_review.v1',plan_sha256=plan_sha,results=results,
        comparison=verdict,product_promotion=False,source_files_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()})
    output=ROOT/'reports/r05_review.json';output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    lines=['# R05 直接条件残差复盘','','按保存事件重算全部指标；短窗条件改善不等于整曲制谱可用。','',
        '| 模型 | 条件通过 | 完整音符 F1 | 时间点 F1 | 空窗 | 数量比 |','|---|---:|---:|---:|---:|---:|']
    for name,r in results.items():
        lines.append(f"| {name} | {r['passed_fields']}/13 | {r['macro_full_note_f1']:.4f} | {r['macro_timing_point_f1']:.4f} | {r['empty_windows']}/8 | {r['aggregate_count_ratio']:.3f} |")
    lines+=['','裁决：'+('研究保留，仍未产品晋升。' if verdict['short_window_gate_passed'] else '否决并撤回本轮架构。'),'']
    lines+=['- '+reason for reason in verdict['reasons']]
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(verdict,ensure_ascii=False))


if __name__=='__main__':main()
