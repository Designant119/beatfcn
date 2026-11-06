"""Frozen R07B blank-context gates, using independently rescored events."""
import hashlib
import json
from pathlib import Path
from statistics import median

from scripts.review_r04 import summarize, decide as quality_guardrails

ROOT = Path(__file__).resolve().parents[1]


def decide(control, candidate, gate):
    result = quality_guardrails(control, candidate, gate)
    reasons = result['reasons']
    errors = {name: [abs(row['first_onset_error_ms']) for row in value['factual']
                     if row['first_onset_error_ms'] is not None]
              for name, value in [('control', control), ('candidate', candidate)]}
    before = median(errors['control']) if errors['control'] else None
    after = median(errors['candidate']) if errors['candidate'] else None
    if before is None or after is None or after > before * gate['maximum_first_onset_median_error_ratio']:
        reasons.append('first-onset error did not fall sufficiently')
    lost = [name for name, field in control['control_fields'].items()
            if field['passed'] and not candidate['control_fields'][name]['passed']]
    if gate['no_strict_condition_loss'] and lost:
        reasons.append('previously working condition lost')
    result.update(short_window_gate_passed=not reasons,
        first_onset_median_error_ms={'control': before, 'candidate': after},
        lost_control_fields=lost, product_promotion=False,
        next_action='Retain only for full-song and actual editing validation; conditions still require evidence.'
        if not reasons else 'Reject R07B and revert its owned context-transform patch; audit the data-representation two-strike breaker.')
    return result


def main():
    path = ROOT / 'bench/rhythm-v2/r07b-blank-context-local-v1.json'
    plan = json.loads(path.read_text()); plan_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    panel = json.loads((ROOT / 'reports/r03_window_raw/panel.json').read_text())
    paths = {'control': ROOT / 'reports/r03_window_raw/r03-window-all.benchmark.json',
             'candidate': ROOT / 'reports/r07b_evaluate.json'}
    raw = {k: json.loads(p.read_text()) for k, p in paths.items()}
    if raw['candidate']['plan_sha256'] != plan_sha:
        raise ValueError('candidate plan identity differs')
    targets = {k: {r['case_id']: r['truth_events'] for r in v['records'] if r['field'] is None}
               for k, v in raw.items()}
    if targets['control'] != targets['candidate']:
        raise ValueError('paired targets differ')
    results = {k: summarize(v, panel) for k, v in raw.items()}
    verdict = decide(results['control'], results['candidate'], plan['quality_gate'])
    report = dict(schema='beatfcn.r07b_blank_context_review.v1', plan_sha256=plan_sha,
        results=results, comparison=verdict, product_promotion=False,
        source_files_sha256={k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()})
    output = ROOT / 'reports/r07b_review.json'
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    lines = ['# R07B 空白上下文监督复盘', '', '按保存事件重算；短窗门槛不是产品验收。', '',
        '| 模型 | 首点 ±20ms | 完整音符 F1 | 时间点 F1 | 空窗 | 数量比 | 条件通过 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, r in results.items():
        lines.append(f"| {name} | {r['first_onset_matches']}/8 | {r['macro_full_note_f1']:.4f} | "
                     f"{r['macro_timing_point_f1']:.4f} | {r['empty_windows']}/8 | "
                     f"{r['aggregate_count_ratio']:.3f} | {r['passed_fields']}/13 |")
    lines += ['', '裁决：' + ('研究保留，仍未产品晋升。' if verdict['short_window_gate_passed']
                            else '否决并撤回本轮上下文干预。'), '']
    lines += ['- ' + reason for reason in verdict['reasons']]
    output.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(verdict, ensure_ascii=False))


if __name__ == '__main__':
    main()
