"""Audit saved paired outputs; no inference and no acceptance-gate changes."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from beatfcn.train.control_panel import REQUEST_BANDS, response_gate

ROOT = Path(__file__).resolve().parents[1]


def analyze(candidate, control, review):
    factual = {name: {r['case_id']: r for r in payload['records'] if r['field'] is None}
               for name, payload in [('candidate', candidate), ('control', control)]}
    if factual['candidate'].keys() != factual['control'].keys():
        raise ValueError('paired case identities differ')
    cases = []
    for scores in review['results']['candidate']['factual']:
        key = scores['case_id']
        before, after = factual['control'][key], factual['candidate'][key]
        if (before['truth_events'], before['c0_us'], before['mode']) != (
                after['truth_events'], after['c0_us'], after['mode']):
            raise ValueError('paired targets or coordinates differ')
        times = {name: sorted({e['onset_us'] for e in source['generated_events']})
                 for name, source in [('before', before), ('after', after)]}
        truth_times = sorted({e['onset_us'] for e in after['truth_events']})
        bins = []
        for i in range(4):
            left = after['c0_us'] + i * 2_000_000
            right = left + 2_000_000
            counts = {name: sum(left <= e['onset_us'] < right for e in events)
                      for name, events in [('control', before['generated_events']),
                                           ('candidate', after['generated_events']),
                                           ('truth', after['truth_events'])]}
            bins.append(dict(start_s=i * 2, end_s=(i + 1) * 2, **counts))
        for name, events in [('control', before['generated_events']),
                             ('candidate', after['generated_events']), ('truth', after['truth_events'])]:
            if sum(row[name] for row in bins) != len(events):
                raise ValueError('events outside the fixed eight-second window')
        case = {k: scores[k] for k in ('case_id', 'chart_id', 'mode', 'first_onset_error_ms',
            'empty', 'full_note_f1', 'timing_point_f1', 'prediction_count', 'truth_count',
            'predicted_types', 'truth_types', 'reference_edit_saved_fraction',
            'timing_edit_saved_fraction')}
        first = times['after'][0] if times['after'] else None
        case.update(first_onset_unchanged=times['before'][:1] == times['after'][:1],
            timing_sequence_unchanged=times['before'] == times['after'], two_second_bins=bins,
            truth_groups=len(truth_times),
            truth_groups_before_first_prediction=sum(t < first - 20_000 for t in truth_times)
                if first is not None else None,
            note_decisions=after['diagnostics'].get('note_decisions', {}))
        cases.append(case)
    grouped = defaultdict(list)
    for row in candidate['records']:
        if row['field'] is not None:
            grouped[(row['case_id'], row['field'])].append(row)
    triplets = []
    for (case_id, field), rows in grouped.items():
        rows.sort(key=lambda r: REQUEST_BANDS[field].index(r['requested']))
        gate = response_gate(rows, field)
        signatures = [json.dumps(r['generated_events'], sort_keys=True) for r in rows]
        triplets.append(dict(case_id=case_id, field=field, **gate,
            identical_events=len(set(signatures)) == 1,
            predicted_types=[r['predicted_types'] for r in rows]))
    return dict(schema='beatfcn.window_bad_cases.v1', cases=cases, condition_triplets=triplets,
        measured_triplets=len(triplets), identical_triplets=sum(r['identical_events'] for r in triplets),
        strictly_monotonic_triplets=sum(r['strictly_monotonic'] for r in triplets),
        all_in_band_triplets=sum(r['all_in_band'] for r in triplets),
        passed_triplets=sum(r['passed'] for r in triplets),
        unmeasured_requests=sum(r['remaining'] * 3 for r in candidate['field_decisions'].values()),
        first_onset_unchanged_windows=sum(r['first_onset_unchanged'] for r in cases),
        positive_full_note_edit_proxy_windows=sum(r['reference_edit_saved_fraction'] > 0 for r in cases),
        new_inference_calls=0, product_promotion=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', default='reports/r06_evaluate.json')
    parser.add_argument('--control', default='reports/r03_window_raw/r03-window-density.benchmark.json')
    parser.add_argument('--review', default='reports/r06_review.json')
    parser.add_argument('--output', default='reports/r06_bad_cases.json')
    args = parser.parse_args()
    paths = {name: ROOT / getattr(args, name) for name in ('candidate', 'control', 'review')}
    payloads = {name: json.loads(path.read_text()) for name, path in paths.items()}
    fingerprints = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    for name in ('candidate', 'control'):
        if fingerprints[name] != payloads['review']['source_files_sha256'][name]:
            raise ValueError(f'{name} differs from the reviewed artifact')
    report = analyze(**payloads)
    report['source_sha256'] = fingerprints
    output = ROOT / args.output
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    lines = ['# 保存事件的错误分布', '', '只复用既有结果，不追加生成，不改变冻结裁决。', '',
        f"三档条件请求共 {report['measured_triplets']} 组：逐事件相同 {report['identical_triplets']} 组，"
        f"严格单调 {report['strictly_monotonic_triplets']} 组，全部在带 {report['all_in_band_triplets']} 组，"
        f"同时通过 {report['passed_triplets']} 组；按提前停止规则未测 {report['unmeasured_requests']} 个请求。", '',
        '| 片段 | 首点误差 ms | 输出/人谱音符 | 完整 F1 | 时间点 F1 | 输出类型 |',
        '|---|---:|---:|---:|---:|---|']
    for row in report['cases']:
        error = '空窗' if row['empty'] else f"{row['first_onset_error_ms']:.0f}"
        lines.append(f"| {row['case_id']} | {error} | {row['prediction_count']}/{row['truth_count']} | "
            f"{row['full_note_f1']:.4f} | {row['timing_point_f1']:.4f} | {row['predicted_types']} |")
    lines += ['', '| 片段 | 0–2s 输出/人谱 | 2–4s | 4–6s | 6–8s | 首次输出前漏掉的时间组 |',
              '|---|---:|---:|---:|---:|---:|']
    for row in report['cases']:
        bins = ' | '.join(f"{b['candidate']}/{b['truth']}" for b in row['two_second_bins'])
        missed = '整窗为空' if row['empty'] else f"{row['truth_groups_before_first_prediction']}/{row['truth_groups']}"
        lines.append(f"| {row['case_id']} | {bins} | {missed} |")
    lines += ['', '| 片段 / 条件 | 低、中、高档实际值 | 逐事件相同 |', '|---|---|---|']
    for row in report['condition_triplets']:
        values = ', '.join('未定义' if v is None else f'{v:.3f}' for v in row['values'])
        lines.append(f"| {row['case_id']} / {row['field']} | {values} | {row['identical_events']} |")
    lines += ['', '这里只统计固定诊断面板，不能推断整库成功率。空窗不计首点误差改善；'
              '相对于同一人谱的差异也不能独立裁决另一份合理谱面。']
    output.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('cases', 'condition_triplets', 'source_sha256')}))


if __name__ == '__main__':
    main()
