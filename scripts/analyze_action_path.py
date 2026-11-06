"""Recompute D07 evidence; target-informed scores can never promote a model."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from beatfcn.events.canonical import Event, MODE_TYPES
from beatfcn.model.evaluation import score_events

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'reports/action-path-diagnostic-v1.json'
    payload = json.loads(source.read_text())
    plan = json.loads((ROOT / 'bench/rhythm-v2/action-path-diagnostic-v1.json').read_text())
    panel = json.loads((ROOT / 'reports/r03_window_raw/panel.json').read_text())
    cases = {c['case_id']: c for c in panel['cases']}
    expected = {(m, c, v) for m in plan['models'] for c in cases for v in plan['variants']}
    seen = set(); scores = defaultdict(list); teacher = defaultdict(lambda: [[0]*4 for _ in range(4)])
    types = defaultdict(lambda: defaultdict(Counter)); bad_cases = []
    for r in payload['records']:
        key = (r['model'], r['case_id'], r['variant'])
        if key in seen or key not in expected:
            raise ValueError('duplicate or unexpected diagnostic identity')
        seen.add(key); c = cases[r['case_id']]
        if r['chart_id'] != c['row']['chart_id'] or r['c0_us'] != c['c0_us'] or not r['diagnostic_only']:
            raise ValueError('diagnostic contract differs')
        if r['variant'] == 'teacher_forced':
            target = teacher[(r['model'], r['mode'])]
            matrix = r['decisions']['type']['confusion_truth_rows_pred_columns']
            for i in range(4):
                for j in range(4): target[i][j] += matrix[i][j]
            continue
        pred = [Event.from_json(e) for e in r['generated_events']]
        truth = [Event.from_json(e) for e in r['truth_events']]
        recomputed = score_events(pred, truth, r['mode'])
        if recomputed != r['scores'] or not r['complete']:
            raise ValueError('saved metrics differ or inference incomplete')
        group = (r['model'], r['variant'])
        row = dict(case_id=r['case_id'], complete=r['complete'], legal=r['legal'],
                   predicted_types=r['predicted_types'], truth_types=r['truth_types'],
                   first_onset_error_ms=(min(e.onset_us for e in pred)-min(e.onset_us for e in truth))/1000 if pred else None,
                   empty=not pred)
        for metric in ('full_note', 'timing_point'):
            item = recomputed['edit_utility'][metric]
            row[metric + '_f1'] = 2*item['matched']/(item['pred']+item['true']) if item['pred']+item['true'] else 1.
        for kind in MODE_TYPES[r['mode']]:
            counts = score_events([e for e in pred if e.type == kind], [e for e in truth if e.type == kind],
                                  r['mode'])['edit_utility']['full_note']
            types[group][r['mode']+':'+kind].update({k: counts[k] for k in ('matched', 'pred', 'true')})
        scores[group].append(row)
        if r['variant'] == 'free' and (not pred or abs(row['first_onset_error_ms']) > 20):
            bad_cases.append(dict(model=r['model'], **row))
        if r['variant'] == 'free' and (not r['replay_equal'] or r['truth_supplied']):
            raise ValueError('free inference changed or was supplied hidden truth')
    if seen != expected or not payload['complete'] or payload['optimizer_updates'] or payload['checkpoint_saved']:
        raise ValueError('incomplete or non-read-only diagnostic')
    summary = {}
    for (model, variant), rows in scores.items():
        n = len(rows)
        summary.setdefault(model, {})[variant] = dict(
            cases=rows, macro_full_note_f1=sum(r['full_note_f1'] for r in rows)/n,
            macro_timing_point_f1=sum(r['timing_point_f1'] for r in rows)/n,
            first_onset_within_20ms=sum(r['first_onset_error_ms'] is not None and abs(r['first_onset_error_ms']) <= 20 for r in rows),
            empty_cases=sum(r['empty'] for r in rows), illegal_cases=sum(not r['legal'] for r in rows),
            types={k: dict(v, recall=v['matched']/v['true'] if v['true'] else None)
                   for k, v in types[model, variant].items()})
    comparisons = {}
    for model, variants in summary.items():
        a, b, c = [variants[v]['macro_full_note_f1'] for v in ('free', 'oracle_onsets', 'oracle_onsets_and_outer_history')]
        comparisons[model] = dict(onset_oracle_gain=b-a, added_true_outer_history_gain=c-b,
            onset_hypothesis_threshold_met=b-a >= .15, outer_history_hypothesis_threshold_met=c-b >= .15,
            product_promotion=False)
    report = dict(complete=True, product_promotion=False, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        results=summary, comparisons=comparisons, teacher_type_confusion={':'.join(k):v for k,v in teacher.items()},
        bad_cases=bad_cases, optimizer_updates=0,
        limitation='Oracle scores disclose target times and earlier target groups. These are causal diagnostic contrasts on eight windows, not usable-model quality or population estimates.')
    output = ROOT / 'reports/d07_action_path_review.json'; output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines = ['# D07：选时、前文与动作错误分离', '',
        '所有 oracle 结果使用了真值提示，只能用于定位。没有训练或晋升模型。', '',
        '| 模型 | 自由生成完整音符 F1 | 给正确起音 | 再给正确较早谱面组 | 自由生成首起音 ±20ms |',
        '| --- | ---: | ---: | ---: | ---: |']
    for model, v in summary.items():
        lines.append(f"| {model} | {v['free']['macro_full_note_f1']:.4f} | {v['oracle_onsets']['macro_full_note_f1']:.4f} | "
                     f"{v['oracle_onsets_and_outer_history']['macro_full_note_f1']:.4f} | {v['free']['first_onset_within_20ms']}/8 |")
    lines += ['', '两份权重的首起音均 0/8 命中；非空片段首音符延后约 0.66–5.34 秒。正确前文在正确起音之上又增加约 .21–.24 F1，支持错误前文的放大效应，不能据此认定使用历史本身有错。',
        '', '下一候选目标：让没有正确前文可依靠的起步时间与首音符类型得到独立监督权重；其余样本、模型、解码不变。先检验当前平均损失是否稀释这些位置，再用真实探针和匹配对照裁决。',
        '', '完整类型召回、teacher-forced 混淆、逐片段首音符误差及 oracle 合法性见同名 JSON。']
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(comparisons=comparisons,first_onsets={m:v['free']['first_onset_within_20ms'] for m,v in summary.items()})))


if __name__ == '__main__': main()
