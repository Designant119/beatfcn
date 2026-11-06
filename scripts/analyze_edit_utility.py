"""Rescore saved rollouts locally; no network, sampling, or model updates."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beatfcn.events.canonical import Event
from beatfcn.model.edit_utility import edit_utility


def interval(values):
    by_song = defaultdict(list)
    for song, value in values:
        if value is not None:
            by_song[song].append(value)
    x = np.array([np.mean(v) for v in by_song.values()])
    ci = None
    if len(x) >= 2:
        draws = np.random.default_rng(20261024).choice(x, (5000, len(x)), replace=True).mean(1)
        ci = np.quantile(draws, [.025, .975]).tolist()
    return {'mean_song': float(x.mean()) if len(x) else None, 'song_bootstrap_95': ci, 'songs': len(x)}


def main():
    directory = ROOT / 'reports/capability_review_raw/evaluation-review-v1/records'
    records, files = [], {}
    for path in sorted(directory.glob('*.json')):
        r = json.loads(path.read_text())
        if r['condition'] != 'baseline':
            continue
        files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        item = {k: r[k] for k in ('id', 'checkpoint', 'song', 'chart_id', 'mode', 'request_kind', 'complete')}
        item.update(legal=not r.get('legality'), failure=r.get('failure'))
        if r.get('complete') and not r.get('legality'):
            pred = [Event.from_json(e) for e in r['generated_events']]
            truth = [Event.from_json(e) for e in r['truth_events']]
            item['metrics'] = edit_utility(pred, truth, r['mode'])
            item['sensitivity'] = {f'delete_{d}_review_{v}': edit_utility(pred, truth, r['mode'],
                                    delete_cost=d, review_cost=v)['full_note']['saved_fraction']
                                   for d, v in ((.25, 0.), (.5, 0.), (1., .1), (1., .25))}
            for label, subset in (('drag', lambda e: e.type == 'drag'), ('non_drag', lambda e: e.type != 'drag')):
                item[label] = edit_utility([e for e in pred if subset(e)], [e for e in truth if subset(e)], r['mode'])
        records.append(item)
    manifest = json.loads((ROOT / 'bench/capability-v1/manifest.json').read_text())
    for checkpoint in ('3500', '4500'):
        present = {(r['chart_id'], r['request_kind']) for r in records if r['checkpoint'] == checkpoint}
        for q in manifest['requests']:
            if (q['chart_id'], q['kind']) not in present:
                records.append({'checkpoint': checkpoint, 'chart_id': q['chart_id'], 'song': q['song'],
                                'mode': q['mode'], 'request_kind': q['kind'], 'complete': False,
                                'failure': {'reason': 'missing_result'}, 'legal': False})
    groups = defaultdict(list)
    for r in records:
        groups['/'.join(r[k] for k in ('checkpoint', 'mode', 'request_kind'))].append(r)
    summary = {}
    for key, rows in sorted(groups.items()):
        good = [r for r in rows if 'metrics' in r]
        report = {'attempted': len(rows), 'complete_legal': len(good), 'unscored': len(rows) - len(good)}
        for metric in ('timing_point', 'rhythm_head', 'typed_head', 'full_note'):
            xs = [(r['song'], r['metrics'][metric]['saved_fraction']) for r in good]
            report[metric] = {**interval(xs),
                              'positive_requests': sum(x is not None and x > 0 for _, x in xs),
                              'empty_targets': sum(x is None for _, x in xs)}
        report['sensitivity'] = {label: interval([(r['song'], r['sensitivity'][label]) for r in good])
                                  for label in (good[0]['sensitivity'] if good else ())}
        for metric in ('drag', 'non_drag'):
            report[metric] = interval([(r['song'], r[metric]['full_note']['saved_fraction']) for r in good])
        summary[key] = report
    out = {'schema': 'beatfcn.edit_utility_review.v1', 'source_files': files, 'groups': summary, 'records': records,
           'scope': 'Retrospective historical v1 rollouts; not a new-model promotion test. All expected requests retained.',
           'limitations': ['Reference reconstruction proxy, not measured editing time or acceptability.',
                          'Old inputs include known capacity failures; new v2 benchmark remains necessary.',
                          'Bootstrap intervals are descriptive and not multiple-comparison adjusted.',
                          'No audio baseline result yet; no quality promotion from this report.'],
           'training_updates': 0, 'new_remote_compute': False}
    target = ROOT / 'reports/edit_utility_review_2026-10-24.json'
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    lines = ['# 历史生成结果的制谱修改量复盘', '',
             '这是重建同一参考谱的插入/删除操作代理，不是人类实际耗时。替换、移动可能更便宜，合理的替代谱也可能被扣分。', '',
             '默认新增与删除各计 1，匹配容差 20 ms；完整音符要求 Hold 两端都匹配。时间点先去重，不用和弦重数增加召回。', '',
             '净节省比例 = 1 − (缺失数 + 多余数) / 参考数。负数表示按此操作模型，修改量超过空白起稿；空目标单列。', '',
             '| 检查点 / 模式 / 请求 | 完成合法 / 应评 | 时间点净节省 | 完整 note 净节省 | 完整 note 95% 区间 |',
             '|---|---:|---:|---:|---|']
    for key, row in summary.items():
        pct = lambda x: f'{x:.1%}' if x is not None else '—'
        ci = row['full_note']['song_bootstrap_95']
        lines.append(f"| {key} | {row['complete_legal']}/{row['attempted']} | {pct(row['timing_point']['mean_song'])} | "
                     f"{pct(row['full_note']['mean_song'])} | {(' ~ '.join(pct(x) for x in ci)) if ci else '—'} |")
    lines += ['', '历史 v1 数据的容量失败保留在分母；均值与区间仅来自完成合法的非空请求。不同模式、请求、drag 与非 drag 分开保存在 JSON。',
              '', '不能由这些结果认定人类会更慢，也不能据此宣布模型已对制谱有帮助。下一轮必须补简单音频基线、修复后的输入口径和候选模型的同面板比较。']
    target.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'report': str(target), 'records': len(records), 'groups': len(summary), 'remote_compute': False}))


if __name__ == '__main__':
    main()
