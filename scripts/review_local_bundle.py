"""Close the local recipe experiment from saved raw predictions, not promises."""
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import subprocess

import numpy as np

from scripts.run_local_bundle import ART, PLAN, REPORT, ROOT, ARMS, sha, write, journal


def metrics(rows):
    active = [r for r in rows if r.get('attempted')]
    matched = Counter(); predicted = Counter(); truth = Counter()
    per_song = defaultdict(list)
    for row in active:
        predicted.update(row['types']); truth.update(row['target_types'])
        matched.update({k: v['tp'] for k, v in row['scores']['per_type'].items()})
        per_song[row['chart_id']].append(row['scores']['full_note_f1'])
    return dict(windows=len(rows), attempted=len(active),
        complete=sum(r['prediction']['complete'] and not r['legality'] for r in active),
        nonempty=sum(bool(r['prediction']['events']) for r in active),
        single_type_windows=sum(len(r['types']) == 1 for r in active),
        all_tap_windows=sum(set(r['types']) == {'tap'} for r in active),
        full_note_f1=float(np.mean([r['scores']['full_note_f1'] for r in active])) if active else 0.,
        onset_f1=float(np.mean([r['scores']['onset_f1'] for r in active])) if active else 0.,
        predicted_types=dict(predicted), target_types=dict(truth), matched_types=dict(matched),
        density_ratio=sum(predicted.values()) / max(sum(truth.values()), 1),
        close_positive_fraction=sum(r['positive_gaps_under_10ms'] for r in active) /
                                max(sum(r['positive_gaps'] for r in active), 1),
        max_same_time=max((r['max_same_time'] for r in active), default=0),
        per_song_f1={k: float(np.mean(v)) for k, v in per_song.items()})


def plots():
    data = {a: json.loads((ART / a / 'draft-final.json').read_text()) for a in ARMS}
    paths = []
    for j in (0, 4, 8, 12):
        if any(j >= len(data[a]) or not data[a][j].get('attempted') for a in ARMS):
            continue
        source = data['baseline'][j]
        start = source['request']['start_us']
        end = source['request']['end_us']
        groups = [('Human target', source['target']),
                  ('Original model', data['baseline'][j]['prediction']['events']),
                  ('Combined recipe', data['bundle'][j]['prediction']['events'])]
        kinds = ['tap', 'hold', 'flick', 'drag']
        colors = ['#1674cc', '#cd8312', '#c3426d', '#23846d']
        # A square SVG avoids Quick Look's crop-to-square thumbnail behavior.
        width, height = 1400, 1400
        left, right = 110, 1360
        x = lambda us: left + (us - start) / (end - start) * (right - left)
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
               '<rect width="100%" height="100%" fill="white"/>',
               '<style>text{font-family:Arial,sans-serif;fill:#263444;font-size:15px}</style>',
               f'<text x="30" y="32" style="font-size:22px">{html.escape(source["chart_id"])} / window start {start / 1e6:g}s</text>',
               '<text x="30" y="57">Native event times; Phira spatial layout excluded. Hold spans are clipped at the window edge.</text>']
        for pane, (title, events) in enumerate(groups):
            top = 145 + pane * 370
            svg += [f'<text x="30" y="{top}" style="font-size:18px">{title}: {len(events)} notes</text>',
                    f'<defs><clipPath id="pane{pane}"><rect x="{left}" y="{top + 20}" width="{right-left}" height="300"/></clipPath></defs>']
            for second in range(9):
                px = left + second / 8 * (right - left)
                svg.append(f'<path d="M {px} {top+30} V {top+310}" stroke="#dce2e9"/>')
            for k, kind in enumerate(kinds):
                y = top + 65 + k * 65
                svg.append(f'<text x="35" y="{y+5}">{kind}</text>')
                subset = [e for e in events if e['type'] == kind]
                svg.append(f'<g clip-path="url(#pane{pane})" fill="{colors[k]}" stroke="{colors[k]}">')
                for event in subset:
                    px = x(event['onset_us'])
                    if kind == 'hold':
                        svg.append(f'<path d="M {px:.3f} {y} H {x(event["end_us"]):.3f}" stroke-width="3"/>')
                    svg.append(f'<circle cx="{px:.3f}" cy="{y}" r="3"/>')
                svg.append('</g>')
        for second in range(9):
            px = left + second / 8 * (right - left)
            svg.append(f'<text x="{px}" y="1260" text-anchor="middle">{second}</text>')
        svg += ['<text x="735" y="1305" text-anchor="middle">Seconds relative to the validation window</text>', '</svg>']
        path = ART / 'review' / f'window-{j:02d}.svg'; path.parent.mkdir(exist_ok=True)
        path.write_text('\n'.join(svg))
        subprocess.run(['/usr/bin/qlmanage', '-t', '-s', '1400', '-o', str(path.parent), str(path)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=30)
        png = path.with_suffix('.svg.png')
        if not png.exists():
            raise FileNotFoundError('Native SVG renderer did not write a PNG')
        paths.append(str(png.relative_to(ROOT)))
    return paths


def main():
    plan = json.loads(PLAN.read_text()); report = json.loads(REPORT.read_text())
    if report['status'] != 'evaluated':
        raise ValueError('Cannot close an unfinished evaluation')
    verified = {}
    for path, expected in {**plan['source_sha256'], **plan['target_sha256'],
                           **report.get('evaluation_source_sha256', {})}.items():
        actual = sha(ROOT / path)
        if actual != expected:
            raise ValueError('Frozen source changed: ' + path)
        verified[path] = actual
    result = {}
    for arm in ARMS:
        result[arm] = {task: metrics(json.loads((ART / arm / f'{task}-final.json').read_text()))
                       for task in ('draft', 'completion')}
        low = json.loads((ART / arm / 'density_low-final.json').read_text())
        high = json.loads((ART / arm / 'density_high-final.json').read_text())
        density = []
        for a, b in zip(low, high):
            if a.get('attempted') and b.get('attempted'):
                width = (a['request']['end_us'] - a['request']['start_us']) / 1e6
                x, y = len(a['prediction']['events']) / width, len(b['prediction']['events']) / width
                density.append(dict(index=a['index'], low=x, high=y, direction=y > x,
                                    low_in_band=2 <= x < 3, high_in_band=6 <= y < 8))
        result[arm]['density'] = dict(rows=density, direction=sum(x['direction'] for x in density),
                                     in_band=sum(x['low_in_band'] + x['high_in_band'] for x in density))
    base, candidate = (result[a]['draft'] for a in ARMS)
    gates = dict(full_note_gain=candidate['full_note_f1'] - base['full_note_f1'] >= plan['gates']['draft_full_f1_gain'],
        onset_gain=candidate['onset_f1'] - base['onset_f1'] >= plan['gates']['draft_onset_f1_gain'],
        completion=candidate['complete'] / candidate['windows'] >= plan['gates']['draft_complete_fraction'],
        non_tap_types=sum(candidate['matched_types'].get(k, 0) > 0 for k in ('hold', 'flick', 'drag')) >= 2,
        reference_gain=result['bundle']['completion']['full_note_f1'] - result['baseline']['completion']['full_note_f1'] >= .02,
        density=result['bundle']['density']['direction'] >= 3)
    report.update(status='completed', complete=True, passed=all(gates.values()), decision_gates=gates,
                  review=result, source_verification_count=len(verified), source_verification_passed=True,
                  visual_artifacts=plots(), human_playtest=False, new_cloud_compute_calls=0,
                  promotion='No production promotion; saved checkpoints belong to this small-data experiment only',
                  attribution='Combined method result; legacy single-dimension failure counters unchanged because attribution is mixed')
    write(REPORT, report)
    write(ART / 'verification.json', dict(source_sha256=verified, plan_sha256=sha(PLAN),
        prediction_sha256={str(p.relative_to(ROOT)): sha(p) for a in ARMS for p in (ART / a).glob('*-final.json')}))
    lines = ['# 本地小规模组合实验', '',
        '用户授权全部手段联合；比较完整组合方案与原版，不归因到某一单项。', '',
        f"训练24歌曲组件/39谱/174窗，验证8独立歌/16窗；两臂各{report['updates']['baseline']}更新。",
        f"参数：原版{report['parameters']['baseline']:,}，组合{report['parameters']['bundle']:,}。",
        f"本地训练含中途评价{report['training_wall_s']:.1f}秒，最终评价{report['evaluation_wall_s']:.1f}秒；云调用0。", '',
        '| 指标 | 原版 | 组合 |', '|---|---:|---:|']
    for label, key in [('50ms完整音符F1', 'full_note_f1'), ('50ms起音F1', 'onset_f1'),
                       ('完整窗口', 'complete'), ('非空窗口', 'nonempty'), ('输出/真值数量比', 'density_ratio'),
                       ('正间隔<10ms比例', 'close_positive_fraction'), ('最大同刻数量', 'max_same_time')]:
        lines.append(f'| {label} | {base[key]:.4f} | {candidate[key]:.4f} |')
    lines += ['', '匹配类型数（hold要求双端点50ms）：', '```json',
              json.dumps({a: result[a]['draft']['matched_types'] for a in ARMS}, ensure_ascii=False, indent=2), '```',
              '', '组合验收：', '```json', json.dumps(gates, ensure_ascii=False, indent=2), '```', '',
              '本轮是8秒局部任务与4秒缺口补全，不是整曲、Mania或真人可玩性结论。条件评测只给事前固定密度请求，没有从隐藏验证谱回填条件。',
              '学习组大小改变生成算法；较少同刻重复只是结构行为，必须结合匹配召回与数量判断。',
              '健康权重未进入质量训练；普通序列任务曝光相同，组合增加了自身历史参考、分头目标和额外参数，不能称单一架构消融。',
              '', '原始曲线、请求产物、重放历史、图表和核验信息在 `artifacts/local-bundle-v1/`；机器报告为 `reports/local_bundle_v1.json`。']
    path = ROOT / 'reports/local_bundle_review_v1.md'; path.write_text('\n'.join(lines) + '\n')
    journal(f"Evaluate/Closeout：两臂各{report['updates']['baseline']}更新，8独立歌16窗；原版/组合完整音符F1={base['full_note_f1']:.6f}/{candidate['full_note_f1']:.6f}，完整窗口{base['complete']}/{candidate['complete']}，组合门{gates}。整体{'通过本轮小规模研究门' if report['passed'] else '未通过完整组合门'}；保留局部结果与所有原始失败证据，不提升生产默认。{len(verified)}源/标签SHA核验通过，0新云；混合归因，旧单维失败计数不重写。报告reports/local_bundle_review_v1.md。")
    print(json.dumps(dict(passed=report['passed'], review=result, gates=gates), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
