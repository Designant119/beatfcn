"""Audit the fixed1000 matched access experiment from saved artifacts only."""
import json
from pathlib import Path

from scripts.run_local_portfolio_mamba_health import sha, write


ROOT = Path(__file__).resolve().parents[1]


def main():
    plan_path = ROOT / 'bench/rhythm-v2/local-access-comparison-v1.json'
    quality_path = ROOT / 'reports/local_access_quality_v1.json'
    review_path = ROOT / 'reports/local_access_quality_review_v1.json'
    markdown = ROOT / 'reports/local_access_quality_review_v1.md'
    if review_path.exists() or markdown.exists():
        raise FileExistsError('Preserve the experiment review')
    plan, quality = (json.loads(p.read_text()) for p in (plan_path, quality_path))
    if quality['status'] != 'completed' or quality['health_weights_used']:
        raise ValueError('Require completed fresh quality run')
    if quality['plan_sha256'] != sha(plan_path):
        raise ValueError('Plan identity mismatch')
    for path, expected in plan['source_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('Frozen source drift: ' + path)
    summaries, records, identities = {}, {}, {}
    for arm in ('global', 'mixed'):
        result = quality['arms'][arm]
        if result['status'] != 'completed' or result['updates'] != 1000:
            raise ValueError('Incomplete matched endpoint')
        if result['initial_weight_sha256'] != plan['initial_weight_sha256'][arm]:
            raise ValueError('Initialization identity mismatch')
        folder = ROOT / 'artifacts/local-access-comparison-v1' / arm
        if sha(folder / 'quality-1000.pt') != result['final_checkpoint_sha256']:
            raise ValueError('Checkpoint identity mismatch')
        summaries[arm] = result['evaluations']['1000']
        records[arm] = {task: json.loads((folder / f'quality-1000-{task}.json').read_text())
                        for task in summaries[arm]}
        identities.update({str(p.relative_to(ROOT)): sha(p) for p in folder.glob('quality-1000*')})
    for task in records['global']:
        for left, right in zip(records['global'][task], records['mixed'][task], strict=True):
            for key in ('index', 'target', 'request', 'component'):
                if left[key] != right[key]:
                    raise ValueError(f'Unmatched {task} case: {key}')
    baseline, candidate = (summaries[arm]['draft'] for arm in ('global', 'mixed'))
    deltas = {key: candidate[key] - baseline[key]
              for key in ('full_note_f1', 'onset10', 'onset50', 'density_ratio')}
    wins = sum(value['full_note_f1'] > baseline['components'][component]['full_note_f1']
               for component, value in candidate['components'].items())
    thresholds = plan['core_screen']
    gates = dict(full_f1=deltas['full_note_f1'] >= thresholds['full_f1_gain'],
                 onset10=deltas['onset10'] >= -thresholds['onset10_max_regression'],
                 component_wins=wins >= thresholds['minimum_component_full_f1_wins'])
    failures = [dict(arm=arm, task=task, index=row['index'], failure=row['prediction']['failure'])
                for arm, tasks in records.items() for task, rows in tasks.items()
                for row in rows if not row['prediction']['complete']]
    review = dict(id=plan['id'], status='reviewed', main_hypothesis_supported=all(gates.values()),
        optimizer_updates_in_review=0, new_inference_calls=0, health_weights_used=False,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in (plan_path, quality_path, Path(__file__))},
        artifact_sha256=identities, parameter_counts=plan['parameters'],
        initial_weight_sha256=plan['initial_weight_sha256'], summaries=summaries,
        mixed_minus_global=deltas, component_full_f1_wins=wins,
        preregistered_gates=gates, preregistered_thresholds=thresholds, failed_requests=failures,
        empty_counts={arm: [len(row['prediction']['events']) for row in tasks['empty']]
                      for arm, tasks in records.items()},
        reference_full_gain={arm: tasks['completion']['full_note_f1'] - tasks['reference_masked']['full_note_f1']
                             for arm, tasks in summaries.items()},
        worst_windows={arm: sorted(tasks['draft'], key=lambda row: row['scores']['full_note_f1'])[:4]
                       for arm, tasks in records.items()},
        interpretation='This tests two additional temporal attention masks on one common eight-block mel-only skeleton. It does not isolate depth/width against old models, add whole-song input or test emotion understanding.',
        cloud_compute_calls=0, confirmation_scored=False)
    write(review_path, review)
    lines = ['# 同参数访问分工：固定1000更新审查', '',
        '两臂各16,696,116参数、相同初态、相同训练请求流，健康权重丢弃。'
        'global=`CCGCGCGC`，mixed=`CCWCGCWC`；只改变两层额外canvas注意力mask。', '',
        '| 开发16组件／32窗 | global | mixed | mixed−global |', '|---|---:|---:|---:|']
    for title, key in [('完整音符F1', 'full_note_f1'), ('onset10', 'onset10'),
                       ('onset50', 'onset50'), ('预测／真值数量比', 'density_ratio')]:
        lines.append(f'| {title} | {baseline[key]:.6f} | {candidate[key]:.6f} | {deltas[key]:+.6f} |')
    lines += ['', f'组件完整F1改善{wins}/16。事前核心门：完整F1至少+.015、onset10退化不超过.01、'
        f'至少10/16组件完整F1改善；实际判定{gates}，整体未通过。'
        '不把完整F1小幅上升单独解释为通过，也不改门槛或追加步数。', '',
        f'两个空窗分别输出：{review["empty_counts"]}；失败请求{len(failures)}。'
        f'参考完整F1增益：{review["reference_full_gain"]}。', '',
        '| 类型 | global F1 | mixed F1 | global TP | mixed TP |', '|---|---:|---:|---:|---:|']
    for kind in ('tap', 'hold', 'flick', 'drag'):
        left, right = baseline['per_type'][kind], candidate['per_type'][kind]
        lines.append(f'| {kind} | {left["f1"]:.6f} | {right["f1"]:.6f} | {left["tp"]} | {right["tp"]} |')
    lines += ['', '两臂均未使用预训练声学参数，确有跨歌定位信号；但共同八层骨架、归一化、学习率、种子等'
        '与旧mel-only不同，不能把它们与旧模型的差值单独归因为深度或预训练。'
        '本次不支持“将这两层全局注意力限制为窗口更好”的具体假设，不否定LFM3、混合算子或深度分配整体。', '',
        '上游编码／地址读取已全局访问；本次没有新增整曲上下文。没有试听、情绪标签、编辑计时或原生试玩；'
        '确认集未评分，云调用0。最差原生窗口、逐组件及所有身份见JSON审查。', '']
    markdown.write_text('\n'.join(lines))
    print(json.dumps({k: review[k] for k in ('mixed_minus_global', 'component_full_f1_wins', 'preregistered_gates', 'main_hypothesis_supported')}, indent=2))


if __name__ == '__main__':
    main()
