"""Summarize the secondary readout audit without more inference or updates."""
import json
from pathlib import Path

from scripts.run_local_portfolio_mamba_health import sha, write


ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'reports/local_package_readout_v1.json'
    out = ROOT / 'reports/local_package_readout_review_v1.json'
    markdown = ROOT / 'reports/local_package_readout_review_v1.md'
    if out.exists() or markdown.exists():
        raise FileExistsError('Preserve the readout review')
    report = json.loads(source.read_text())
    if report['status'] != 'completed' or report['optimizer_updates'] != 0:
        raise ValueError('Require completed zero-update readout audit')
    for path, digest in report['artifact_sha256'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Readout artifact drift: ' + path)
    summaries = report['summaries']
    components = summaries['draft-sample-0']['components']
    gains = {c: sum(summaries[f'draft-sample-{i}']['components'][c]['onset10']
        - summaries[f'draft-permuted-{i}']['components'][c]['onset10'] for i in range(4)) / 4
        for c in components}
    types = {}
    for kind in ('tap', 'hold', 'flick', 'drag'):
        rows = [summaries[f'draft-sample-{i}']['per_type'][kind] for i in range(4)]
        types[kind] = {key: sum(row[key] for row in rows) / 4 for key in ('tp', 'pred', 'true', 'f1')}
    review = dict(status='reviewed', optimizer_updates=0, new_inference_calls=0,
        checkpoint_sha256=report['checkpoint_sha256'], source_sha256=sha(source),
        wall_s=report['wall_s'], four_sample_means=report['four_sample_means'],
        component_onset10_gain_over_permutation=gains,
        positive_components=sum(v > 0 for v in gains.values()),
        sample0=summaries['draft-sample-0'], type_means=types,
        empty_counts_by_sample=[sum(summaries[f'empty-sample-{i}']['predicted_types'].values()) for i in range(4)],
        incomplete_requests=sum(v['attempted'] - v['complete'] for v in summaries.values()),
        interpretation='Sub-mode probability contains temporal information on this reused development panel; it does not establish calibrated chart decisions, emotion understanding or cloud readiness.',
        primary_endpoint_unchanged='Original fixed2000 count-mode fullF1 .008073 and onset10 .005717 are not replaced by this post1000-designed secondary audit.',
        cloud_compute_calls=0, confirmation_scored=False)
    write(out, review)
    sample, null = (report['four_sample_means'][kind] for kind in ('sample', 'permuted'))
    lines = ['# 事件包冻结2000权重：次要祖先采样检查', '',
        '在看到1000步稀疏输出后单独登记；0更新、温度1、4个固定种子全部等权，不择优。'
        '原主终点的mode完整F1 .008073、onset10 .005717保持不变。', '',
        '| 16开发组件／32窗，4样本均值 | 原时间地址 | 固定帧置换 |',
        '|---|---:|---:|']
    for title, key in [('完整音符F1', 'full_note_f1'), ('onset10', 'onset10'),
                       ('onset50', 'onset50'), ('预测／真值数量比', 'density_ratio')]:
        lines.append(f'| {title} | {sample[key]:.6f} | {null[key]:.6f} |')
    lines += ['',
        f'onset10相对帧置换增益 {sample["onset10"] - null["onset10"]:+.6f}；'
        f'组件方向为正 {review["positive_components"]}/16。两侧数量比接近，因此不能只用多发音符解释全部定位差。'
        '这是反复使用开发集上的探索性时序证据，未作统计确认或全曲结论。', '',
        f'两空窗合计误发，4个样本分别为 {review["empty_counts_by_sample"]}；'
        'hold与类型仍弱。接近真值的总体数量不能代替空窗抑制或协调的单张谱。', '',
        '| 类型，4样本均值 | TP | 预测 | 真值 | F1 |', '|---|---:|---:|---:|---:|']
    for kind, v in types.items():
        lines.append(f'| {kind} | {v["tp"]:.2f} | {v["pred"]:.2f} | {v["true"]:.2f} | {v["f1"]:.6f} |')
    lines += ['', '没有加载健康权重，没有新增训练。概率、每个样本与置换输出、完整来源SHA均已保存。'
        '确认集未评分、云调用0、未试听／人工编辑／试玩。', '',
        '可支持的判断是：逐格mode确实漏读了部分可定位概率；仍不能把当前模型称为可靠草稿生成器。'
        '不据此自动加步数、挑样本、调温度或启动H100。', '']
    markdown.write_text('\n'.join(lines))
    print(json.dumps({k: review[k] for k in ('positive_components', 'empty_counts_by_sample', 'incomplete_requests')}, indent=2))


if __name__ == '__main__':
    main()
