"""Read saved R12 predictions; no model execution, parameter changes or training."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def key(event):
    return tuple(event[k] for k in ('onset_us', 'end_us', 'type', 'lane'))


def main():
    report = json.loads((ROOT / 'reports/p2c_fit.json').read_text())
    term_losses = defaultdict(list)
    for row in report['final']['cores']:
        for name, loss in row['terms'].items():
            term_losses[name].append(loss)
    residuals = {name: dict(mean_over_applicable_cores=sum(values) / len(values),
                           contribution_to_reported_mean=sum(values) / 4,
                           geometric_mean_target_probability=math.exp(-sum(values) / len(values)))
                 for name, values in term_losses.items()}
    rows = []
    for row in report['generation']['cores']:
        truth = Counter(key(e) for e in row['target_events'])
        rounds = []
        for generation in row['rounds']:
            pred = Counter(key(e) for e in generation['events'])
            rounds.append(dict(round=generation['round'],
                pred=generation['pred'], true=generation['true'], matched=generation['matched'],
                full_note_f1=generation['full_note_f1'],
                exact_onset_group_recall=generation['exact_onset_group_recall'],
                mania_lane_conflicts=generation['mania_lane_conflicts'],
                exact_missing=list((truth - pred).elements()),
                exact_extra=list((pred - truth).elements()), per_type=generation['per_type'],
                block_count_errors=[dict(block=i, pred=p, true=t) for i, (p, t) in enumerate(
                    zip(generation['block_counts'], row['target_block_counts'])) if p != t]))
        rows.append(dict(identity=row['identity'], rounds=rounds))
    comparisons = []
    for experiment, path in [('R10', 'reports/p2b_fit.json'), ('R11', 'reports/p2b_occupancy_fit.json'),
                             ('R12', 'reports/p2c_fit.json')]:
        result = json.loads((ROOT / path).read_text())
        finals = [x['rounds'][-1] for x in result['generation']['cores']]
        comparisons.append(dict(experiment=experiment, fixed_training_cores=4,
            macro_full_note_f1=sum(x['full_note_f1'] for x in finals) / 4,
            macro_exact_group_recall=sum(x['exact_onset_group_recall'] for x in finals) / 4,
            predicted=sum(x['pred'] for x in finals), true=sum(x['true'] for x in finals),
            matched=sum(x['matched'] for x in finals),
            mania_conflicts=sum(x['mania_lane_conflicts'] for x in finals)))
    review = dict(schema='beatfcn.p2c_probe_review.v1', complete=True,
        passed=report['passed'], decision='registered_joint_gate_failed_no_promotion',
        scope='Four fixed training cores. These are memorisation results, not a held-out benchmark.',
        comparison_scope='Matched seeds/common initial modules/data/100 steps; R12 changes grouped targets and necessary readout jointly.',
        optimizer_updates=report['optimizer_updates'], gates=report['gates'],
        loss_initial=report['initial']['mean_loss'], loss_final=report['final']['mean_loss'],
        loss_reduction=1-report['final']['mean_loss']/report['initial']['mean_loss'],
        residual_loss= residuals, comparisons=comparisons, cores=rows,
        interpretation=[
            'Predicted group and member counts are correct on these four training cores; no hidden count input.',
            'All target heads and tails match exactly; one tap at7.062s has lane1 instead of lane0.',
            'Remaining cross entropy is mainly head confidence and mania lane supervision; no positive theoretical floor from soft targets.',
            'Correct argmax is compatible with nonzero CE. Neither changing its scale nor posthoc temperature would establish better charts.',
            'The old <=.1 gate remains failed. This is useful mechanism evidence, not grounds to change the registered decision.',
            'Condition removal/swap was skipped by the registered early stop; thirteen-field control and held-out utility remain unproven.'],
        new_paid_compute_usd=0, checkpoint_saved=False, model_promotion=False)
    (ROOT / 'reports/p2c_probe_review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
    text = '''# R12 组表示探针复盘

相同四个训练片段、相同公共模块随机初值和100次更新：R12生成72个音符，71个完整匹配，没有多放、没有mania冲突。原门槛里的各头/类型、数量、自由生成均通过；总监督loss为0.511112，高于预注册0.1，因此联合门槛仍失败，不能晋升或直接派发正式训练。

| 100步训练片段拟合 | 生成数/真实数 | 完整匹配 | 宏平均完整F1 | 精确起音组召回 | mania冲突 |
| --- | --- | --- | --- | --- | --- |
'''
    for row in comparisons:
        text += f"| {row['experiment']} | {row['predicted']}/{row['true']} | {row['matched']} | {row['macro_full_note_f1']:.6f} | {row['macro_exact_group_recall']:.6f} | {row['mania_conflicts']} |\n"
    text += '''
这些是训练片段记忆结果，不是独立验证集。R12共同改变组目标与必要读出，不能只把收益归给组数头。第一轮与第四轮在此数据上相同，尚无证据表明回改轮数带来收益。

唯一离散错误：osu:753220在7.062秒的tap应为键道0，预测键道1；其时间和类型均正确。两个Phira片段的tap、hold、flick、drag全部保留，合法drag重数未删。所有80个400ms块的组数准确，没有靠真实数目、密度配额、删点、补点或吸附修正结果。

残余loss的约43.5%来自400分类起音头，约20.8%来自mania键道。正确的argmax不等于目标概率已接近1；这解释了为什么离散结果很好而CE仍高，没有证明继续训练或换温度就会改善独立谱面。本轮没有软目标/正则造成的正下界，不能借此免除原门槛。下一轮不能把loss重标度当进步，也不能事后放宽到0.6。

保留表示回环、误差证据和当前候选的隔离档案；本轮模型配置从工作执行路径撤下，不保存权重。条件移除/交换按预定止损跳过，不能宣称全条件有效。下一步继续原始数据的时钟、解析、去重与低/中/高覆盖验收；用真实分布补充诊断，而非为让小批loss过线追加训练。3500对照保持，正式训练规模门槛未满足。

本轮无新增付费；35.41秒模型用时是本地CPU探针，不能换算为与旧模型同质量GPU加速或3500步报价。完整逐步、逐头、逐事件结果见p2c_fit.json，原门槛未修改。
'''
    (ROOT / 'reports/p2c_probe_review.md').write_text(text)
    print(json.dumps(dict(passed=review['passed'], comparisons=comparisons), ensure_ascii=False))


if __name__ == '__main__':
    main()
