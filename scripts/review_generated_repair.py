"""Independent, read-only error review for the completed R15 objective probe."""
import hashlib
import json
from pathlib import Path

from scripts.acoustic_probe_metrics import event_metrics
from scripts.audit_lane_feasibility import occupancy_profile

ROOT = Path(__file__).resolve().parents[1]


def main():
    base_path, run_path = ROOT/'reports/r13_acoustic_identifiability.json', ROOT/'reports/r15_fit.json'
    baseline, run = json.loads(base_path.read_text()), json.loads(run_path.read_text())
    if not run['complete'] or run['optimizer_updates'] != 100 or run.get('error'):
        raise ValueError('R15 is not a completed 100-update probe')
    rows = []
    for original, candidate in zip(baseline['generation']['original']['cores'], run['generation']['original']['cores']):
        if original['identity'] != candidate['identity'] or original['target_events'] != candidate['target_events']:
            raise AssertionError('comparison identity changed')
        truth, mode = candidate['target_events'], candidate['identity']['mode']
        rounds = []
        for r in candidate['rounds']:
            measured = event_metrics(r['events'], truth, mode)
            if abs(measured['full_note_f1'] - r['full_note_f1']) > 1e-12:
                raise AssertionError('independent F1 disagrees')
            rounds.append(dict(round=r['round'], **measured,
                lane_independent_peak=max(x['required_lanes'] for x in occupancy_profile(r['events']))
                    if mode=='mania_4k' and r['events'] else None))
        rows.append(dict(identity=candidate['identity'], rounds=rounds,
            baseline_final_f1=original['rounds'][-1]['full_note_f1'],
            final_f1_delta=rounds[-1]['full_note_f1']-original['rounds'][-1]['full_note_f1'],
            self_revision_f1_delta=rounds[-1]['full_note_f1']-rounds[0]['full_note_f1']))
    modes = {}
    for mode in ('mania_4k','phira'):
        selected=[r for r in rows if r['identity']['mode']==mode]
        modes[mode]=dict(baseline_final_f1=sum(r['baseline_final_f1'] for r in selected)/len(selected),
            final_f1=sum(r['rounds'][-1]['full_note_f1'] for r in selected)/len(selected),
            first_f1=sum(r['rounds'][0]['full_note_f1'] for r in selected)/len(selected),
            final_f1_delta=sum(r['final_f1_delta'] for r in selected)/len(selected),
            per_round=[dict(round=i+1, macro_f1=sum(r['rounds'][i]['full_note_f1'] for r in selected)/len(selected),
                conflicts=sum(r['rounds'][i]['mania_lane_conflicts'] for r in selected),
                types={k:{f:sum(r['rounds'][i]['per_type'][k][f] for r in selected)
                          for f in ('pred','true','matched')} for k in ('tap','hold','flick','drag')}) for i in range(4)])
    bad=run['generation']['known_translated_bad_case']['cores'][0]
    witnesses=[dict(round=r['round'],**max(occupancy_profile(r['events']), key=lambda x:x['required_lanes'])) for r in bad['rounds']]
    summary=dict(id='R15-review',complete=True,decision='reject_and_isolate',model_promotion=False,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (base_path,run_path)},
        optimizer_updates=100,extra_no_grad_forwards=run['extra_no_grad_forwards'],
        pre_intervention_exact=all(r['pre_intervention_loss_difference']==0 for r in run['history'][:3]),
        initial_state_exact=run['initial_state_sha256']==baseline['initial_state_sha256'],
        loss=dict(baseline=baseline['final']['mean_loss'],candidate=run['final']['mean_loss']),
        modes=modes,cores=rows,known_bad_case=dict(identity=bad['identity'],witnesses=witnesses),
        gates=run['gates'],skipped_evaluation=run['skipped_evaluation'],
        wall_s=run['wall_s'],peak_rss_gib=run['peak_rss_gib'],new_cloud_compute_usd=0)
    (ROOT/'reports/r15_review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    text='''# R15：真实生成草稿监督未通过；撤回本轮干预

此轮保持 R13 数据、初始化、模型、损失、优化器、100 次更新与解码不变，仅将每四步一次的合成错误可见内容换成当前模型的停止梯度预测。可见 mask 与原随机序列精确保持，原训练真值仍是监督。前三个干预前更新逐 core loss 精确复现；实际健康检查和13项相关回归通过。

| 原始训练片段，第四轮 | R13 | R15 | 差值 |
| --- | ---: | ---: | ---: |
'''
    for mode in ('mania_4k','phira'):
        m=modes[mode];text+=f"| {mode} 完整 F1 | {m['baseline_final_f1']:.6f} | {m['final_f1']:.6f} | {m['final_f1_delta']:+.6f} |\n"
    text+=f'''
总监督 loss（同一 all-MASK 评价）从基线 {summary['loss']['baseline']:.6f} 变为 {summary['loss']['candidate']:.6f}，原 .1 门槛未通过。模型在训练片段仍未充分拟合，不能据此声称已经收敛或这一方法普遍无效。

R15 的第一轮 mania/Phira F1 为 {modes['mania_4k']['first_f1']:.6f}/{modes['phira']['first_f1']:.6f}，四轮分别降为 {modes['mania_4k']['final_f1']:.6f}/{modes['phira']['final_f1']:.6f}。Phira 最终仍只匹配1/4 hold、0/2 flick；drag匹配15/16，但这不能抵消稀少类型消失。

已知平移坏例的物理占用峰值为12、13、13、12，较R13的8、8、7、9更差。替换训练草稿来源没有修复尾点/类型与完整可行性问题。必要失败证据已充分，未扩测其余平移和音频错配；这些项目保留未验证，不写成通过或实际失败。

真实100步耗时{summary['wall_s']:.2f}秒，峰值进程内存{summary['peak_rss_gib']:.3f}GiB；另有200次无梯度前向，不能称两种训练的总计算量相同。新增云费用$0，没有保存或晋升权重。

决定：撤下本轮拥有的目标干预与临时恢复入口，保留原始逐轮事件、类型统计、头损失与无解证据。不继续调腐坏比例、mask、权重或追加步数来救回R15；下一模型实验切回架构，研究事件属性的条件依赖。R14键道组件仍未测到架构效果，上游无解没有被重新排键解决。正式规模守卫与预算预留不变。

完整证据：[独立复盘](r15_review.json)、[原始模型输出](r15_fit.json)、[资源凭据](r15_fit_resources.json)、[上游失败定位](d33_lane_feasibility_review.md)。
'''
    (ROOT/'reports/r15_review.md').write_text(text)
    print(json.dumps({k:summary[k] for k in ('decision','pre_intervention_exact','loss','wall_s')},ensure_ascii=False))


if __name__=='__main__':
    main()
