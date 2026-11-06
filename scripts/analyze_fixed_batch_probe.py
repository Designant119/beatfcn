"""Audit saved real-input and memorisation evidence without cloud computation."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from beatfcn.train.probe import assess_generation,assess_memorisation


def main():
    source=ROOT/'reports/fixed-batch-health-v1.json'
    raw=json.loads(source.read_text())
    protocol=raw['protocol']['probe'];last_step=protocol['steps']
    supervised=assess_memorisation(raw['history'],raw['gradients'],protocol['supervised_gate'],last_step)
    generated=assess_generation([r for r in raw['generation'] if r['step']==last_step],
                               len(raw['protocol']['rows']),protocol['generation_gate']['full_note_f1_at_least'])
    nonfinite={k:v for k,v in raw['data_health']['tensors'].items() if v['nan']+v['posinf']+v['neginf']}
    rows=[]
    for row in raw['generation']:
        teacher=next(t for t in raw['per_core_teacher_forced'] if
                     (t['chart_id'],t['c0_us'],t['step'])==(row['chart_id'],row['c0_us'],row['step']))
        full=row['scores']['edit_utility']['full_note']
        decisions=row['diagnostics']['note_decisions']
        rows.append(dict(chart_id=row['chart_id'],mode=row['mode'],step=row['step'],
            predicted_types=row['predicted_types'],truth_types=row['truth_types'],
            predicted_notes=row['scores']['pred'],target_notes=row['scores']['true'],
            onset_f1_20ms=row['scores']['onset_20ms']['f1'],
            full_note_f1_20ms=2*full['matched']/(full['pred']+full['true']),
            reference_edit_saved_fraction=full['saved_fraction'],
            teacher_type_accuracy=teacher['decisions']['type']['accuracy'],
            teacher_type_confusion=teacher['decisions']['type']['confusion_truth_rows_pred_columns'],
            teacher_type_recall={name:(conf[i]/sum(conf) if sum(conf) else None)
                                 for i,(name,conf) in enumerate(zip(('tap','hold','flick','drag'),
                                    teacher['decisions']['type']['confusion_truth_rows_pred_columns']))},
            generation_type_proposed=decisions['proposed_by_type'],generation_type_emitted=decisions['emitted_by_type'],
            hold_to_tap=decisions['hold_to_tap'],hold_end_failures=decisions['hold_end_failures'],
            complete=row['complete'],legal=row['legal']))
    result=dict(source=str(source.relative_to(ROOT)),source_sha256=raw['source_sha256'],
        real_input_tensor_count=len(raw['data_health']['tensors']),nonfinite_tensors=nonfinite,
        optimizer_updates=raw['optimizer_updates'],supervised_gate=supervised,generation_gate=generated,
        rows=rows,loss_curve=[dict(step=r['step'],loss=r['loss']) for r in raw['history']],
        all_gradient_norms_finite=all(g['finite'] for g in raw['gradients']),
        peak_allocated_gib=raw['peak_allocated_gib'],peak_reserved_gib=raw['peak_reserved_gib'],
        function_wall_s=raw['function_wall_s'],checkpoint_saved=raw['checkpoint_saved'],
        quality_promotion=False,
        conclusion='Real fixed-batch memorisation passes; baseline type mistakes already occur with correct teacher history. '
                   'These two examples do not establish held-out musical understanding or condition control. '
                   'No hold-to-tap fallback occurred in these four generations; it does not explain their type collapse.')
    (ROOT/'reports/fixed_batch_review_2026-10-25.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    lines=['# 固定真实 batch 探针复盘','',
           '结论：输入与记忆能力探针通过；产品实用性和泛化尚未通过。没有保存或晋升拟合后的权重。','',
           f"真实输入 {result['real_input_tensor_count']} 个张量，非有限值张量 {len(nonfinite)} 个。100 步监督 loss 从 {raw['history'][0]['loss']:.6f} 降至 {raw['history'][-1]['loss']:.6f}，所有活跃预测头准确率 100%，100 次梯度检查均有限。",'',
           '| 步数 | mania 完整音符 F1 | Phira 完整音符 F1 |',
           '| --- | --- | --- |']
    for step in (0,last_step):
        values=[r['full_note_f1_20ms'] for r in rows if r['step']==step]
        lines.append(f'| {step} | {values[0]:.4f} | {values[1]:.4f} |')
    lines.extend(['','完整音符指标使用 20ms 容差，要求类型、mania 键道及 hold 两端均匹配。这里只评价两段训练 core，前文给正确的人谱历史；不是从零整曲 benchmark。','',
        '## 具体错误模式','',
        '- 基线 mania 生成 58 个音符，目标 32；两个生成 hold 均未匹配到目标 hold。正确历史下的两个真实 hold 也均被类型头判为 tap，键道准确率仅 50%。',
        '- 基线 Phira 生成 21 个 tap，其余类型为零。即使提供正确历史，类型准确率也只有 34.375%；真实 hold/flick/drag 的类型召回均为零。因此这两个片段的类型问题不只发生于自由生成阶段。',
        '- 四次生成没有 hold→tap 回退、端点失败或超时。不能把该片段的类型消失归咎于尾点修复前的回退；也不能据此推断所有谱都没有边界问题。',
        '- 100 步后两段各生成 32 个音符，类型构成与目标一致，完整音符 F1=1，长按尾误差为零。输入编码、梯度和解码至少能够承载这些例子的正确答案。','',
        '## 限制与下一步','',
        '这没有证明注意力/音频表征已经足够，也没有证明条件起效：模型可以记住固定两首歌。下一轮继续数据层的条件支持检验，先固定相同音频/上下文，只改变有真实目标支持的条件，防止靠歌曲或历史识别答案。完整窗口全条件标注与难度重采样分开。','',
        f"函数耗时 {raw['function_wall_s']:.2f}s，峰值显存 allocated {raw['peak_allocated_gib']:.3f} GiB、reserved {raw['peak_reserved_gib']:.3f} GiB。小 batch 诊断不是吞吐实验；利用率不能用于宣称训练性能改善。$0.30229333 预留保留，未将函数秒数当已结算费用。",
        '',f"证据：`{source.relative_to(ROOT)}`；源码 SHA256 `{raw['source_sha256']}`。"])
    (ROOT/'reports/fixed_batch_review_2026-10-25.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ('real_input_tensor_count','nonfinite_tensors','supervised_gate','generation_gate','quality_promotion')},ensure_ascii=False))


if __name__=='__main__':main()
