"""Recompute excerpt quality and condition bands from saved generated events."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from beatfcn.events.canonical import Event
from beatfcn.events import controls as C
from beatfcn.model.evaluation import score_events


def main():
    raw=json.loads((ROOT/'reports/condition-pair-health-v1.json').read_text())
    rows=[]
    for record in raw['generation']:
        events=[Event.from_json(e) for e in record['generated_events']]
        truth=[Event.from_json(e) for e in record['truth_events']]
        scores=score_events(events,truth,record['mode'])
        if scores!=record['scores']:raise ValueError('saved event scores disagree with recomputation')
        full=scores['edit_utility']['full_note']
        values=C.raw_controls(record['mode'],events,[(record['c0_us'],record['c0_us']+8_000_000)])
        fields={name:dict(requested=band,achieved=values[name],in_band=C.band_of(name,values[name])==band)
                for name,band in record['requested_controls'].items() if name in C.MODE_FIELDS[record['mode']]}
        for name,result in fields.items():
            if result!=record['achieved_controls'][name]:raise ValueError('saved condition scores disagree')
        rows.append({**{k:record[k] for k in ('pair_id','chart_id','step','variant','mode','predicted_types','truth_types','complete','legal')},
                     'full_note_f1':2*full['matched']/(full['pred']+full['true']),
                     'predicted_notes':len(events),'target_notes':len(truth),'conditions':fields})
    first=raw['history'][0];last=raw['history'][-1]
    initial_differences=[]
    for pair in raw['protocol']['pairs']:
        before=[g for g in raw['generation'] if g['pair_id']==pair['pair_id'] and g['step']==0]
        a,b=[g['generated_events'] for g in before]
        initial_differences.append(dict(pair_id=pair['pair_id'],identical_events=a==b,
            same_type_counts=before[0]['predicted_types']==before[1]['predicted_types'],
            different_event_records=sum(x!=y for x,y in zip(a,b))+abs(len(a)-len(b))))
    report=dict(schema='beatfcn.condition_pair_review.v1',source='reports/condition-pair-health-v1.json',
        source_sha256=raw['source_sha256'],quality_promotion=False,probe_passed=raw['probe_passed'],
        initial_loss=first['loss'],final_loss=last['loss'],optimizer_updates=raw['optimizer_updates'],rows=rows,
        initial_condition_effect=initial_differences,
        generation_gate=raw['generation_gate'],condition_gate=raw['condition_gate'],
        supervision_gate=raw['supervised_gate'],data_health_passed=raw['data_health']['passed'],
        checkpoint_saved=raw['checkpoint_saved'],function_wall_s=raw['function_wall_s'],
        limitation='Two training-song pairs; joint controls covary. Does not prove individual-field control, audio dependence, or held-out chart utility.')
    out=ROOT/'reports/condition_pair_review_2026-10-25.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    text=['# 条件配对诊断','',
          '结论：相同音乐和空白上下文下，完整条件向量能够在 100 步记忆训练后选择不同真实目标。未晋升模型。','',
          f"监督 loss {first['loss']:.6f} → {last['loss']:.6f}；100 步梯度有限，全部活跃预测头准确率 100%。四段完整音符 F1 均为 1.0，所有适用请求档位在带。",'',
          '| 目标谱 | 第 0 步完整音符 F1 | 第 100 步 | 移除条件后 |',
          '| --- | --- | --- | --- |']
    for pair in raw['protocol']['pairs']:
        for item in pair['items']:
            cid=item['row']['chart_id']
            fetch=lambda step,variant:next(r for r in rows if r['chart_id']==cid and r['step']==step and r['variant']==variant)
            text.append(f"| {cid} | {fetch(0,'conditioned')['full_note_f1']:.4f} | {fetch(100,'conditioned')['full_note_f1']:.4f} | {fetch(100,'removed')['full_note_f1']:.4f} |")
    text.extend(['','在每对样本中，移除条件后两份输出逐事件完全相同；交换条件后输出逐事件与另一份条件结果完全相同。生成开始时，除 controls 外的所有实际 metadata 和音频读取计划一致，没有人谱历史或目标空间泄漏。','',
        '初始模型在 mania 两个条件下都生成 35 个 tap；Phira 两个条件下都生成 2 tap + 10 drag。拟合后可以切换到 mania 的 38 / 49 个真实音符，和 Phira 的 51 / 39 个真实音符。Phira 一份有 38 drag、另一份有 5 hold，未用后处理强塞配额。','',
        '初始结果并非逐事件完全相同：mania 有 18/35 条事件记录不同（包含键道变化），Phira 有 1/12 条的起音相差 4ms。因此应称数量/类型构成没有响应，不能称条件完全没有任何影响。','',
        '这排除了“条件输入通路根本不能影响解码”的解释，支持优先检查训练中是否有足够迫使模型区分条件的样本。但多个条件同时变化，尚不能证明任意单字段都有效，更不能把两首训练歌的记忆当泛化。','',
        '下一项正式干预应只改变正样本组织：相同音频/可编辑范围的多谱目标，配以同范围全部条件；原始类型和艺术 drag 保留。难度重采样、损失调整、模型结构变更分别实验。需要训练音频完全隔离的验证集、固定非答案派生请求和逐字段可行性审查，随后才做 1000 步复盘。','',
        f"函数 {raw['function_wall_s']:.2f}s；峰值显存 allocated {raw['peak_allocated_gib']:.3f} GiB、reserved {raw['peak_reserved_gib']:.3f} GiB。无拟合检查点保存。",
        '',f"完整逐事件和张量证据：`{report['source']}`。"])
    (ROOT/'reports/condition_pair_review_2026-10-25.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(probe_passed=report['probe_passed'],records=len(rows),condition_gate=report['condition_gate'],quality_promotion=False),ensure_ascii=False))


if __name__=='__main__':main()
