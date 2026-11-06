"""Independent R16 audit from saved raw predictions; no model import or updates."""
import hashlib
import json
from pathlib import Path

from scripts.acoustic_probe_metrics import event_metrics, physical_events
from beatfcn.model.lane_draft import make_lane_draft, maximum_overlap

ROOT = Path(__file__).resolve().parents[1]


def audit(doc):
    cases=[]
    for core in doc['generation']['original']['cores']:
        rounds=[]
        for raw in core['rounds']:
            metrics=event_metrics(raw['events'],core['target_events'],core['identity']['mode'])
            for key in ('full_note_f1','matched','true','pred','mania_lane_conflicts'):
                if metrics[key]!=raw[key]:raise ValueError(f'saved metric mismatch: {key}')
            for kind, counts in metrics['per_type'].items():
                for key, value in counts.items():
                    if value!=raw['per_type'][kind][key]:raise ValueError('type count mismatch')
                recall=counts['matched']/counts['true'] if counts['true'] else None
                if recall!=raw['per_type'][kind]['recall']:raise ValueError('type recall mismatch')
            rounds.append(dict(round=raw['round'],**metrics))
        cases.append(dict(identity=core['identity'],rounds=rounds))
    modes={}
    for mode in ('mania_4k','phira'):
        rows=[c for c in cases if c['identity']['mode']==mode]
        types={k:{f:sum(c['rounds'][-1]['per_type'][k][f] for c in rows) for f in ('true','pred','matched')}
               for k in ('tap','hold','flick','drag')}
        hold=types['hold']
        modes[mode]=dict(first_f1=sum(c['rounds'][0]['full_note_f1'] for c in rows)/len(rows),
            final_f1=sum(c['rounds'][-1]['full_note_f1'] for c in rows)/len(rows),
            full_hold_f1=2*hold['matched']/(hold['pred']+hold['true']),per_type=types)
        for key in ('first_f1','final_f1','full_hold_f1','per_type'):
            if modes[mode][key]!=doc['summary'][mode][key]:raise ValueError('saved summary mismatch')
    witness=doc['generation']['known_translated_bad_case']['cores'][0]
    peaks=[maximum_overlap(make_lane_draft(r['events'])) if r['events'] else 0 for r in witness['rounds']]
    tails=[]
    for raw in witness['rounds']:
        physical=physical_events(raw['events'],witness['identity'])
        tails.append(dict(round=raw['round'],anchors=[dict(true_head_us=head,true_tail_us=55_608_000,
            nearby_predicted_holds=[e for e in physical if e['type']=='hold' and abs(e['onset_us']-head)<=20_000])
            for head in (55_308_000,55_408_000,55_508_000)]))
    rare=[]
    for core in doc['generation']['original']['cores']:
        if core['identity']['mode']!='phira':continue
        for truth in core['target_events']:
            if truth['type'] not in ('hold','flick'):continue
            rare.append(dict(identity=core['identity'],truth=truth,rounds=[dict(round=r['round'],
                nearby_predictions=[e for e in r['events'] if abs(e['onset_us']-truth['onset_us'])<=20_000])
                for r in core['rounds']]))
    return dict(modes=modes,mean_full_hold_f1=sum(m['full_hold_f1'] for m in modes.values())/2,
                peaks=peaks,witness=tails,cases=cases,rare_type_transitions=rare)


def main():
    paths={'A':'reports/r16_a_fit.json','B':'reports/r16_b_fit.json'}
    docs={a:json.loads((ROOT/p).read_text()) for a,p in paths.items()}
    for a,d in docs.items():
        if not d['complete'] or d.get('error') or d['optimizer_updates']!=100:raise ValueError(f'{a} not complete100')
    if docs['A']['initial_state_sha256']!=docs['B']['initial_state_sha256'] or docs['A']['initial']!=docs['B']['initial']:
        raise ValueError('paired initialization differs')
    for a,b in zip(docs['A']['generation']['original']['cores'],docs['B']['generation']['original']['cores']):
        if a['identity']!=b['identity'] or a['target_events']!=b['target_events']:raise ValueError('paired targets differ')
    audits={a:audit(d) for a,d in docs.items()}
    gain=audits['B']['mean_full_hold_f1']-audits['A']['mean_full_hold_f1']
    mechanism=gain>=.1 and all(audits['B']['modes'][m]['full_hold_f1']>=audits['A']['modes'][m]['full_hold_f1'] for m in audits['A']['modes'])
    if mechanism!=docs['B']['gates']['mechanism_gain'] or gain!=docs['B']['paired_gain']:raise ValueError('paired mechanism gate mismatch')
    report=dict(id='R16',complete=True,paired_identity_verified=True,raw_metrics_recomputed=True,
        mechanism_passed=mechanism,hold_f1_gain=gain,system_passed=docs['B']['passed'],
        decision='retain_probe_only_no_promotion' if docs['B']['passed'] else 'reject_and_isolate',
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths.values()},
        arms=audits,health_updates=2,fit_updates_per_arm=100,new_cloud_compute_usd=0,formal_training_started=False,
        limitation='Four training songs/eight cores and one overlapping translated witness; no held-out or whole-song quality; 100 steps not a convergence claim.')
    (ROOT/'reports/r16_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# R16：本轮预测事件条件化尾点的配对结果','',
      '两臂容量及初值相同、初始logits精确复现R13；仅B给尾点适配层输入本轮预测起点/类型。全部原监督、合成腐坏、数据、优化器、100步保持。',
      '', '| 指标 | A：未知属性 | B：预测属性 | B−A |','| --- | ---: | ---: | ---: |']
    for mode in ('mania_4k','phira'):
        for key,label in (('full_hold_f1','完整hold F1'),('final_f1','完整事件F1')):
            a=audits['A']['modes'][mode][key];b=audits['B']['modes'][mode][key]
            lines.append(f'| {mode} {label} | {a:.6f} | {b:.6f} | {b-a:+.6f} |')
    lines.extend(['',f'机制主指标两模式完整hold F1均值变化{gain:+.6f}，原门槛+.10且各模式不退化；通过={mechanism}。完整系统联合门槛通过={docs["B"]["passed"]}。',
        '',f'已知平移坏例四轮原始物理占用：A {audits["A"]["peaks"]}，B {audits["B"]["peaks"]}。所有峰值在任何删音、裁尾或排键之前计算。三个真实hold逐轮预测尾点保存在JSON，不以预测配对代替真值。'])
    for arm in ('A','B'):
        d=docs[arm];m=audits[arm]['modes'];p=m['phira']['per_type']
        lines.extend(['',f'{arm}：最终all-MASK loss {d["final"]["mean_loss"]:.6f}；mania第一→四轮 {m["mania_4k"]["first_f1"]:.6f}→{m["mania_4k"]["final_f1"]:.6f}，Phira {m["phira"]["first_f1"]:.6f}→{m["phira"]["final_f1"]:.6f}。Phira hold {p["hold"]["matched"]}/{p["hold"]["true"]}、flick {p["flick"]["matched"]}/{p["flick"]["true"]}。',
            f'各100次更新、800次训练前向；模型运行{d["wall_s"]:.2f}秒、峰值RSS {d["peak_rss_gib"]:.3f}GiB。扩展评测状态：{d.get("skipped_evaluation","已执行，见原报告")}'])
    lines.extend(['','此为固定预算快速拟合对照，不是无限计算下的表达能力判断，也不是整曲或留出歌曲能力。数据规模门槛、原质量门槛、历史失败和预算预留不变；新增云费用$0。',
        '',f'决定：{report["decision"]}。组件与系统门槛分开裁决；不通过则归档本轮拥有的模型入口并从默认路径撤下，不继续调适配层宽度、权重或步数。',
        '', '证据：[独立重算](r16_review.json)、[A原始预测](r16_a_fit.json)、[B原始预测](r16_b_fit.json)、[真实健康](r16_health_v2.json)。'])
    (ROOT/'reports/r16_review.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='arms'},ensure_ascii=False))


if __name__=='__main__':main()
