"""Compare the saved R10/R11 single-variable objective control; no new inference."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from beatfcn.events.canonical import Event
from beatfcn.model.event_canvas import RhythmEvent


ROOT = Path(__file__).resolve().parents[1]


def main():
    base = json.loads((ROOT / 'reports/p2b_fit.json').read_text())
    result_path = ROOT / 'reports/p2b_occupancy_fit.json'
    candidate = json.loads(result_path.read_text())
    if base['initial_state_sha256'] != candidate['initial_state_sha256']:
        raise ValueError('initial model differs')
    catalog_path = ROOT / 'inputs/r09_positive_pairs/catalog.pt'
    if hashlib.sha256(catalog_path.read_bytes()).hexdigest() != candidate['input_sha256'][str(catalog_path.relative_to(ROOT))]:
        raise ValueError('catalog differs')
    catalog = torch.load(catalog_path, map_location='cpu', weights_only=False)
    pairs = {p['pair_id']: p for p in catalog['train_pairs']}
    cores = []
    for a, b in zip(base['generation']['cores'], candidate['generation']['cores']):
        if a['identity'] != b['identity']:
            raise ValueError('target/exposure identity differs')
        item = b['identity']['item']
        entry = pairs[item['pair_id']]['entries'][item['side']]
        truth = Counter(RhythmEvent.from_event(Event.from_json(e), item['mode']) for e in entry['truth_events'])
        rows = []
        for r in b['rounds']:
            pred = Counter(RhythmEvent(**e) for e in r['events'])
            rows.append(dict(round=r['round'], true=r['true'], pred=r['pred'],
                matched=r['matched'], full_note_f1=r['full_note_f1'],
                exact_onset_group_recall=r['exact_onset_group_recall'],
                exact_missing=sum((truth - pred).values()), exact_excess=sum((pred - truth).values()),
                per_type=r['per_type'], mania_lane_conflicts=r['mania_lane_conflicts']))
        cores.append(dict(identity=b['identity'], rounds=rows,
                          baseline_first_count=a['rounds'][0]['pred'], baseline_fourth_count=a['rounds'][-1]['pred']))
    summary = {}
    for number in (1, 4):
        rows = [c['rounds'][number - 1] for c in cores]
        summary[str(number)] = dict(true=sum(r['true'] for r in rows), pred=sum(r['pred'] for r in rows),
            matched=sum(r['matched'] for r in rows), exact_missing=sum(r['exact_missing'] for r in rows),
            exact_excess=sum(r['exact_excess'] for r in rows),
            macro_full_note_f1=sum(r['full_note_f1'] for r in rows) / len(rows),
            macro_exact_group_recall=sum(r['exact_onset_group_recall'] for r in rows) / len(rows),
            hold_matched=sum(r['per_type']['hold']['matched'] for r in rows),
            hold_true=sum(r['per_type']['hold']['true'] for r in rows))
    report = dict(schema='beatfcn.p2b_occupancy_review.v1', complete=True,
        decision='reject_proper_slot_nll_configuration', controlled_change='kind loss normalization only',
        initial_weights_identical=True, optimizer_updates=candidate['optimizer_updates'],
        code_plan=candidate['plan_sha256'], source_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        gates=candidate['gates'], rounds=summary, cores=cores,
        final_loss=candidate['final']['mean_loss'], model_wall_s=candidate['wall_s'],
        peak_rss_gib=candidate['peak_rss_gib'], new_paid_compute_usd=0,
        findings=[
            'The unweighted objective eliminates extra predictions in the selected cores but collapses true occupancy and all six holds.',
            'Four-round refinement further reduces recall; an editable network does not automatically learn useful repair.',
            'Head/lane/tail classification at true slots remains 100%, so this necessary test isolates a major occupancy/type decision problem.',
            'Lower scalar loss is not quality improvement; R10 and R11 scalar objectives differ and must not be compared as a common score.',
            'Neither objective configuration passes, and neither is promoted. No interpolation of class weights or longer fitting is queued.'
        ],
        next_route='Representation audit of learned block cardinality / onset-group ownership; infer counts from model output, never hidden counts or requested hard quotas. Freeze a single candidate only after full rhythm/hold/multiplicity roundtrip and leak tests.',
        scope='Fixed training memorisation, not held-out architecture/control quality.')
    (ROOT / 'reports/p2b_occupancy_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['# R11：占用损失单变量对照', '',
        '**否决。** 同初始化、同真实四core、同主干、同100步、同解码；唯一改变为kind损失的槽位归一化。没有新增付费或保存权重。', '',
        '|片段|真实数|R10四轮|R11四轮|R11完整F1|R11精确组召回|', '|---|---:|---:|---:|---:|---:|']
    for c in cores:
        r = c['rounds'][-1]
        lines.append(f'|{c["identity"]["chart_id"]}|{r["true"]}|{c["baseline_fourth_count"]}|{r["pred"]}|{r["full_note_f1"]:.3f}|{r["exact_onset_group_recall"]:.3f}|')
    a, b = summary['1'], summary['4']
    lines += ['', f'第一轮只保留{a["matched"]}/72个音符；四轮只保留{b["matched"]}/72个，六个hold全部丢失。没有多余音符，但这不是成功：一个Phira片段成为空谱，四轮修订进一步减少召回。', '',
        '真实槽位的时间、尾点和mania键道分类仍全对，缺失来自“是否存在/属于哪类”的决定。两种权重分别表现为过量和压空；新证据支持重审独立槽位的占用表示，不能宣称已证实任何替代架构。', '',
        '下一步研究并验证模型预测区间事件数、起音组归属的表示，减少大量独立EMPTY决策。数量必须由模型从音频、条件与已知上下文预测，不能从隐藏真值填入，也不能按用户密度强制删补来制造控制有效。保留原始毫秒、Phira重数、艺术drag和跨core hold。先回环/泄漏/真实小批，仍不进入长训。', '',
        '原门槛全部保留，loss=.35756仍不达<=.1，且稀有类型和自由生成门槛失败。不同损失的数值大小不可当公平质量排名；这一结果只限四个记忆片段。']
    (ROOT / 'reports/p2b_occupancy_review.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'decision':report['decision'],'gates':report['gates'],'rounds':summary}, ensure_ascii=False))


if __name__ == '__main__':
    main()
