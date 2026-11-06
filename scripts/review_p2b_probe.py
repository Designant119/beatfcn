"""Audit saved R10 predictions without training or generating any new sample."""
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from beatfcn.events.canonical import Event
from beatfcn.model.event_canvas import RhythmEvent


ROOT = Path(__file__).resolve().parents[1]


def key(event):
    return (event.onset_us, event.end_us, event.type, event.lane)


def event_from_key(value):
    return dict(zip(('onset_us', 'end_us', 'type', 'lane'), value))


def main():
    fit_path = ROOT / 'reports/p2b_fit.json'
    fit = json.loads(fit_path.read_text())
    catalog_path = ROOT / 'inputs/r09_positive_pairs/catalog.pt'
    if hashlib.sha256(catalog_path.read_bytes()).hexdigest() != fit['input_sha256'][str(catalog_path.relative_to(ROOT))]:
        raise ValueError('frozen catalog identity changed')
    catalog = torch.load(catalog_path, map_location='cpu', weights_only=False)
    pairs = {p['pair_id']: p for p in catalog['train_pairs']}
    cores = []
    for row in fit['generation']['cores']:
        identity = row['identity']
        item = identity['item']
        entry = pairs[item['pair_id']]['entries'][item['side']]
        mode = entry['row']['mode']
        truth = [RhythmEvent.from_event(Event.from_json(e), mode) for e in entry['truth_events']]
        truth_counter = Counter(map(key, truth))
        by_time = defaultdict(list)
        for e in truth:
            by_time[e.onset_us].append(e)
        rounds = []
        for generation in row['rounds']:
            pred = [RhythmEvent(**e) for e in generation['events']]
            counts = Counter(map(key, pred))
            extras = list((counts - truth_counter).elements())
            missing = list((truth_counter - counts).elements())
            excess_exact_duplicates = sum(max(0, n - max(1, truth_counter[k])) for k, n in counts.items())
            extra_details = []
            for value in extras:
                event = RhythmEvent(*value)
                nearest = min(truth, key=lambda t: abs(t.onset_us - event.onset_us)) if truth else None
                extra_details.append(dict(event=event_from_key(value),
                                          nearest_truth=asdict(nearest) if nearest else None,
                                          head_delta_ms=(event.onset_us - nearest.onset_us) / 1000 if nearest else None,
                                          at_true_onset=event.onset_us in by_time,
                                          exact_existing_event=value in truth_counter))
            rounds.append(dict(round=generation['round'], prediction_count=len(pred), target_count=len(truth),
                               exact_retained=sum((counts & truth_counter).values()),
                               exact_missing=len(missing), excess_count=len(extras),
                               excess_exact_duplicate_multiplicity=excess_exact_duplicates,
                               extras_at_true_onset=sum(x['at_true_onset'] for x in extra_details),
                               extras_off_true_onsets=sum(not x['at_true_onset'] for x in extra_details),
                               full_note_f1=generation['full_note_f1'],
                               exact_onset_group_recall=generation['exact_onset_group_recall'],
                               mania_lane_conflicts=generation['mania_lane_conflicts'],
                               extra_events=extra_details,
                               missing_events=[event_from_key(e) for e in missing]))
        slots = 20 * 32
        positives = len(truth)
        negative_count = slots - positives
        cores.append(dict(identity=identity, rounds=rounds,
                          occupancy_loss=dict(positive_slots=positives, empty_slots=negative_count,
                              positive_per_slot_weight=.5 / positives,
                              empty_per_slot_weight=.5 / negative_count,
                              per_positive_to_per_empty_weight_ratio=negative_count / positives,
                              empty_slot_accuracy=next(c for c in fit['final']['cores']
                                  if c['identity']['chart_id'] == identity['chart_id'])['accuracy']['kind_4'])))
    rounds_summary = {}
    for number in (1, 4):
        rows = [c['rounds'][number - 1] for c in cores]
        rounds_summary[str(number)] = dict(targets=sum(r['target_count'] for r in rows),
            predicted=sum(r['prediction_count'] for r in rows),
            exact_retained=sum(r['exact_retained'] for r in rows),
            exact_missing=sum(r['exact_missing'] for r in rows), excess=sum(r['excess_count'] for r in rows),
            excess_exact_duplicate_multiplicity=sum(r['excess_exact_duplicate_multiplicity'] for r in rows),
            extras_at_true_onset=sum(r['extras_at_true_onset'] for r in rows),
            extras_off_true_onsets=sum(r['extras_off_true_onsets'] for r in rows),
            macro_f1=sum(r['full_note_f1'] for r in rows) / len(rows),
            macro_exact_group_recall=sum(r['exact_onset_group_recall'] for r in rows) / len(rows),
            mania_conflicts=sum(r['mania_lane_conflicts'] for r in rows))
    report = dict(schema='beatfcn.p2b_probe_review.v1', complete=True, decision='reject_this_configuration',
        source_sha256=hashlib.sha256(fit_path.read_bytes()).hexdigest(),
        optimizer_updates=fit['optimizer_updates'], model_wall_s=fit['wall_s'], peak_rss_gib=fit['peak_rss_gib'],
        initial_loss=fit['initial']['mean_loss'], final_loss=fit['final']['mean_loss'],
        loss_reduction=1 - fit['final']['mean_loss'] / fit['initial']['mean_loss'], gates=fit['gates'],
        rounds=rounds_summary, cores=cores, new_inference=False, new_paid_compute_usd=0,
        findings=[
            'Four real training cores, not held-out quality or a fair training comparison against warm-start P1.',
            'Target-slot head/lane/hold classifications reached 100%; kind includes empty false positives.',
            'Retained target events and extra event multiplicities are counted separately; legitimate Phira multiplicities are not filtered.',
            '1-to-4 rounds reduce some excess but do not pass the registered group/legality gates.',
            'Per-slot positive-vs-empty weight ratio is 23.6 to 79 in these cores. This makes occupancy calibration a specific objective-level hypothesis, not a confirmed causal explanation.',
            'No learned weights saved, no condition ablation after necessary failures, no new formal quality result.'
        ],
        next_route='Objective/representation audit: quantify sparse-slot occupancy and proper calibration, without retuning the backbone or simply extending training.',
        false_conclusions_to_avoid=['Attention is fundamentally broken.', 'Whole-canvas revision has been proven useful.',
                                    'All extra Phira duplicate rhythms are illegal.', 'Fresh P2B and warm P1 costs/accuracy are a controlled comparison.'])
    (ROOT / 'reports/p2b_probe_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['# R10 P2-B：真实小批探针复盘', '',
             '**当前配置否决，不进入长训。** 100次本地更新完成；没有新增付费、没有保存权重。', '',
             f'全MASK监督loss {report["initial_loss"]:.5f}→{report["final_loss"]:.5f}，降低{report["loss_reduction"]:.2%}，但未达到预注册的<=.1。有效头/类型准确率门槛通过，完整生成与合法性门槛失败。', '',
             '|训练片段|目标数|1轮生成数|4轮生成数|4轮完整音符F1|4轮精确起音组召回|',
             '|---|---:|---:|---:|---:|---:|']
    for c in cores:
        a, b = c['rounds'][0], c['rounds'][-1]
        lines.append(f'|{c["identity"]["chart_id"]}|{b["target_count"]}|{a["prediction_count"]}|{b["prediction_count"]}|{b["full_note_f1"]:.3f}|{b["exact_onset_group_recall"]:.3f}|')
    a, b = rounds_summary['1'], rounds_summary['4']
    lines += ['', f'第一轮精确保留{a["exact_retained"]}/{a["targets"]}个目标事件，同时多出{a["excess"]}个；第四轮精确保留{b["exact_retained"]}个，仍多出{b["excess"]}个。第四轮有{b["extras_at_true_onset"]}个多余音符落在真实起音，{b["extras_off_true_onsets"]}个不在真实起音。这里保留合法Phira事件重数，不把所有重复节奏都判脏。', '',
              '主要坏例是多余槽位激活，不能笼统说成“没有学会时间”或“和弦都被拆散”。当前同刻组指标因多余音符而受损；mania仍有键道冲突。四轮修订作用有限，不能仅凭网络允许回改就声称已学会取舍。', '',
              '损失给全部非空槽和全部空槽各一半总权重。四个片段中，一个非空槽的权重是一个空槽的23.6–79倍；空槽分类即使达到97%–99%，640个槽仍会产生多余音符。这给出下一轮目标层的具体假设，尚不能证明权重是唯一原因。不能通过隐藏真值数、事后提高阈值、去重或加长训练把这轮改判通过。', '',
              '下一步先审计占用决策的监督与解码是否校准、在真实各密度层有何变化，再冻结一个目标层对照。暂不增加主干深度/宽度，不购买长训。3500/P1保持。', '',
              '局限：只有四个训练片段、两种单字段条件；不能推断泛化或正式13字段控制。P1的旧拟合从3500初始化，P2从零且监督任务不同，二者耗时与结果不可作为公平架构胜负。']
    (ROOT / 'reports/p2b_probe_review.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: report[k] for k in ('decision', 'gates', 'rounds', 'initial_loss', 'final_loss')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
