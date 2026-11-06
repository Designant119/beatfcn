"""Frozen real-data representation check, with no optimizer or model inference."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.events.canonical import Event
from beatfcn.model.event_canvas import (
    EMPTY, MASK, KIND, CanvasOverflow, ClockLayout, RhythmEvent,
    active_holds, decode_targets, encode_targets, masked_view, new_canvas,
)


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start = time.perf_counter()
    plan_path = ROOT / 'bench/rhythm-v2/p2-event-canvas-roundtrip-v1.json'
    plan = json.loads(plan_path.read_text())
    catalog_path = ROOT / plan['catalog']
    if digest(catalog_path) != plan['catalog_sha256']:
        raise ValueError('training catalog identity changed')
    # Identity-verified artifact produced locally; validation targets are not inspected.
    catalog = torch.load(catalog_path, map_location='cpu', weights_only=False)
    entries = {}
    for pair in catalog['train_pairs']:
        for entry in pair['entries']:
            if entry['row']['split'] != 'train':
                raise ValueError('non-training row in representation probe')
            entries.setdefault((entry['row']['chart_id'], entry['example'].c0_us, entry['target_sha256']), entry)
    rows, failures, all_values = [], [], []
    for (chart_id, c0, target_sha), entry in entries.items():
        mode = entry['row']['mode']
        truth = [RhythmEvent.from_event(Event.from_json(e), mode) for e in entry['truth_events']]
        for phase in plan['phases_us']:
            layout = ClockLayout(c0, c0 + 8_000_000, plan['block_us'], phase)
            copies = tuple(map(int, entry['example'].copy_times + c0))
            identity = dict(chart_id=chart_id, audio_sha256=entry['row']['audio_sha256'],
                            c0_us=c0, phase_us=phase, target_sha256=target_sha)
            try:
                labels = encode_targets(truth, layout, mode, slots=plan['slots_per_block'], copy_times=copies)
                restored = decode_targets(labels)
                if Counter(restored) != Counter(truth):
                    raise AssertionError('rhythm event multiset changed')
                blank = new_canvas(layout, mode, slots=plan['slots_per_block'], copy_times=copies)
                hidden = masked_view(labels, np.zeros(labels.values.shape[:2], dtype=bool))
                np.testing.assert_array_equal(blank.values, hidden.values)
                if not np.all(hidden.values[..., KIND] == MASK):
                    raise AssertionError('target-derived empty positions leaked')
                carries = 0
                for _, boundary, _ in layout.blocks[1:]:
                    # The state can be built solely from heads already emitted.
                    past = [e for e in restored if e.onset_us < boundary]
                    carry = active_holds(past, boundary)
                    expected = [e for e in truth if e.type == 'hold' and e.onset_us < boundary < e.end_us]
                    if Counter(carry) != Counter(expected):
                        raise AssertionError('cross-block hold state changed')
                    carries += len(carry)
                count = (labels.values[..., KIND] < EMPTY).sum(-1)
                rows.append({**identity, 'mode':mode, 'shape':list(labels.values.shape),
                             'events':len(truth), 'max_notes_per_block':int(count.max()),
                             'empty_slots':int((labels.values[..., KIND] == EMPTY).sum()),
                             'empty_blocks':int((count == 0).sum()), 'carry_states':carries,
                             'exact_roundtrip':True, 'blank_no_target_leak':True})
                all_values.append(labels.values.reshape(-1, 5))
            except (ValueError, AssertionError) as exc:
                failures.append({**identity, 'error':f'{type(exc).__name__}: {exc}'})
    values = np.concatenate(all_values) if all_values else np.empty((0, 5), dtype=np.int64)
    health = {'dtype':str(values.dtype), 'rows':len(values), 'nonfinite':int((~np.isfinite(values)).sum()),
              'fields':['type_or_empty', 'head_offset_ms', 'copy_index', 'absolute_tail_ms', 'lane']}
    if len(values):
        health.update(min=values.min(0).tolist(), max=values.max(0).tolist(),
                      mean=values.mean(0).tolist(), std=values.std(0).tolist())
    # This canary must fail rather than losing the 33rd event.
    overflow_rejected = False
    try:
        encode_targets([RhythmEvent(i*1000, i*1000, 'drag') for i in range(33)],
                       ClockLayout(0, 400000), 'phira', slots=32)
    except CanvasOverflow:
        overflow_rejected = True
    passed = not failures and len(rows) == 2*len(entries) and overflow_rejected
    summary = {}
    for mode in ('mania_4k', 'phira'):
        selected = [r for r in rows if r['mode'] == mode]
        summary[mode] = {'roundtrips':len(selected), 'events_including_both_phases':sum(r['events'] for r in selected),
                         'maximum_notes_per_block':max((r['max_notes_per_block'] for r in selected), default=0),
                         'hold_carry_states':sum(r['carry_states'] for r in selected)}
    result = {'schema':'beatfcn.p2_event_canvas_probe.v1', 'complete':True, 'passed':passed,
              'plan_sha256':digest(plan_path), 'catalog_sha256':digest(catalog_path),
              'code_sha256':{str(p.relative_to(ROOT)):digest(p) for p in
                             (Path(__file__), ROOT/'beatfcn/model/event_canvas.py')},
              'train_windows':len(entries), 'train_audio':len({r['audio_sha256'] for r in rows}),
              'modes':summary, 'tensor_health':health, 'overflow_canary_rejected':overflow_rejected,
              'failures':failures, 'rows':rows, 'wall_s':time.perf_counter()-start,
              'model_updates':0, 'new_paid_compute_usd':0, 'model_promotion':False,
              'limits':['Prototype capacity only, not full-corpus admission.',
                        'No trained model, no audio inference, no quality/control improvement claimed.',
                        'Rhythm fields roundtrip; Phira native spatial targets are intentionally not encoded.',
                        'COPY_TIME unit tests do not prove the future generator preserves all protected objects.']}
    (ROOT/'reports/p2_event_canvas_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'reports/p2_event_canvas_probe.md').write_text(
        '# P2 事件画布表示探针\n\n'
        f'结果：{"通过" if passed else "失败"}。{len(entries)} 个真实训练谱窗、两个块相位，共 {len(rows)} 次完整节奏字段回环。\n\n'
        f'模式结果：`{json.dumps(summary,ensure_ascii=False)}`。\n\n'
        f'失败数 {len(failures)}；非有限值 {health["nonfinite"]}；33 音符/32 槽溢出明确拒绝：{overflow_rejected}。\n\n'
        '已验证时间、类型、尾点、mania 键道和事件重数；全 MASK 输入不含隐藏事件数，跨块 hold 状态由已生成头构建。\n\n'
        '这是表示正确性证据，不是模型能力。32 槽仅为原型容量，完整语料仍须重新审计；Phira 空间不作为目标。保护对象的完整生成链路尚未实现。无模型更新、无新增付费。\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows',)},ensure_ascii=False,indent=2))
    if not passed:
        raise SystemExit('event canvas representation gate failed')


if __name__ == '__main__':
    main()
