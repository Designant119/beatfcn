"""D41: label-conditioned candidate coverage, never a free-generation decoder."""
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np

from beatfcn.model.event_canvas import ClockLayout, RhythmEvent
from beatfcn.model.group_canvas import encode_group_targets
from scripts.r09_local_data import file_sha, verify_files

ROOT = Path(__file__).resolve().parents[1]


def ranks(logits, targets):
    """Stable descending rank with the smaller class ID breaking exact ties."""
    order = np.argsort(-np.asarray(logits, dtype=np.float64), axis=-1, kind='stable')
    inverse = np.argsort(order, axis=-1, kind='stable')
    return np.take_along_axis(inverse, np.asarray(targets)[:, None], axis=-1)[:, 0] + 1


def ordered_chord_candidates(logits):
    """The training representation orders mania members by increasing lane."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 4 or not 1 <= len(logits) <= 4:
        raise ValueError('requires one through four mania members')
    choices = np.array(list(combinations(range(4), len(logits))), dtype=np.int64)
    scores = logits[np.arange(len(logits))[None], choices].sum(axis=-1)
    order = np.argsort(-scores, kind='stable')
    return choices[order], scores[order]


def tail_candidates(block, cell, ms, *, head_ms, target_ms):
    """Ranks over all representable legal tails, with a truth-given head."""
    block, cell, ms = [np.asarray(a, dtype=np.float64) for a in (block, cell, ms)]
    if cell.shape != (50,) or ms.shape != (40,) or not head_ms < target_ms < len(block) * 2000:
        raise ValueError('tail clock or target outside represented support')
    scores = (block[:, None, None] + cell[None, :, None] + ms[None, None, :]).reshape(-1)
    tails = np.arange(len(scores), dtype=np.int64)
    legal = tails > head_ms
    order = np.flatnonzero(legal)[np.argsort(-scores[legal], kind='stable')]
    exact_rank = int(np.flatnonzero(order == target_ms)[0]) + 1
    within = np.abs(order - target_ms) <= 20
    tolerance_rank = int(np.flatnonzero(within)[0]) + 1
    return dict(exact_rank=exact_rank, within20ms_rank=tolerance_rank,
                independent_argmax_ms=int(block.argmax() * 2000 + cell.argmax() * 40 + ms.argmax()),
                legal_argmax_ms=int(order[0]), legal_support=int(legal.sum()),
                top5_tail_ms=order[:5].tolist(),
                score_margin_to_exact=float(scores[order[0]] - scores[target_ms]))


def label_vectors(labels):
    n = labels.multiplicity.reshape(-1)
    notes = labels.notes.reshape(-1, 3)
    heads = labels.heads.reshape(-1, 2)
    vectors = dict(block_count=(labels.counts, np.ones_like(labels.counts, dtype=bool)),
                   member_count=(n - 1, n > 0), head=(heads[:, 0], n > 0),
                   kind=(notes[:, 0], notes[:, 0] < 4))
    if (heads[:, 1] >= 0).any():
        raise ValueError('D32 does not permit hidden COPY times')
    if labels.mode == 'mania_4k':
        vectors['lane'] = notes[:, 2], notes[:, 0] < 4
    hold = notes[:, 0] == 1
    if hold.any():
        vectors.update(tail_block=(notes[:, 1] // 2000, hold),
                       tail_cell=(notes[:, 1] % 2000 // 40, hold),
                       tail_ms=(notes[:, 1] % 40, hold))
    return vectors


def summarize(cores):
    lanes = [row for c in cores for row in c['lane_groups']]
    notes = [r for g in lanes for r in g['member_target_ranks']]
    wrong = sum(r != 1 for r in notes)
    singles_wrong = sum(g['independent_note_errors'] for g in lanes if g['size'] == 1)
    tails = [row for c in cores for row in c['hold_tails']]
    by_size = {}
    for size in sorted({g['size'] for g in lanes}):
        selected = [g for g in lanes if g['size'] == size]
        rs = [r for g in selected for r in g['member_target_ranks']]
        by_size[str(size)] = dict(groups=len(selected), notes=len(rs),
            independent_note_correct=sum(r == 1 for r in rs),
            ordered_note_correct=sum(size - g['ordered_note_errors'] for g in selected),
            independent_group_correct=sum(g['independent_note_errors'] == 0 for g in selected),
            ordered_group_correct=sum(g['ordered_note_errors'] == 0 for g in selected),
            duplicate_groups=sum(g['independent_has_duplicate'] for g in selected),
            target_joint_rank_histogram={str(r):sum(g['target_joint_rank'] == r for g in selected)
                for r in range(1, max(g['target_joint_rank'] for g in selected) + 1)},
            member_topk_correct={str(k):sum(r <= k for r in rs) for k in (1, 2, 3)})
    return dict(lane_notes=len(notes), lane_errors=wrong,
        single_note_errors=singles_wrong,
        single_share_of_errors=singles_wrong / wrong if wrong else None,
        lane_topk_correct={str(k):sum(r <= k for r in notes) for k in (1, 2, 3)},
        lane_top2_coverage=sum(r <= 2 for r in notes) / len(notes) if notes else None,
        oracle_ordered_note_correct=sum(g['size'] - g['ordered_note_errors'] for g in lanes),
        oracle_by_chord_size=by_size,
        hold_notes=len(tails),
        exact_tail_topk={str(k):sum(t['exact_rank'] <= k for t in tails) for k in (1, 2, 5, 10, 100)},
        tolerance_tail_topk={str(k):sum(t['within20ms_rank'] <= k for t in tails) for k in (1, 2, 5, 10, 100)},
        hold_kind_top1=sum(t['kind_rank'] == 1 for t in tails),
        hold_lane_top2=sum(t['lane_rank'] is not None and t['lane_rank'] <= 2 for t in tails))


def main():
    start = time.perf_counter()
    plan_path = ROOT / 'bench/rhythm-v2/d41-parallel-candidates-v1.json'
    plan = json.loads(plan_path.read_text())
    verify_files(plan)
    destination = ROOT / plan['result']
    if destination.exists():
        raise FileExistsError('preserve prior audit')
    raw = json.loads((ROOT / plan['report']).read_text())
    arrays = np.load(ROOT / plan['logits'], allow_pickle=False)
    if raw['logits']['sha256'] != file_sha(ROOT / plan['logits']):
        raise ValueError('saved logit identity mismatch')
    cases, stats, used_keys = {}, {}, set()
    for variant in ('original', 'translated', 'swapped'):
        cases[variant] = []
        for i, core in enumerate(raw['generation'][variant]['cores']):
            identity = core['identity']
            labels = encode_group_targets([RhythmEvent(**e) for e in core['target_events']],
                ClockLayout(identity['c0_us'], identity['c0_us'] + 8_000_000), identity['mode'])
            vectors = label_vectors(labels)
            prefix = f'{variant}__core{i}'
            outputs, maps = {}, {}
            for name, (targets, mask) in vectors.items():
                keys = {field:f'{prefix}__{name}__{field}' for field in ('logits', 'targets', 'indices')}
                used_keys.update(keys.values())
                logits, saved_targets, indices = [arrays[keys[field]] for field in ('logits', 'targets', 'indices')]
                if not np.array_equal(indices, np.flatnonzero(mask)) or not np.array_equal(saved_targets, targets[mask]):
                    raise ValueError('target/active-slot identity mismatch: ' + prefix + '/' + name)
                if logits.ndim != 2 or len(logits) != len(indices) or not np.isfinite(logits).all():
                    raise ValueError('invalid saved scores')
                outputs[name] = logits
                maps[name] = {int(flat):row for row, flat in enumerate(indices)}
                stats[keys['logits']] = dict(shape=list(logits.shape), nonfinite=0,
                    minimum=float(logits.min()), maximum=float(logits.max()),
                    mean=float(logits.mean()), std=float(logits.std()))
            row = dict(identity=identity, lane_groups=[], hold_tails=[])
            notes = labels.notes.reshape(-1, 3)
            head_bases = np.repeat([b for b, _, _ in labels.layout.blocks], labels.groups)
            head_ms = head_bases // 1000 + labels.heads.reshape(-1, 2)[:, 0]
            if identity['mode'] == 'mania_4k':
                for group in np.flatnonzero(labels.multiplicity.reshape(-1) > 0):
                    size = int(labels.multiplicity.reshape(-1)[group])
                    indices = list(range(int(group) * 8, int(group) * 8 + size))
                    logits = outputs['lane'][[maps['lane'][n] for n in indices]]
                    target = notes[indices, 2]
                    choices, scores = ordered_chord_candidates(logits)
                    joint_rank = int(np.flatnonzero(np.all(choices == target[None], axis=-1))[0]) + 1
                    pred = logits.argmax(-1)
                    row['lane_groups'].append(dict(flat_group=int(group), size=size,
                        source_head_us=int(head_ms[group] * 1000 + identity['source_clip_start_us']),
                        targets=target.tolist(), independent=pred.tolist(), ordered=choices[0].tolist(),
                        independent_note_errors=int((pred != target).sum()),
                        ordered_note_errors=int((choices[0] != target).sum()),
                        independent_has_duplicate=len(set(pred.tolist())) < size,
                        member_target_ranks=ranks(logits, target).tolist(),
                        target_joint_rank=joint_rank, legal_chord_support=len(choices),
                        top_joint_choices=choices.tolist(),
                        joint_score_margin=float(scores[0] - scores[joint_rank - 1])))
            for flat in np.flatnonzero(notes[:, 0] == 1):
                logits = [outputs[n][maps[n][int(flat)]] for n in ('tail_block', 'tail_cell', 'tail_ms')]
                tail = tail_candidates(*logits, head_ms=int(head_ms[flat // 8]), target_ms=int(notes[flat, 1]))
                kind = outputs['kind'][maps['kind'][int(flat)]][None]
                lane_rank = None
                if identity['mode'] == 'mania_4k':
                    lane = outputs['lane'][maps['lane'][int(flat)]][None]
                    lane_rank = int(ranks(lane, [notes[flat, 2]])[0])
                row['hold_tails'].append(dict(flat_note=int(flat), **tail,
                    source_head_us=int(head_ms[flat // 8] * 1000 + identity['source_clip_start_us']),
                    source_target_tail_us=int(notes[flat, 1] * 1000 + identity['source_clip_start_us']),
                    target_ms=int(notes[flat, 1]),
                    field_ranks={n:int(ranks(logits[j][None], [int(vectors[n][0][flat])])[0])
                        for j, n in enumerate(('tail_block', 'tail_cell', 'tail_ms'))},
                    kind_rank=int(ranks(kind, [1])[0]), lane_rank=lane_rank))
            cases[variant].append(row)
    if used_keys != set(arrays.files):
        raise ValueError('unaccounted saved arrays')
    summaries = {variant:{mode:summarize([c for c in cores if c['identity']['mode'] == mode])
                    for mode in ('mania_4k', 'phira')} for variant, cores in cases.items()}
    original = summaries['original']['mania_4k']
    result = dict(id='D41', complete=True, oracle_slot_conditioning=True,
        free_generation_effect_measured=False, model_promotion=False,
        limitations=plan['limitations'], plan_sha256=file_sha(plan_path),
        input_sha256=plan['input_sha256'], all_saved_active_labels_verified=True,
        saved_arrays_verified=len(used_keys), tensor_stats=stats, cases=cases, summaries=summaries,
        routing=dict(chord_only_cannot_address_majority=original['single_share_of_errors'] > .5,
                     beam2_original_lane_coverage_at_least95=original['lane_top2_coverage'] >= .95),
        optimizer_updates=0, model_forwards=0, new_paid_compute_usd=0,
        wall_s=time.perf_counter() - start)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(summaries=summaries, routing=result['routing'], wall_s=result['wall_s'])))


if __name__ == '__main__':
    main()
