"""Difficulty-spread probe: does the model use ANOTHER human chart of this song?

Teacher-forced (cheap). Frozen held-out VAL windows whose chart has a denser
sibling chart of the same song (same mode, different canonical chart): the
realistic "the charter made the harder difficulty, derive this one" case. The
human window is teacher-forced under:
  ``none``            blank reference (historical contract)
  ``sibling``         the next-denser sibling's events in [start-2 s, end+2 s) as role
                      human_alternative (alternative_context training contract)
  ``sibling_shift1s`` the same sibling retimed by +1 s (same notes, wrong alignment)
Gaps (nats, paired over windows; > 0 = the treatment helps):
  sibling_vs_none      CE(none) - CE(sibling)            uses the other chart at all
  alignment_specific   CE(sibling_shift1s) - CE(sibling) uses it at the right TIME,
                                                          not just its density/style
Conditions use unconditional controls / unspecified curve, as the other teacher panels.
"""
from collections import Counter
import time

import torch

from beatfcn.model.full_event_codec import encode_events, teacher_sequence
from beatfcn.train.alternative_context import ROLE, SiblingIndex, reference_inputs_with_alternative
from beatfcn.train.audio_dependence import FIELD_GROUPS, _accumulate, _field_values, token_losses
from beatfcn.train.continuation_benchmark import HEADLINE_FIELDS, _paired, _rank
from beatfcn.train.counterfactual_exposure import condition_inputs
from beatfcn.train.history_context import chart_arrays, shift_arrays, visible_source
from beatfcn.train.local_panel import collate
from beatfcn.train.local_request_panel import event_request, request_inputs

SCHEMA = 'spread-panel-v1'
PAD_US = 2_000_000
SHIFT_US = 1_000_000
CONDITIONS = ('none', 'sibling', 'sibling_shift1s')
COMPARISONS = (('sibling_vs_none', 'none', 'sibling'),
               ('alignment_specific', 'sibling_shift1s', 'sibling'),
               ('shifted_vs_none', 'none', 'sibling_shift1s'))
MEANING = ('Teacher-forced CE of the human window with another human chart of the same song visible '
           '(human_alternative). gap = CE(control) - CE(treatment) in nats; > 0 means the treatment helps. '
           'alignment_specific > 0 means the model uses the other chart at the right time, not only its style.')


def freeze_spread_panel(panel, *, windows=200, seed=20261102, tape_sha256=None, split='val'):
    """Round-robin over charts (hash order) with a denser sibling; sibling = the next-denser chart."""
    index = SiblingIndex(panel, split=split)
    by_chart = {}
    for i, row in enumerate(panel.records):
        if row['split'] != split or row['request'].get('known_carry'):
            continue
        by_chart.setdefault(row['chart_index'], []).append(i)
    queues, skipped = [], Counter()
    for chart_index, indices in sorted(by_chart.items()):
        siblings = index.siblings(chart_index)
        mine = index.heads(chart_index) if siblings else None
        denser = sorted((index.heads(j), j) for j in siblings if index.heads(j) > mine) if siblings else []
        if not denser:
            skipped['no_denser_sibling'] += 1
            continue
        sibling = denser[0][1]
        queues.append((chart_index, sibling, sorted(indices, key=lambda i: _rank(seed, 'window', i))))
    queues.sort(key=lambda q: _rank(seed, 'chart', q[0]))
    chosen, depth = [], 0
    while len(chosen) < windows and any(depth < len(q[2]) for q in queues):
        for chart_index, sibling, queue in queues:
            if depth >= len(queue) or len(chosen) >= windows:
                continue
            i = queue[depth]
            row = panel.records[i]
            chosen.append(dict(index=i, chart_id=row['chart_id'], audio_sha256=row['audio_sha256'], mode=row['mode'],
                               request=row['request'], sibling_index=sibling,
                               sibling_chart_id=panel.charts[sibling]['chart_id'],
                               target_heads=index.heads(chart_index), sibling_heads=index.heads(sibling)))
        depth += 1
    return dict(schema=SCHEMA, tape_sha256=tape_sha256, seed=seed, split=split, windows=chosen,
                conditions=list(CONDITIONS), comparisons=[list(c) for c in COMPARISONS], pad_us=PAD_US,
                shift_us=SHIFT_US, cell_us=100_000, charts_with_denser_sibling=len(queues), skipped=dict(skipped),
                windows_by_mode=dict(Counter(w['mode'] for w in chosen)), meaning=MEANING,
                controls='unconditional: encode_controls({}) + unspecified 100 ms curve')


def validate_spread_panel(panel, protocol):
    for w in protocol['windows']:
        row = panel.records[w['index']]
        if any(row[k] != w[k] for k in ('chart_id', 'audio_sha256', 'mode', 'request')):
            raise ValueError('Frozen spread window changed')
        if panel.charts[w['sibling_index']]['chart_id'] != w['sibling_chart_id']:
            raise ValueError('Frozen spread sibling changed')


def spread_sources(panel, window, condition, request, protocol):
    if condition == 'none':
        return ()
    arrays = chart_arrays(panel, window['sibling_index'])
    if condition == 'sibling_shift1s':
        arrays = shift_arrays(arrays, int(protocol.get('shift_us', SHIFT_US)))
        keep = (arrays['onset_us'] >= 0) & (arrays['end_us'] <= request.duration_us)
        arrays = {k: v[keep] for k, v in arrays.items()}
    elif condition != 'sibling':
        raise ValueError('Unknown spread condition: ' + condition)
    pad = int(protocol.get('pad_us', PAD_US))
    region = (max(0, request.start_us - pad), min(request.duration_us, request.end_us + pad))
    return (visible_source(arrays, mode=request.mode, audio_sha256=window['audio_sha256'], role=ROLE,
                           region=region, source_id=f"alternative:{window['sibling_chart_id']}"),)


def _prepare(panel, window, vocab, conditions, protocol):
    row = panel.records[window['index']]
    request = event_request(row['request'])
    cond = condition_inputs(request.mode, request, render_cell_us=protocol.get('cell_us', 100_000))
    base, _ = request_inputs(panel.features, row['audio_sha256'], request, cond.pop('controls'))
    base.update(cond)
    targets = panel.targets(window['index'])
    sequence = teacher_sequence(encode_events(targets, request, vocab), request, vocab)
    variants, info = {}, {}
    for condition in conditions:
        sources = spread_sources(panel, window, condition, request, protocol)
        inputs = dict(base)
        if sources:
            reference, _ = reference_inputs_with_alternative(request, row, (), sources)
            inputs.update(reference)
        variants[condition] = inputs
        info[condition] = dict(reference_rows=int(len(inputs['reference'])),
                               reference_events=int(sum(len(s.events) for s in sources)))
    return request, targets, sequence, variants, info


def _block(rows, conditions, comparisons):
    field_ce = {}
    for condition in conditions:
        field_ce[condition] = {}
        for group in FIELD_GROUPS:
            total = count = 0
            for r in rows:
                t, c = _field_values(r['conditions'][condition], group)
                total += t
                count += c
            field_ce[condition][group] = dict(mean=total/count if count else None, tokens=count)
        joint = [r['conditions'][condition]['joint_ce'] for r in rows if r['conditions'][condition]['joint_ce'] is not None]
        field_ce[condition]['joint'] = dict(mean=sum(joint)/len(joint) if joint else None, tokens=len(joint))
    gaps = {}
    for name, control, treatment in comparisons:
        gaps[name] = {}
        for group in list(FIELD_GROUPS) + ['joint']:
            paired = []
            for r in rows:
                a, b = r['conditions'][control], r['conditions'][treatment]
                if group == 'joint':
                    if a['joint_ce'] is not None and b['joint_ce'] is not None:
                        paired.append(a['joint_ce'] - b['joint_ce'])
                    continue
                ta, ca = _field_values(a, group)
                tb, cb = _field_values(b, group)
                if ca and cb:
                    paired.append(ta/ca - tb/cb)
            ctrl, treat = field_ce[control][group]['mean'], field_ce[treatment][group]['mean']
            gaps[name][group] = dict(pooled_gap_nats=(ctrl-treat) if ctrl is not None and treat is not None else None,
                                     **_paired(paired))
    return field_ce, gaps


def run_spread_teacher(model, panel, protocol, *, device, batch_size=8, deadline=None):
    validate_spread_panel(panel, protocol)
    device = torch.device(device)
    vocab = getattr(model, 'vocab', None)
    conditions = protocol.get('conditions', CONDITIONS)
    comparisons = [tuple(c) for c in protocol.get('comparisons', COMPARISONS)]
    records, started = [], time.monotonic()
    windows = protocol['windows']
    for first in range(0, len(windows), batch_size):
        if deadline is not None and time.monotonic() >= deadline:
            break
        group = windows[first:first+batch_size]
        prepared = [_prepare(panel, w, vocab, conditions, protocol) for w in group]
        results = [dict(index=w['index'], chart_id=w['chart_id'], sibling=w['sibling_chart_id'], mode=w['mode'],
                        event_count=len(p[1]), tokens=int(len(p[2]['targets'])), conditions={}, references=p[4])
                   for w, p in zip(group, prepared)]
        for condition in conditions:
            inp, seq = collate([dict(inputs=p[3][condition], sequence=p[2]) for p in prepared])
            inp = {k: v.to(device) for k, v in inp.items()}
            seq = {k: v.to(device) for k, v in seq.items()}
            with torch.no_grad():
                losses, forced, _ = token_losses(model, inp, seq)
            losses, forced = losses.float().cpu(), forced.cpu()
            seq_cpu = {k: v.cpu() for k, v in seq.items()}
            for row, result in enumerate(results):
                result['conditions'][condition] = _accumulate(losses, forced, seq_cpu, row)
        records.extend(results)
    complete = [r for r in records if all(c in r['conditions'] for c in conditions)]
    summary = dict(meaning=MEANING, expected_windows=len(windows), measured_windows=len(complete),
                   complete=len(complete) == len(windows), windows_by_mode=dict(Counter(r['mode'] for r in complete)),
                   elapsed_s=time.monotonic() - started, by_mode={})
    if complete:
        summary['field_ce'], summary['gaps'] = _block(complete, conditions, comparisons)
        for mode in sorted({r['mode'] for r in complete}):
            ce, gaps = _block([r for r in complete if r['mode'] == mode], conditions, comparisons)
            summary['by_mode'][mode] = dict(field_ce=ce, gaps=gaps, headline={
                name: {f: gaps[name][f]['paired_mean_nats'] for f in HEADLINE_FIELDS if f in gaps[name]} for name in gaps})
    return dict(records=records, summary=summary)
