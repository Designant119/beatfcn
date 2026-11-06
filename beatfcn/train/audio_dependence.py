"""Teacher-forced audio-dependence screen: does the model's likelihood use the audio?

No generation. For ~200 frozen held-out windows (all val charts, both modes) the
human target is teacher-forced under four acoustic conditions:

- real:            the window's own audio;
- foreign:         another held-out song's window of the same mode and core length;
- zero_local_mel:  local MERT/Pupu taps and 10 ms mel set to zero (song summary kept);
- roll:            local taps and mel rolled by a fixed per-window +-1..3 s shift.

The headline is the per-field cross-entropy gap (condition minus real). A model
that listens has clearly positive foreign/zero/roll gaps on the time fields
(head_block/head_cell/head_ms) and on mania/kind; gaps near zero mean the
decoder predicts the chart from its own history and the request alone.

Forced tokens (a single legal choice) are excluded from field means and counted.
"""
from collections import Counter
from contextlib import nullcontext
import hashlib
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from beatfcn.model.full_event_codec import FIELDS, encode_events, teacher_sequence
from beatfcn.train.density_curve import grid_density_curve
from beatfcn.train.local_panel import collate
from beatfcn.train.local_request_panel import event_request, request_inputs

CONDITIONS = ('real', 'foreign', 'zero_local_mel', 'roll')
COUNTERFACTUALS = CONDITIONS[1:]
FIELD_GROUPS = dict(group=('group',), head_block=('head_block',), head_cell=('head_cell',),
    head_ms=('head_ms',), mania=('mania',), member=('member',), kind=('kind',),
    tails=('tail_block', 'tail_cell', 'tail_ms'),
    head_time=('head_block', 'head_cell', 'head_ms'))
ROLL_CHOICES_S = (-3., -2., -1., 1., 2., 3.)
MEANING = ('Teacher-forced CE of the human chart with real vs foreign/zeroed/rolled audio. '
           'Positive gap (condition minus real) = the model uses the audio for that field.')


def _rank(seed, *parts):
    return hashlib.sha256(':'.join(map(str, (seed,)+parts)).encode()).hexdigest()


def _identity(panel, index):
    row = panel.records[index]
    chart = panel.charts[row['chart_index']]
    return dict(index=index, chart_id=row['chart_id'], audio_sha256=row['audio_sha256'],
        component=row['component'], mode=row['mode'], split=row['split'], request=row['request'],
        target_sha256=chart.get('target_sha256'), event_count=row.get('event_count'))


def val_candidates(panel, *, allow_carry=False):
    """Held-out request rows usable without hidden context.

    Rows with an incoming hold carry are excluded by default: the counterfactual
    training loader drops carries, so the teacher target would differ from training.
    """
    return [i for i, row in enumerate(panel.records)
            if row['split'] == 'val' and (allow_carry or not row['request'].get('known_carry'))]


def find_donor(panel, index, candidates, seed):
    """Deterministic foreign window: same mode/core length, other song and component."""
    row = panel.records[index]
    span = row['request']['end_us']-row['request']['start_us']
    options = [j for j in candidates if panel.records[j]['mode'] == row['mode']
               and panel.records[j]['audio_sha256'] != row['audio_sha256']
               and panel.records[j]['component'] != row['component']
               and panel.records[j]['request']['end_us']-panel.records[j]['request']['start_us'] == span]
    return min(options, key=lambda j: _rank(seed, 'donor', index, j)) if options else None


def freeze_teacher_panel(panel, *, windows=200, seed=20261101, tape_sha256=None,
                         roll_choices_s=ROLL_CHOICES_S, cell_us=100_000, allow_carry=False):
    """Round-robin over every val chart (hash order), alternating modes."""
    candidates = val_candidates(panel, allow_carry=allow_carry)
    by_chart = {}
    for index in candidates:
        by_chart.setdefault(panel.records[index]['chart_index'], []).append(index)
    queues = {}
    for chart, indices in sorted(by_chart.items()):
        mode = panel.records[indices[0]]['mode']
        queues.setdefault(mode, []).append(sorted(indices, key=lambda i: _rank(seed, 'window', i)))
    for mode in queues:
        queues[mode].sort(key=lambda q: _rank(seed, 'chart', panel.records[q[0]]['chart_index']))
    chosen, skipped, depth = [], Counter(), 0
    # Interleave modes chart by chart: iterating every chart of one mode before
    # the next filled 400 windows with mania only (652 mania vs 384 Phira charts).
    from itertools import zip_longest
    interleaved = [q for group in zip_longest(*(queues[m] for m in sorted(queues))) for q in group if q is not None]
    while len(chosen) < windows and any(depth < len(q) for qs in queues.values() for q in qs):
        for mode in ('*',):
            for queue in interleaved:
                if depth < len(queue) and len(chosen) < windows:
                    index = queue[depth]
                    donor = find_donor(panel, index, candidates, seed)
                    if donor is None:
                        skipped['no_matched_foreign_window'] += 1
                        continue
                    roll = roll_choices_s[int(_rank(seed, 'roll', index), 16) % len(roll_choices_s)]
                    chosen.append(dict(_identity(panel, index), foreign_index=donor,
                        foreign_audio_sha256=panel.records[donor]['audio_sha256'], roll_s=roll))
        depth += 1
    return dict(schema='audio-dependence-panel-v1', tape_sha256=tape_sha256, seed=seed, windows=chosen,
        conditions=list(CONDITIONS), cell_us=cell_us, controls='unconditional: encode_controls({}) + unspecified 100 ms curve (= fully dropped training sample)',
        val_charts=len(by_chart), skipped=dict(skipped),
        semantics=dict(real='own audio', foreign='other val song, same mode and core length',
            zero_local_mel='local taps + 10ms mel zeroed after normalization; song summary kept',
            roll='local taps + mel rolled by roll_s (wrap within the window context); clocks and song kept'),
        meaning=MEANING)


def validate_windows(panel, windows):
    for window in windows:
        row = panel.records[window['index']]
        for key in ('chart_id', 'audio_sha256', 'component', 'mode', 'split', 'request'):
            if row[key] != window[key]:
                raise ValueError('Frozen audio-dependence window changed: '+key)
        if row['split'] != 'val':
            raise ValueError('Audio-dependence screen must use held-out windows')
        target = panel.charts[row['chart_index']].get('target_sha256')
        if window.get('target_sha256') is not None and target != window['target_sha256']:
            raise ValueError('Frozen target identity changed')
        if 'foreign_index' in window:
            donor = panel.records[window['foreign_index']]
            if donor['audio_sha256'] != window['foreign_audio_sha256'] or donor['mode'] != row['mode']:
                raise ValueError('Foreign donor changed')


def intervene(inputs, condition, *, foreign=None, roll_s=None):
    """Return a modified copy; the request clock and controls are never touched."""
    if condition == 'real':
        return inputs
    result = {key: np.array(value, copy=True) for key, value in inputs.items()}
    streams = [s for s in ('local', 'mel', 'song') if s in result]
    if condition == 'foreign':
        if foreign is None:
            raise ValueError('Foreign features missing')
        for stream in streams:
            for key in (stream, stream+'_times', stream+'_valid'):
                result[key] = np.array(foreign[key], copy=True)
    elif condition in ('zero_local_mel', 'audio_zero'):
        for stream in (('local', 'mel') if condition == 'zero_local_mel' else streams):
            if stream in result:
                result[stream].fill(0)
    elif condition in ('roll', 'local_roll'):
        if roll_s is None:
            raise ValueError('Roll shift missing')
        for stream, axis in (('local', 1), ('mel', 0)):
            if stream not in result:
                continue
            valid = np.flatnonzero(result[stream+'_valid'])
            if len(valid) > 1:
                times = result[stream+'_times'][valid]
                shift = int(round(roll_s/float(np.median(np.diff(times)))))
                data = np.take(result[stream], valid, axis=axis)
                index = [slice(None)]*data.ndim
                index[axis] = valid
                result[stream][tuple(index)] = np.roll(data, shift, axis=axis)
    else:
        raise ValueError('Unknown audio condition: '+condition)
    return result


def base_inputs(panel, index, *, cell_us=100_000, curve=None, charter_timing=None):
    """Same condition inputs as training: unconditional controls (applicability +
    scope, no values -- what a fully dropped training sample sees) and the
    density curve on the training render grid (``cell_us``, 100 ms).

    ``charter_timing`` (optional beatfcn.train.charter_timing.ChartTimingTable)
    adds ``beat`` from the window's OWN chart timing (absent state for charts
    without valid timing); None keeps the historical inputs exactly."""
    from beatfcn.train.counterfactual_exposure import condition_inputs
    row = panel.records[index]
    request = event_request(row['request'])
    conditions = condition_inputs(request.mode, request, segments=curve or [], render_cell_us=cell_us)
    inputs, _ = request_inputs(panel.features, row['audio_sha256'], request, conditions.pop('controls'))
    inputs.update(conditions)
    if charter_timing is not None:
        from beatfcn.train.charter_timing import attach_window_beat
        attach_window_beat(inputs, charter_timing, row['chart_id'], request.start_us)
    return inputs, request


def foreign_features(panel, donor_index):
    donor = panel.records[donor_index]
    request = event_request(donor['request'])
    return panel.features.window(donor['audio_sha256'], request.start_us, request.end_us)


def teacher_target(panel, index, request, vocab):
    targets = panel.targets(index)
    return targets, teacher_sequence(encode_events(targets, request, vocab), request, vocab)


def _logits(output):
    return output['logits'] if isinstance(output, dict) else output


def token_losses(model, inputs, sequence, *, aux=False):
    """Per-token CE under the legal mask (same rule as event_loss), forced flags, aux.

    aux=True asks the model for its onset-head outputs (``return_aux``) in the
    same forward; models without it return aux=None.
    """
    import inspect
    extra = None
    if aux and 'return_aux' in inspect.signature(model.forward).parameters:
        output, extra = model(inputs, sequence, return_aux=True)
    else:
        output = model(inputs, sequence)
    logits = _logits(output).float()
    allowed, targets = sequence['allowed'], sequence['targets']
    selected = logits.masked_fill(~allowed, -1e9)
    losses = F.cross_entropy(selected.flatten(0, 1), targets.flatten(), reduction='none').view_as(targets)
    return losses, allowed.sum(-1) <= 1, extra


def _peaks(probabilities, times, valid, span_s, threshold=.5):
    p = probabilities
    keep = []
    for i in range(len(p)):
        if not valid[i] or not 0 <= times[i] < span_s or p[i] < threshold:
            continue
        if (i == 0 or p[i] >= p[i-1]) and (i+1 == len(p) or p[i] > p[i+1]):
            keep.append(int(round(times[i]*1e6)))
    return keep


def onset_head_scores(aux, truths, spans_s):
    """Peak-picked onset-head F1 at 50 ms (threshold 0.5) when the model has an onset head.

    aux: {'onset_logits' [B,T] or [B,T,1+types], 'onset_times', 'onset_valid'}.
    Channel 0 is "any onset"; channels 1+k score TYPES[k] onsets when present.
    truths: per row [(rel_onset_us, type)].
    """
    from beatfcn.events.canonical import TYPES
    from beatfcn.model.sample import match_f1
    if not aux or aux.get('onset_logits') is None:
        return None
    logits = aux['onset_logits'].float()
    if logits.ndim == 2:
        logits = logits[..., None]
    probabilities = torch.sigmoid(logits).cpu().numpy()
    times = aux['onset_times'].float().cpu().numpy()
    valid = aux['onset_valid'].bool().cpu().numpy()
    out = []
    for row, (truth, span) in enumerate(zip(truths, spans_s)):
        entry = {}
        for channel in range(probabilities.shape[-1]):
            kind = 'any' if channel == 0 else TYPES[channel-1]
            target = sorted({t for t, k in truth if kind == 'any' or k == kind})
            predicted = _peaks(probabilities[row, :, channel], times[row], valid[row], span)
            tp, n_pred, n_true, _ = match_f1(predicted, target, 50_000)
            entry[kind] = dict(tp=tp, pred=n_pred, true=n_true)
        out.append(entry)
    return out


def _accumulate(losses, forced, sequence, row):
    valid = sequence['valid'][row]
    fields = sequence['fields'][row][valid].tolist()
    values = losses[row][valid].tolist()
    forced_row = forced[row][valid].tolist()
    totals = {}
    for field, value, is_forced in zip(fields, values, forced_row):
        name = FIELDS[field]
        entry = totals.setdefault(name, [0., 0, 0])
        if is_forced:
            entry[2] += 1
        else:
            entry[0] += value
            entry[1] += 1
    joint = float(np.mean(values)) if values else None
    return dict(joint_ce=joint, fields={k: dict(sum=v[0], count=v[1], forced=v[2]) for k, v in totals.items()})


def run_teacher_screen(model, panel, protocol, *, device, batch_size=8, deadline=None, autocast=True,
                       charter_timing=None):
    """Return complete per-window raw CE records plus the gap summary.

    Caller owns eval/train flags and RNG; this function only runs no-grad forwards.
    ``charter_timing`` (optional ChartTimingTable): every window gets its own
    chart's beat features (labelled ``charter_timing_input`` in the summary).
    """
    validate_windows(panel, protocol['windows'])
    device = torch.device(device)
    vocab = getattr(model, 'vocab', None)
    amp = (torch.autocast('cuda', dtype=torch.bfloat16)
           if autocast and device.type == 'cuda' else nullcontext())
    onset_capable = bool(getattr(model.c, 'onset_head', False))
    records, started = [], time.monotonic()
    windows = protocol['windows']
    for first in range(0, len(windows), batch_size):
        if deadline is not None and time.monotonic() >= deadline:
            break
        group = windows[first:first+batch_size]
        prepared = []
        for window in group:
            inputs, request = base_inputs(panel, window['index'], cell_us=protocol.get('cell_us', 100_000),
                                          charter_timing=charter_timing)
            targets, sequence = teacher_target(panel, window['index'], request, vocab)
            prepared.append(dict(window=window, inputs=inputs, request=request, sequence=sequence,
                truth_rel_us=[e.onset_us-request.start_us for e in targets],
                truth_typed=[(e.onset_us-request.start_us, e.type) for e in targets],
                span_s=(request.end_us-request.start_us)/1e6,
                foreign=(foreign_features(panel, window['foreign_index'])
                         if 'foreign_index' in window else None)))
        results = [dict(index=p['window']['index'], chart_id=p['window']['chart_id'], mode=p['window']['mode'],
                        event_count=len(p['truth_rel_us']), tokens=int(len(p['sequence']['targets'])),
                        roll_s=p['window'].get('roll_s'), foreign_index=p['window'].get('foreign_index'),
                        conditions={}) for p in prepared]
        for condition in protocol.get('conditions', CONDITIONS):
            if deadline is not None and time.monotonic() >= deadline:
                break
            rows = [dict(inputs=intervene(p['inputs'], condition, foreign=p['foreign'], roll_s=p['window'].get('roll_s')),
                         sequence=p['sequence']) for p in prepared]
            inp, seq = collate(rows)
            inp = {k: v.to(device) for k, v in inp.items()}
            seq = {k: v.to(device) for k, v in seq.items()}
            with torch.no_grad(), amp:
                losses, forced, aux = token_losses(model, inp, seq, aux=onset_capable)
                onset = (onset_head_scores(aux, [p['truth_typed'] for p in prepared],
                                           [p['span_s'] for p in prepared]) if onset_capable else None)
            losses, forced = losses.float().cpu(), forced.cpu()
            seq_cpu = {k: v.cpu() for k, v in seq.items()}
            for row, result in enumerate(results):
                entry = _accumulate(losses, forced, seq_cpu, row)
                if onset is not None:
                    entry['onset_head_50ms'] = onset[row]
                result['conditions'][condition] = entry
            del inp, seq, losses, forced
        if charter_timing is not None:
            for result in results:
                result['charter_timing_present'] = charter_timing.segments(result['chart_id']) is not None
        records.extend(results)
    summary = summarize_teacher(records, protocol)
    summary['elapsed_s'] = time.monotonic()-started
    if charter_timing is not None:
        summary['charter_timing_input'] = dict(charter_timing.identity(),
            windows_with_timing=sum(r['charter_timing_present'] for r in records), windows=len(records))
    return dict(records=records, summary=summary)


def _field_values(entry, group):
    total = sum(entry['fields'].get(f, {}).get('sum', 0.) for f in FIELD_GROUPS[group])
    count = sum(entry['fields'].get(f, {}).get('count', 0) for f in FIELD_GROUPS[group])
    return total, count


def summarize_teacher(records, protocol):
    expected = len(protocol['windows'])
    complete = [r for r in records if all(c in r['conditions'] for c in protocol.get('conditions', CONDITIONS))]
    out = dict(meaning=MEANING, expected_windows=expected, measured_windows=len(complete),
        complete=len(complete) == expected,
        windows_by_mode=dict(Counter(r['mode'] for r in complete)),
        positive_windows=sum(r['event_count'] > 0 for r in complete),
        field_ce={}, gaps={}, by_mode={})

    def block(rows):
        field_ce, gaps = {}, {}
        for condition in protocol.get('conditions', CONDITIONS):
            field_ce[condition] = {}
            for group in FIELD_GROUPS:
                total = count = 0
                for r in rows:
                    t, c = _field_values(r['conditions'][condition], group)
                    total += t
                    count += c
                field_ce[condition][group] = dict(mean=total/count if count else None, tokens=count)
            joint = [r['conditions'][condition]['joint_ce'] for r in rows
                     if r['conditions'][condition]['joint_ce'] is not None]
            field_ce[condition]['joint'] = dict(mean=float(np.mean(joint)) if joint else None, tokens=len(joint))
        for condition in COUNTERFACTUALS:
            if condition not in field_ce:
                continue
            gaps[condition] = {}
            for group in list(FIELD_GROUPS)+['joint']:
                paired = []
                for r in rows:
                    if group == 'joint':
                        a, b = r['conditions']['real']['joint_ce'], r['conditions'][condition]['joint_ce']
                        if a is not None and b is not None:
                            paired.append(b-a)
                        continue
                    ta, ca = _field_values(r['conditions']['real'], group)
                    tb, cb = _field_values(r['conditions'][condition], group)
                    if ca and cb:
                        paired.append(tb/cb-ta/ca)
                real, other = field_ce['real'][group]['mean'], field_ce[condition][group]['mean']
                values = np.asarray(paired)
                gaps[condition][group] = dict(
                    pooled_gap_nats=(other-real) if real is not None and other is not None else None,
                    paired_mean_nats=float(values.mean()) if len(values) else None,
                    paired_sd_nats=float(values.std(ddof=1)) if len(values) > 1 else None,
                    paired_se_nats=float(values.std(ddof=1)/math.sqrt(len(values))) if len(values) > 1 else None,
                    windows=int(len(values)),
                    fraction_windows_positive=float((values > 0).mean()) if len(values) else None)
        onset = {}
        for condition in protocol.get('conditions', CONDITIONS):
            counts = [r['conditions'][condition].get('onset_head_50ms') for r in rows]
            counts = [c for c in counts if c is not None]
            if counts:
                per_kind = {}
                for kind in counts[0]:
                    tp, p, t = (sum(c[kind][k] for c in counts) for k in ('tp', 'pred', 'true'))
                    per_kind[kind] = dict(tp=tp, pred=p, true=t, f1=2*tp/(p+t) if p+t else None)
                onset[condition] = dict(per_kind['any'], per_type=per_kind)
        return field_ce, gaps, onset or None

    if complete:
        out['field_ce'], out['gaps'], out['onset_head_f1_50ms'] = block(complete)
        for mode in sorted({r['mode'] for r in complete}):
            ce, gaps, onset = block([r for r in complete if r['mode'] == mode])
            out['by_mode'][mode] = dict(field_ce=ce, gaps=gaps, onset_head_f1_50ms=onset)
        out['headline'] = {cond: {group: out['gaps'].get(cond, {}).get(group, {}).get('pooled_gap_nats')
                                  for group in ('head_time', 'head_block', 'head_cell', 'head_ms', 'mania',
                                                'kind', 'tails', 'group', 'joint')}
                           for cond in COUNTERFACTUALS if cond in out['gaps']}
    return out
