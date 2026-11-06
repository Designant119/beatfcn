"""Continuation / infill benchmark: does the model use THIS song's visible chart?

Two readouts, never mixed:

1. Teacher-forced (cheap, primary). Frozen held-out windows (both modes,
   start > 32 s, no incoming carry, a full 16 s suffix exists). The human
   window is teacher-forced under several visible-reference conditions:
   ``none`` (blank reference = the historical training contract), human prefix
   8/16/32 s (role human_input), prefix 16 s as model_history, infill = prefix
   16 s + suffix 16 s, and a FOREIGN prefix control (another held-out song's
   chart, same mode and length, retimed onto this clock). Readouts are per-field
   CE gaps with paired SE; ``song_specific`` = CE(foreign) - CE(real prefix)
   measures use of this song's history beyond generic density/style.

2. Free generation. Continuation: the human chart of [0, X) is visible
   (human_input), the model writes [X, end) window by window; each window sees
   the human prefix plus its own generated windows (model_history) up to 32 s
   back. Infill: human outside [a, b) visible, the model fills one 8-16 s gap;
   the window ending at b also sees a 16 s human suffix. Every job has an
   unconditioned twin (blank reference, same windows/carry/RNG seeds). Metrics
   vs the hidden human region: CER, onset/note F1 @50 ms, density ratio vs the
   visible context (and the human's own ratio), type-mix JS and IOI JS.

The rollout only receives visible human events; hidden human events are kept in
a separate truth map that is read only by the scorer.
"""
from collections import Counter
import hashlib
import math
import time

import numpy as np
import torch

from beatfcn.events.canonical import Event, LaneLoc, LineLoc, TYPES
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest, encode_events, teacher_sequence
from beatfcn.train.audio_dependence import FIELD_GROUPS, _accumulate, _field_values, token_losses
from beatfcn.train.counterfactual_exposure import condition_inputs
from beatfcn.train.event_conditions import encode_controls
from beatfcn.train.event_reference import VisibleReference
from beatfcn.train.history_context import (chart_arrays, check_no_leak, reference_inputs, shift_arrays,
                                           visible_source)
from beatfcn.train.local_panel import collate
from beatfcn.train.local_request_panel import event_request, request_inputs

SCHEMA = 'continuation-panel-v1'
MIN_START_US = 32_000_000
SUFFIX_US = 16_000_000
HISTORY_US = 32_000_000
CORE_US = 8_000_000
PREFIX_LENGTHS_S = (8, 16, 32)
INFILL_PREFIX_S = 16
CONDITIONS = ('none', 'prefix8', 'prefix16', 'prefix32', 'prefix16_model_history', 'infill16',
              'foreign8', 'foreign16', 'foreign32')
# (name, control condition, treatment condition): gap = CE(control) - CE(treatment); > 0 = treatment helps.
COMPARISONS = tuple(
    [(f'prefix{L}_vs_none', 'none', f'prefix{L}') for L in PREFIX_LENGTHS_S]
    + [(f'song_specific_prefix{L}', f'foreign{L}', f'prefix{L}') for L in PREFIX_LENGTHS_S]
    + [(f'foreign{L}_vs_none', 'none', f'foreign{L}') for L in PREFIX_LENGTHS_S]
    + [('infill16_vs_none', 'none', 'infill16'), ('infill16_vs_prefix16', 'prefix16', 'infill16'),
       ('model_history_role_vs_human_role', 'prefix16', 'prefix16_model_history')])
HEADLINE_FIELDS = ('head_time', 'kind', 'mania', 'tails', 'group', 'joint')
MEANING = ('Teacher-forced CE of the human window under visible-reference conditions. '
           'gap = CE(control) - CE(treatment) in nats; > 0 means the treatment reference helps. '
           'song_specific_prefixL = CE(foreign prefix) - CE(this song\'s prefix): > 0 means the model uses '
           'this song\'s history, not just generic density/style.')


def _rank(seed, *parts):
    return hashlib.sha256(':'.join(map(str, (seed,)+parts)).encode()).hexdigest()


# ------------------------------------------------------------------ panel
def continuation_candidates(panel, *, min_start_us=MIN_START_US, suffix_us=SUFFIX_US):
    out = []
    for i, row in enumerate(panel.records):
        q = row['request']
        if (row['split'] == 'val' and not q.get('known_carry') and q['start_us'] > min_start_us
                and q['end_us'] + suffix_us <= q['duration_us']):
            out.append(i)
    return out


def freeze_continuation_panel(panel, *, windows=200, seed=20261102, tape_sha256=None,
                              min_start_us=MIN_START_US, suffix_us=SUFFIX_US):
    """Round-robin over val charts (hash order), modes interleaved; foreign donor per window."""
    from itertools import zip_longest
    candidates = continuation_candidates(panel, min_start_us=min_start_us, suffix_us=suffix_us)
    by_chart = {}
    for index in candidates:
        by_chart.setdefault(panel.records[index]['chart_index'], []).append(index)
    queues = {}
    for chart, indices in sorted(by_chart.items()):
        queues.setdefault(panel.records[indices[0]]['mode'], []).append(
            sorted(indices, key=lambda i: _rank(seed, 'window', i)))
    for mode in queues:
        queues[mode].sort(key=lambda q: _rank(seed, 'chart', panel.records[q[0]]['chart_index']))
    interleaved = [q for group in zip_longest(*(queues[m] for m in sorted(queues))) for q in group if q is not None]
    chosen, skipped, depth = [], Counter(), 0
    while len(chosen) < windows and any(depth < len(q) for q in interleaved):
        for queue in interleaved:
            if depth >= len(queue) or len(chosen) >= windows:
                continue
            index = queue[depth]
            row = panel.records[index]
            donors = [j for j in candidates if panel.records[j]['mode'] == row['mode']
                      and panel.records[j]['audio_sha256'] != row['audio_sha256']
                      and panel.records[j]['component'] != row['component']]
            if not donors:
                skipped['no_foreign_donor'] += 1
                continue
            donor = min(donors, key=lambda j: _rank(seed, 'donor', index, j))
            chart = panel.charts[row['chart_index']]
            chosen.append(dict(index=index, chart_id=row['chart_id'], audio_sha256=row['audio_sha256'],
                component=row['component'], mode=row['mode'], request=row['request'],
                target_sha256=chart.get('target_sha256'), event_count=row.get('event_count'),
                foreign_index=donor, foreign_chart_id=panel.records[donor]['chart_id'],
                foreign_audio_sha256=panel.records[donor]['audio_sha256'],
                foreign_start_us=panel.records[donor]['request']['start_us']))
        depth += 1
    return dict(schema=SCHEMA, tape_sha256=tape_sha256, seed=seed, windows=chosen, conditions=list(CONDITIONS),
                comparisons=[list(c) for c in COMPARISONS], min_start_us=min_start_us, suffix_us=suffix_us,
                cell_us=100_000, val_charts=len(by_chart), skipped=dict(skipped),
                windows_by_mode=dict(Counter(w['mode'] for w in chosen)),
                controls='unconditional: encode_controls({}) + unspecified 100 ms curve',
                semantics=dict(none='blank reference (unknown window only)',
                    prefixL='this chart [start-L, start) as human_input',
                    prefix16_model_history='this chart [start-16 s, start) as model_history',
                    infill16='prefix 16 s + suffix [end, end+16 s) as human_input',
                    foreignL='another val chart (same mode, other song/component) [s\'-L, s\') retimed to '
                             '[start-L, start) as human_input; donor windows have no crossing hold'),
                meaning=MEANING)


def validate_panel(panel, protocol):
    for window in protocol['windows']:
        row = panel.records[window['index']]
        for key in ('chart_id', 'audio_sha256', 'mode', 'request'):
            if row[key] != window[key]:
                raise ValueError('Frozen continuation window changed: '+key)
        donor = panel.records[window['foreign_index']]
        if donor['chart_id'] != window['foreign_chart_id'] or donor['mode'] != row['mode']:
            raise ValueError('Frozen foreign donor changed')


# ------------------------------------------------------------- teacher forced
def condition_sources(panel, window, condition, request):
    """Visible references of one teacher condition (never the target window)."""
    if condition == 'none':
        return ()
    row = panel.records[window['index']]
    arrays = chart_arrays(panel, row['chart_index'])
    start, end = request.start_us, request.end_us
    audio, mode = row['audio_sha256'], request.mode

    def prefix(length_s, role='human_input', source=None, chart_id=row['chart_id']):
        region = (start - int(length_s*1e6), start)
        return visible_source(source if source is not None else arrays, mode=mode, audio_sha256=audio,
                              role=role, region=region, source_id=f'{role}-prefix:{chart_id}')
    if condition.startswith('prefix'):
        if condition == 'prefix16_model_history':
            return (prefix(16, role='model_history'),)
        return (prefix(int(condition[len('prefix'):])),)
    if condition == 'infill16':
        suffix = visible_source(arrays, mode=mode, audio_sha256=audio, role='human_input',
                                region=(end, end + SUFFIX_US), source_id=f'human_input-suffix:{row["chart_id"]}')
        return (prefix(INFILL_PREFIX_S), suffix)
    if condition.startswith('foreign'):
        length = int(condition[len('foreign'):])
        donor = panel.records[window['foreign_index']]
        donor_start = int(window['foreign_start_us'])
        shifted = shift_arrays(chart_arrays(panel, donor['chart_index']), start - donor_start)
        return (prefix(length, source=shifted, chart_id='foreign:'+donor['chart_id']),)
    raise ValueError('Unknown continuation condition: '+condition)


def _prepare(panel, window, vocab, conditions, cell_us):
    row = panel.records[window['index']]
    request = event_request(row['request'])
    cond = condition_inputs(request.mode, request, render_cell_us=cell_us)
    base, _ = request_inputs(panel.features, row['audio_sha256'], request, cond.pop('controls'))
    base.update(cond)
    targets = panel.targets(window['index'])
    sequence = teacher_sequence(encode_events(targets, request, vocab), request, vocab)
    variants, info = {}, {}
    for condition in conditions:
        sources = condition_sources(panel, window, condition, request)
        inputs = dict(base)
        if sources:
            reference, _ = reference_inputs(request, row['audio_sha256'], sources)
            inputs.update(reference)
        variants[condition] = inputs
        info[condition] = dict(reference_rows=int(len(inputs['reference'])),
                               reference_events=int(sum(len(s.events) for s in sources)))
    return request, targets, sequence, variants, info


def run_continuation_teacher(model, panel, protocol, *, device, batch_size=8, deadline=None):
    """Per-window per-condition CE records plus the comparison summary (no grad)."""
    validate_panel(panel, protocol)
    device = torch.device(device)
    vocab = getattr(model, 'vocab', None)
    conditions = protocol.get('conditions', CONDITIONS)
    records, started = [], time.monotonic()
    windows = protocol['windows']
    for first in range(0, len(windows), batch_size):
        if deadline is not None and time.monotonic() >= deadline:
            break
        group = windows[first:first+batch_size]
        prepared = [_prepare(panel, w, vocab, conditions, protocol.get('cell_us', 100_000)) for w in group]
        results = [dict(index=w['index'], chart_id=w['chart_id'], mode=w['mode'], event_count=len(p[1]),
                        tokens=int(len(p[2]['targets'])), conditions={}, references=p[4])
                   for w, p in zip(group, prepared)]
        for condition in conditions:
            rows = [dict(inputs=p[3][condition], sequence=p[2]) for p in prepared]
            inp, seq = collate(rows)
            inp = {k: v.to(device) for k, v in inp.items()}
            seq = {k: v.to(device) for k, v in seq.items()}
            with torch.no_grad():
                losses, forced, _ = token_losses(model, inp, seq)
            losses, forced = losses.float().cpu(), forced.cpu()
            seq_cpu = {k: v.cpu() for k, v in seq.items()}
            for row, result in enumerate(results):
                result['conditions'][condition] = _accumulate(losses, forced, seq_cpu, row)
        records.extend(results)
    summary = summarize_continuation_teacher(records, protocol)
    summary['elapsed_s'] = time.monotonic()-started
    return dict(records=records, summary=summary)


def _paired(values):
    values = np.asarray(values, float)
    return dict(paired_mean_nats=float(values.mean()) if len(values) else None,
                paired_sd_nats=float(values.std(ddof=1)) if len(values) > 1 else None,
                paired_se_nats=float(values.std(ddof=1)/math.sqrt(len(values))) if len(values) > 1 else None,
                windows=int(len(values)),
                fraction_windows_positive=float((values > 0).mean()) if len(values) else None)


def _block(rows, conditions):
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
        field_ce[condition]['joint'] = dict(mean=float(np.mean(joint)) if joint else None, tokens=len(joint))
    gaps = {}
    for name, control, treatment in COMPARISONS:
        if control not in conditions or treatment not in conditions:
            continue
        gaps[name] = {}
        for group in list(FIELD_GROUPS)+['joint']:
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


def summarize_continuation_teacher(records, protocol):
    conditions = protocol.get('conditions', CONDITIONS)
    complete = [r for r in records if all(c in r['conditions'] for c in conditions)]
    out = dict(meaning=MEANING, expected_windows=len(protocol['windows']), measured_windows=len(complete),
               complete=len(complete) == len(protocol['windows']),
               windows_by_mode=dict(Counter(r['mode'] for r in complete)),
               positive_windows=sum(r['event_count'] > 0 for r in complete), by_mode={})
    if not complete:
        return out
    out['field_ce'], out['gaps'] = _block(complete, conditions)
    for mode in sorted({r['mode'] for r in complete}):
        ce, gaps = _block([r for r in complete if r['mode'] == mode], conditions)
        out['by_mode'][mode] = dict(field_ce=ce, gaps=gaps,
            headline={name: {f: gaps[name][f]['paired_mean_nats'] for f in HEADLINE_FIELDS if f in gaps[name]}
                      for name in gaps})
    out['reference_rows_max'] = max(v['reference_rows'] for r in complete for v in r['references'].values())
    return out


# ------------------------------------------------------------ free generation
def _rhythm(arrays, mask):
    return [RhythmEvent(int(arrays['onset_us'][i]), int(arrays['end_us'][i]), TYPES[int(arrays['kind'][i])],
                        int(arrays['lane'][i])) for i in np.flatnonzero(mask)]


def generation_jobs(panel, *, prefix_seconds=(30, 60), seed=20261102, max_charts_per_mode=None,
                    infill_range_s=(8, 16), split='val'):
    """(jobs, truths). Jobs hold only VISIBLE human events; truths hold the hidden region."""
    charts = [(i, c) for i, c in enumerate(panel.charts) if c['split'] == split]
    by_mode = {}
    for i, chart in sorted(charts, key=lambda ic: _rank(seed, 'chart', ic[1]['chart_id'])):
        by_mode.setdefault(chart['mode'], []).append((i, chart))
    jobs, truths = [], {}
    for mode in sorted(by_mode):
        selected = by_mode[mode][:max_charts_per_mode] if max_charts_per_mode else by_mode[mode]
        for chart_index, chart in selected:
            arrays = chart_arrays(panel, chart_index)
            duration = int(chart['duration_us'])
            onset = arrays['onset_us']
            base = dict(chart_id=chart['chart_id'], audio_sha256=chart['audio_sha256'], mode=mode,
                        duration_us=duration, stratum=chart.get('stratum'))
            for x in prefix_seconds:
                fill = (int(x*1e6), duration)
                if duration < fill[0] + 2*CORE_US:
                    continue
                visible = (onset < fill[0])
                for conditioned in (True, False):
                    job_id = f'continuation:{x}:{chart["chart_id"]}:{"cond" if conditioned else "uncond"}'
                    jobs.append(dict(base, job_id=job_id, task='continuation', prefix_s=x, fill=fill,
                        conditioned=conditioned, suffix=False, visible_events=_rhythm(arrays, visible),
                        context_regions=[(max(0, fill[0]-HISTORY_US), fill[0])]))
                truths[f'continuation:{x}:{chart["chart_id"]}'] = _rhythm(arrays, ~visible)
            lo_s, hi_s = infill_range_s
            gap_s = lo_s + int(_rank(seed, 'gap', chart['chart_id'])[:8], 16) % (hi_s - lo_s + 1)
            gap = int(gap_s*1e6)
            first, last = MIN_START_US, duration - gap - CORE_US
            if last < first:
                continue
            a = first + int(_rank(seed, 'infill', chart['chart_id'])[8:20], 16) % (last - first + 1)
            a -= a % 1000
            fill = (a, a + gap)
            hidden = (onset >= fill[0]) & (onset < fill[1])
            for conditioned in (True, False):
                job_id = f'infill:{chart["chart_id"]}:{"cond" if conditioned else "uncond"}'
                jobs.append(dict(base, job_id=job_id, task='infill', prefix_s=None, gap_s=gap_s, fill=fill,
                    conditioned=conditioned, suffix=True, visible_events=_rhythm(arrays, ~hidden),
                    context_regions=[(max(0, fill[0]-HISTORY_US), fill[0]),
                                     (fill[1], min(duration, fill[1]+SUFFIX_US))]))
            truths[f'infill:{chart["chart_id"]}'] = _rhythm(arrays, hidden)
    return jobs, truths


def truth_key(job):
    return (f'continuation:{job["prefix_s"]}:{job["chart_id"]}' if job['task'] == 'continuation'
            else f'infill:{job["chart_id"]}')


def _canonical(events, mode, prefix):
    return tuple(Event(f'{prefix}{i}', e.onset_us, e.end_us, e.type,
                       LaneLoc(e.lane) if mode == 'mania_4k' else LineLoc('rhythm', 'above', 0.))
                 for i, e in enumerate(events))


def rollout_sources(state, start, end, *, history_us=HISTORY_US, suffix_us=SUFFIX_US):
    """Human prefix (human_input) + own generated windows (model_history) up to history_us back,
    and the human suffix for the window that closes an infill gap."""
    mode, audio, duration = state['mode'], state['audio_sha256'], state['duration_us']
    fill_start, fill_end = state['fill']
    lower = max(0, start - history_us)
    sources = []
    human_hi = min(start, fill_start)
    if lower < human_hi:
        events = [e for e in state['visible_events'] if lower <= e.onset_us < human_hi]
        sources.append(VisibleReference(f'human-prefix:{state["chart_id"]}', mode, audio, 'human_input',
                                        ((lower, human_hi),), _canonical(events, mode, 'h')))
    model_lo = max(lower, fill_start)
    if model_lo < start:
        events = [e for e in state['generated'] if model_lo <= e.onset_us < start]
        sources.append(VisibleReference(f'generated:{state["job_id"]}', mode, audio, 'model_history',
                                        ((model_lo, start),), _canonical(events, mode, 'g')))
    if state['suffix'] and end == fill_end and fill_end < duration:
        hi = min(duration, fill_end + suffix_us)
        events = [e for e in state['visible_events'] if fill_end <= e.onset_us < hi]
        sources.append(VisibleReference(f'human-suffix:{state["chart_id"]}', mode, audio, 'human_input',
                                        ((fill_end, hi),), _canonical(events, mode, 's')))
    return tuple(sources)


@torch.no_grad()
def rollout_jobs(model, features, jobs, *, device, generator, batch_size=8, token_cap=4096, max_wall_s=1800,
                 intensity_provider=None, history_us=HISTORY_US, suffix_us=SUFFIX_US):
    """Fill each job's region window by window (8 s cores); never reads a hidden human event."""
    if model.training:
        raise ValueError('Rollout requires eval mode')
    states = [dict(job, generated=[], cores=[], cursor=job['fill'][0], complete=False, failure=None) for job in jobs]
    started = time.monotonic()
    deadline = started + max_wall_s
    while True:
        active = [s for s in states if not s['complete'] and s['failure'] is None]
        if not active:
            break
        if time.monotonic() >= deadline:
            for s in active:
                s['failure'] = 'evaluation_wall_limit'
            break
        batch = active[:batch_size]
        rows, requests = [], []
        for state in batch:
            start = state['cursor']
            end = min(start + CORE_US, state['fill'][1])
            known = state['visible_events'] + state['generated']
            carry = tuple(sorted({e for e in known if e.type == 'hold' and e.onset_us < start < e.end_us},
                                 key=lambda e: (e.onset_us, e.lane, e.end_us)))
            request = EventRequest(state['mode'], start, end, state['duration_us'], known_carry=carry)
            sources = rollout_sources(state, start, end, history_us=history_us, suffix_us=suffix_us) \
                if state['conditioned'] else ()
            check_no_leak(sources, request)
            controls = encode_controls(state['mode'], {}, (start, end), origin_us=start)
            inputs, provenance = request_inputs(features, state['audio_sha256'], request, controls, sources=sources)
            if intensity_provider is not None:
                inputs.update(intensity_provider(state['audio_sha256'], start, end))
            rows.append(dict(inputs=inputs, sequence=teacher_sequence(encode_events([], request), request)))
            requests.append(request)
            state['_last'] = dict(start_us=start, end_us=end, carry=len(carry),
                                  reference_rows=int(len(inputs['reference'])),
                                  reference_roles=dict(Counter(r['role'] for r in provenance if r['event_id'])))
        inputs, _ = collate(rows)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        predictions = generator(model, inputs, requests, token_cap, deadline=deadline)
        if len(predictions) != len(batch):
            raise ValueError('Generator returned a different batch size')
        for state, request, prediction in zip(batch, requests, predictions):
            additions = [RhythmEvent(**e) for e in prediction['events']]
            if any(not request.start_us <= e.onset_us < request.end_us for e in additions):
                raise ValueError('Generated event escapes its core')
            state['generated'].extend(additions)
            state['cores'].append(dict(state.pop('_last'), events=len(additions), tokens=prediction['tokens'],
                                       complete=prediction['complete']))
            if not prediction['complete']:
                state['failure'] = ('generation_deadline' if prediction.get('deadline_reached')
                                    else prediction.get('failure') or 'core_token_limit')
            else:
                state['cursor'] = request.end_us
                state['complete'] = request.end_us == state['fill'][1]
    return states, time.monotonic() - started


# -------------------------------------------------------------------- metrics
def _dicts(events):
    return [dict(onset_us=e.onset_us, end_us=e.end_us, type=e.type, lane=e.lane) for e in events]


def _rate(events, regions):
    seconds = sum(hi - lo for lo, hi in regions) / 1e6
    count = sum(any(lo <= e.onset_us < hi for lo, hi in regions) for e in events)
    return count, seconds


def score_job(state, truth, visible_context):
    """Metrics of one filled region vs the hidden human region."""
    from beatfcn.model.evaluation import score_events
    from beatfcn.train.requirement_diagnostics import gap_histogram, js_with_status, onset_gaps_ms
    from beatfcn.train.workflow_metrics import chart_edit_rate
    mode = state['mode']
    draft = sorted(state['generated'], key=lambda e: (e.onset_us, e.lane, e.type))
    lo, hi = state['fill']
    span_s = (hi - lo) / 1e6
    cer = chart_edit_rate(_dicts(draft), _dicts(truth), mode)
    scores = score_events(_canonical(draft, mode, 'd'), _canonical(truth, mode, 't'), mode, tolerances_ms=(50,))
    n_ctx, s_ctx = _rate(visible_context, state['context_regions'])
    ctx_rate = n_ctx / s_ctx if s_ctx else None
    draft_rate, truth_rate = len(draft) / span_s, len(truth) / span_s

    def js(a, b, kind):
        if kind == 'type':
            ca, cb = Counter(e.type for e in a), Counter(e.type for e in b)
            return js_with_status([ca[k] for k in TYPES], [cb[k] for k in TYPES])[0]
        return js_with_status(gap_histogram(onset_gaps_ms(_dicts(a))), gap_histogram(onset_gaps_ms(_dicts(b))))[0]
    context = [e for e in visible_context if any(a <= e.onset_us < b for a, b in state['context_regions'])]
    return dict(
        job_id=state['job_id'], task=state['task'], prefix_s=state.get('prefix_s'), gap_s=state.get('gap_s'),
        chart_id=state['chart_id'], audio_sha256=state['audio_sha256'], mode=mode, conditioned=state['conditioned'],
        complete=state['complete'], failure=state['failure'], fill_us=list(state['fill']),
        draft_events=len(draft), human_events=len(truth),
        cer=cer['cer'], cer_ops=cer['ops'],
        onset=scores['onset_50ms'], note=scores['note_50ms'],
        onset_f1_50ms=scores['onset_50ms']['f1'], note_f1_50ms=scores['note_50ms']['f1'],
        context_rate=ctx_rate, draft_rate=draft_rate, human_rate=truth_rate,
        density_ratio_model=draft_rate / ctx_rate if ctx_rate else None,
        density_ratio_human=truth_rate / ctx_rate if ctx_rate else None,
        log_density_error=math.log((len(draft) + .5) / (len(truth) + .5)),
        type_mix_js_vs_human=js(draft, truth, 'type'), ioi_js_vs_human=js(draft, truth, 'ioi'),
        type_mix_js_vs_context=js(draft, context, 'type'), ioi_js_vs_context=js(draft, context, 'ioi'),
        human_type_mix_js_vs_context=js(truth, context, 'type'), human_ioi_js_vs_context=js(truth, context, 'ioi'),
        cores=len(state['cores']), reference_rows_max=max((c['reference_rows'] for c in state['cores']), default=0))


PAIRED_METRICS = ('cer', 'onset_f1_50ms', 'note_f1_50ms', 'log_density_error', 'type_mix_js_vs_human',
                  'ioi_js_vs_human', 'type_mix_js_vs_context', 'ioi_js_vs_context')
LOWER_IS_BETTER = {'cer', 'type_mix_js_vs_human', 'ioi_js_vs_human', 'type_mix_js_vs_context', 'ioi_js_vs_context'}


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _pooled_f1(rows, key):
    tp = sum(r[key]['tp'] for r in rows)
    pred = sum(r[key]['pred'] for r in rows)
    true = sum(r[key]['true'] for r in rows)
    return 2 * tp / (pred + true) if pred + true else None


def summarize_generation(scored, *, samples=2000, seed=0):
    """Per (task, prefix_s, mode): both arms and the paired conditioned - unconditioned difference."""
    from beatfcn.train.workflow_metrics import bootstrap_mean
    groups = {}
    for row in scored:
        key = (row['task'], row['prefix_s'], row['mode'])
        groups.setdefault(key, {}).setdefault(row['chart_id'], {})['cond' if row['conditioned'] else 'uncond'] = row
    out = []
    for (task, prefix_s, mode), charts in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0, kv[0][2])):
        pairs = [v for v in charts.values() if 'cond' in v and 'uncond' in v]
        entry = dict(task=task, prefix_s=prefix_s, mode=mode, charts=len(pairs),
                     songs=len({v['cond']['audio_sha256'] for v in pairs}),
                     complete_fraction={arm: _mean([float(v[arm]['complete']) for v in pairs]) for arm in ('cond', 'uncond')},
                     arms={}, paired_cond_minus_uncond={})
        for arm in ('cond', 'uncond'):
            rows = [v[arm] for v in pairs]
            by_song = {}
            for r in rows:
                by_song.setdefault(r['audio_sha256'], []).append(r['cer'])
            entry['arms'][arm] = dict(
                cer=_mean([r['cer'] for r in rows]),
                cer_bootstrap=bootstrap_mean([[c for c in v if c is not None] for v in by_song.values()],
                                             samples=samples, seed=seed),
                onset_f1_50ms_pooled=_pooled_f1(rows, 'onset'), note_f1_50ms_pooled=_pooled_f1(rows, 'note'),
                draft_events=int(sum(r['draft_events'] for r in rows)), human_events=int(sum(r['human_events'] for r in rows)),
                density_ratio_model=_mean([r['density_ratio_model'] for r in rows]),
                density_ratio_human=_mean([r['density_ratio_human'] for r in rows]),
                abs_log_density_error=_mean([abs(r['log_density_error']) for r in rows]),
                **{m: _mean([r[m] for r in rows]) for m in PAIRED_METRICS if m not in ('cer', 'log_density_error')},
                human_type_mix_js_vs_context=_mean([r['human_type_mix_js_vs_context'] for r in rows]),
                human_ioi_js_vs_context=_mean([r['human_ioi_js_vs_context'] for r in rows]))
        for metric in PAIRED_METRICS + ('abs_log_density_error',):
            diffs = []
            for v in pairs:
                a, b = v['cond'], v['uncond']
                if metric == 'abs_log_density_error':
                    diffs.append(abs(a['log_density_error']) - abs(b['log_density_error']))
                elif a[metric] is not None and b[metric] is not None:
                    diffs.append(a[metric] - b[metric])
            values = np.asarray(diffs, float)
            entry['paired_cond_minus_uncond'][metric] = dict(
                mean=float(values.mean()) if len(values) else None,
                se=float(values.std(ddof=1)/math.sqrt(len(values))) if len(values) > 1 else None,
                n=int(len(values)),
                better='lower' if metric in LOWER_IS_BETTER or metric == 'abs_log_density_error' else 'higher')
        out.append(entry)
    return out
