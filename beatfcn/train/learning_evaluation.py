"""Held-out generation benchmark v2 with complete raw outputs.

Design (design/BENCHMARK_GUIDE_v2.md):

- All cases of a checkpoint are decoded together by ``generate_batch`` (chunked
  only by ``max_batch_rows`` for memory) under a TOKEN budget of about 2x the
  longest teacher sequence in the panel. Wall time is only a watchdog: a row it
  stops is a reported failure, never a silently truncated success.
- Decoding readouts are kept apart: the primary readout is T=1 ancestral
  sampling with K fixed seeds (per-case ``torch.Generator``), reported as
  mean +- sd over seeds; greedy is secondary. Best-of-K is labelled oracle.
- Results are split into accuracy / human-likeness / condition response /
  hacks-degeneracy / audio grounding, each with a one-line meaning.

The v1 helpers ``metrics``/``summarize``/``run_checkpoint_screen`` keep their
signatures; ``run_checkpoint_screen`` now uses the batched v2 engine and writes
schema ``learning-watch-evaluation-v2`` (old dashboard keys are preserved).
"""
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import inspect
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.full_event_codec import encode_events, teacher_sequence
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.model.evaluation import score_events
from beatfcn.train.audio_dependence import (base_inputs, find_donor, foreign_features, intervene,
    teacher_target, token_losses, val_candidates, validate_windows, _rank)
from beatfcn.train.checkpoint_resume import file_sha
from beatfcn.train.control_metrics import density_request_metrics
from beatfcn.train.counterfactual_density import achieved_density_segments
from beatfcn.train.counterfactual_exposure import DEFAULT_CURVE_CELL_US
from beatfcn.train.local_panel import collate
from beatfcn.train.requirement_benchmarks import as_events
from beatfcn.train import requirement_diagnostics as D

AUDIO_CONDITIONS = ('foreign', 'zero_local_mel')
DENSITY_CONDITIONS = dict(density_low=.5, density_high=2., density_shuffle=2.)
SILENCE_SPLIT_RULE = ('zero-human-note windows only: a predicted event is true_silence if the max frame mean-log-mel '
                      'within +-50 ms of its onset < the song 2nd-percentile frame energy + 1.0 (complete-song mel, '
                      '10 ms frames), else uncharted_music; both rates use the silent-window seconds and sum to '
                      'silence_events_per_s')
DEFAULT_READOUTS = (dict(name='sample_t1', temperature=1., top_p=None, seeds=(0, 1, 2, 3)),
                    dict(name='greedy'))
MEANINGS = dict(
    accuracy='Match against the human chart: onset/note F1 at 20/50/100 ms, per type, hold head+tail; '
             'incomplete rows count against the full truth; covered-prefix F1 + coverage shown separately; '
             'null = same output circularly shifted 1-3 s (F1 that needs no audio alignment).',
    human_likeness='Rhythm statistics vs the same windows\' human charts: gap histogram JS (all gaps, incl. <30 ms), '
                   'fraction of gaps <40 ms, density-profile correlation, type-mix JS, tandem repetition.',
    condition_response='Requested vs produced density for 0.5x/2x/time-reversed reference curves, monotonicity '
                       '(low<free<high), and the accuracy cost of following the request.',
    hacks='Degeneracy detectors: empty positive windows, same output for different songs (template), '
          'silence = music output, events/s on silent windows (split: true_silence = near-silent audio, '
          'uncharted_music = audible music the human left empty), identical output under foreign audio.',
    audio_grounding='Real-minus-foreign / real-minus-zeroed-audio F1 on generated output; teacher-forced CE gaps '
                    'come from the audio-dependence screen.')


# ---------------------------------------------------------------- v1 helpers (kept)
def metrics(prediction, truth, mode):
    predicted, target = as_events(prediction['events'], mode), as_events(truth, mode)
    types = ('tap', 'hold') if mode == 'mania_4k' else ('tap', 'hold', 'flick', 'drag')
    return dict(scores=score_events(predicted, target, mode),
        per_type={kind: score_events([e for e in predicted if e.type == kind],
                                    [e for e in target if e.type == kind], mode)['note_50ms']
                  for kind in types})


def summarize(records, expected):
    positive = [r for r in records if r['case_role'] == 'positive']
    silence = [r for r in records if r['case_role'] == 'natural_silence']
    f1 = [r['metrics']['scores']['note_50ms']['f1'] for r in positive]
    means = lambda values: float(np.mean(values)) if values else None
    by_type = {}
    for record in records:
        for kind, score in record['metrics']['per_type'].items():
            total = by_type.setdefault(kind, Counter())
            total.update({key: score[key] for key in ('tp', 'pred', 'true')})
    for counts in by_type.values():
        counts['precision'] = counts['tp']/counts['pred'] if counts['pred'] else None
        counts['recall'] = counts['tp']/counts['true'] if counts['true'] else None
        counts['f1'] = 2*counts['tp']/(counts['pred']+counts['true']) if counts['pred']+counts['true'] else None
    hold = [r['metrics']['scores']['hold_joint_50ms'] for r in positive]
    pooled = dict(tp=sum(h['tp'] for h in hold), pred=sum(h['pred'] for h in hold),
                  true=sum(h['true'] for h in hold))
    seconds = sum((r['request']['end_us']-r['request']['start_us'])/1e6 for r in silence if 'request' in r)
    false_events = sum(len(r['prediction']['events']) for r in silence)
    return dict(attempted=len(records), expected=expected,
        complete_cases=sum(r['prediction']['complete'] for r in records),
        teacher_joint_ce=means([r['teacher_joint_ce'] for r in records if r.get('teacher_joint_ce') is not None]),
        macro_note_f1_50ms=means([v for v in f1 if v is not None]),
        per_type=by_type,
        hold_joint_f1_50ms=means([r['metrics']['scores']['hold_joint_50ms']['f1']
                                for r in positive if r['metrics']['scores']['hold_joint_50ms']['f1'] is not None]),
        hold_joint_pooled_50ms=dict(pooled, f1=D.f1_of(pooled),
            windows_with_true_holds=sum(h['true'] > 0 for h in hold),
            windows_without_any_hold=sum(h['true'] == 0 and h['pred'] == 0 for h in hold)),
        silence_false_events=false_events,
        silence_false_events_per_s=false_events/seconds if seconds else None,
        silence_cases=len(silence),
        scope='type-diverse held-out short windows; not whole-song or playable-chart acceptance')


# ---------------------------------------------------------------- panel freeze
def _types(panel, index):
    return panel.records[index].get('types') or {}


def freeze_generation_panel(panel, *, per_mode_positive=6, per_mode_silence=2, anchors_per_mode=2,
                            seed=20261101, tape_sha256=None, cell_us=100_000,
                            audio_conditions=AUDIO_CONDITIONS, density_conditions=tuple(DENSITY_CONDITIONS)):
    """Deterministic, type-diverse free windows + silent windows + counterfactual anchors."""
    candidates = val_candidates(panel)
    samples = []
    for mode in ('phira', 'mania_4k'):
        rows = [i for i in candidates if panel.records[i]['mode'] == mode]
        by_audio = {}
        for i in rows:
            if panel.records[i].get('event_count', 0) > 0:
                by_audio.setdefault(panel.records[i]['audio_sha256'], []).append(i)
        best = [min(v, key=lambda i: (-len(_types(panel, i)), _rank(seed, 'free', i))) for v in by_audio.values()]
        best.sort(key=lambda i: _rank(seed, 'song', panel.records[i]['audio_sha256']))
        chosen = best[:per_mode_positive]
        silent = sorted((i for i in rows if panel.records[i].get('event_count', 0) == 0
                         and panel.records[i]['request']['end_us']-panel.records[i]['request']['start_us'] >= 4_000_000),
                        key=lambda i: _rank(seed, 'silence', i))
        quiet, seen = [], set()
        for i in silent:
            if panel.records[i]['audio_sha256'] not in seen and len(quiet) < per_mode_silence:
                quiet.append(i)
                seen.add(panel.records[i]['audio_sha256'])
        for i in chosen:
            samples.append(dict(_sample(panel, i), case_role='positive'))
        for i in quiet:
            samples.append(dict(_sample(panel, i), case_role='natural_silence'))
    protocol = dict(schema='benchmark-v2-generation-panel', tape_sha256=tape_sha256, seed=seed,
        samples=samples, cell_us=cell_us, audio_conditions=list(audio_conditions),
        density_conditions={k: DENSITY_CONDITIONS[k] for k in density_conditions},
        selection=f'hash-ordered distinct val songs, max type diversity; seed {seed}')
    protocol['anchors'] = derive_anchors(panel, protocol, per_mode=anchors_per_mode)
    return protocol


def _sample(panel, index):
    row = panel.records[index]
    chart = panel.charts[row['chart_index']]
    return dict(index=index, chart_id=row['chart_id'], audio_sha256=row['audio_sha256'], component=row['component'],
        split=row['split'], mode=row['mode'], request=row['request'], target_sha256=chart.get('target_sha256'),
        event_count=row.get('event_count'), type_counts=row.get('types', {}))


def derive_anchors(panel, protocol, *, per_mode=2):
    """Counterfactual anchors: positives with a matched foreign donor (panel first, then any val)."""
    anchors, seed = [], protocol.get('seed', 0)
    candidates = val_candidates(panel)
    for mode in ('phira', 'mania_4k'):
        positive = [s for s in protocol['samples'] if s['mode'] == mode and s['case_role'] == 'positive']
        panel_ids = [s['index'] for s in positive]
        for sample in positive:
            if sum(a['mode'] == mode for a in anchors) >= per_mode:
                break
            donor = find_donor(panel, sample['index'], panel_ids, seed) or find_donor(
                panel, sample['index'], candidates, seed)
            if donor is not None:
                anchors.append(dict(index=sample['index'], mode=mode, foreign_index=donor,
                                    foreign_audio_sha256=panel.records[donor]['audio_sha256']))
    return anchors


def validate_generation_panel(panel, protocol):
    validate_windows(panel, protocol['samples'])
    seen = {s['index'] for s in protocol['samples']}
    if len(seen) != len(protocol['samples']):
        raise ValueError('Duplicate frozen sample')
    for anchor in protocol.get('anchors', ()):
        if anchor['index'] not in seen:
            raise ValueError('Anchor outside the frozen samples')
        donor = panel.records[anchor['foreign_index']]
        row = panel.records[anchor['index']]
        if donor['audio_sha256'] == row['audio_sha256'] or donor['mode'] != row['mode']:
            raise ValueError('Foreign audio requires another held-out song in the same mode')


# ---------------------------------------------------------------- cases
def build_cases(panel, protocol):
    """Free cases for every sample, plus audio/density counterfactuals for anchors."""
    cell = protocol.get('cell_us', 100_000)
    cases = []
    samples = {s['index']: s for s in protocol['samples']}
    for sample in protocol['samples']:
        cases.append(dict(case_id=f'{sample["index"]}:free', index=sample['index'], condition='free',
                          case_role=sample['case_role'], mode=sample['mode']))
    for anchor in protocol.get('anchors', ()):
        base = dict(index=anchor['index'], case_role=samples[anchor['index']]['case_role'], mode=anchor['mode'])
        for condition in protocol.get('audio_conditions', AUDIO_CONDITIONS):
            cases.append(dict(base, case_id=f'{anchor["index"]}:{condition}', condition=condition,
                              foreign_index=anchor['foreign_index']))
        for condition in protocol.get('density_conditions', DENSITY_CONDITIONS):
            cases.append(dict(base, case_id=f'{anchor["index"]}:{condition}', condition=condition))
    for case in cases:
        case['cell_us'] = cell
        case['curve_cell_us'] = int(protocol.get('curve_cell_us', DEFAULT_CURVE_CELL_US))
        case['density_factor'] = protocol.get('density_conditions', DENSITY_CONDITIONS).get(case['condition'])
    return cases


def case_inputs(panel, case, charter_timing=None):
    events = panel.targets(case['index'])
    row = panel.records[case['index']]
    request = row['request']
    curve = None
    if case['condition'].startswith('density_'):
        # Same coarse curve semantics as training (achieved heads/s per
        # curve_cell_us, default 2 s), rendered on the 100 ms grid.
        baseline = achieved_density_segments(events, request['start_us'], request['end_us'],
                                             cell_us=int(case.get('curve_cell_us', DEFAULT_CURVE_CELL_US)))
        values = [s['density']*case['density_factor'] for s in baseline]
        if case['condition'] == 'density_shuffle':
            values = list(reversed(values))
        curve = [dict(start_us=s['start_us'], end_us=s['end_us'], density=v) for s, v in zip(baseline, values)]
    inputs, request_object = base_inputs(panel, case['index'], cell_us=case['cell_us'], curve=curve,
                                         charter_timing=charter_timing)
    if case['condition'] in AUDIO_CONDITIONS or case['condition'] == 'roll':
        inputs = intervene(inputs, case['condition'],
            foreign=foreign_features(panel, case['foreign_index']) if case['condition'] == 'foreign' else None,
            roll_s=case.get('roll_s'))
    return inputs, request_object, [asdict(e) for e in events], curve


def token_budget(panel, protocol, vocab, factor=2.):
    """~factor x the longest teacher sequence in the panel (never a wall-time cut)."""
    from beatfcn.train.local_request_panel import event_request
    longest = 1
    for sample in protocol['samples']:
        request = event_request(panel.records[sample['index']]['request'])
        longest = max(longest, len(encode_events(panel.targets(sample['index']), request, vocab)))
    return int(math.ceil(factor*longest)), longest


# ---------------------------------------------------------------- decoding
def sampling_supported():
    return 'temperature' in inspect.signature(generate_batch).parameters


def _row_seed(base_seed, case_id, seed):
    return int(hashlib.sha256(f'{base_seed}:{case_id}:{seed}'.encode()).hexdigest()[:15], 16)


_ROW_GENERATORS = {'supported': None}


def _call_generate(model, inputs, requests, max_tokens, *, deadline, readout, seeds, device, chunk_seed,
                   unconditional=None):
    """Greedy, or T>0 ancestral sampling with per-row generators when supported.

    If generate_batch only accepts one generator, the chunk uses a single
    generator seeded from (base seed, readout, chunk): reproducible for the same
    frozen chunking, but per-case draws then depend on batch composition. The
    rng mode is recorded on every row.
    """
    guided = {}
    if readout.get('guidance') is not None:
        # Classifier-free guidance readout (decoding aid, reported separately).
        guided = dict(guidance=float(readout['guidance']), unconditional_inputs=unconditional)
    if readout.get('temperature') is None:
        if readout.get('onset_fusion') is not None:
            guided['onset_fusion'] = _onset_fusion_option(readout['onset_fusion'])
        return generate_batch(model, inputs, requests, max_tokens, deadline=deadline, **guided), 'greedy'
    kwargs = dict(deadline=deadline, temperature=float(readout['temperature']), top_p=readout.get('top_p'), **guided)
    if readout.get('field_temperature'):
        kwargs['field_temperature'] = dict(readout['field_temperature'])
    if readout.get('onset_fusion') is not None:
        kwargs['onset_fusion'] = _onset_fusion_option(readout['onset_fusion'])
    generator_device = device.type if device.type in ('cuda', 'mps') else 'cpu'
    if _ROW_GENERATORS['supported'] is not False:
        generators = [torch.Generator(device=generator_device).manual_seed(s) for s in seeds]
        try:
            result = generate_batch(model, inputs, requests, max_tokens, generator=generators, **kwargs)
            _ROW_GENERATORS['supported'] = True
            return result, 'per_row'
        except (TypeError, AttributeError) as exc:
            if _ROW_GENERATORS['supported'] or not any(w in str(exc).lower() for w in ('generator', 'list', 'device')):
                raise
            _ROW_GENERATORS['supported'] = False
    generator = torch.Generator(device=generator_device).manual_seed(chunk_seed)
    return generate_batch(model, inputs, requests, max_tokens, generator=generator, **kwargs), 'shared_batch_generator'


def _onset_fusion_option(value):
    """Readout ``onset_fusion``: a weight (e.g. .5) or generate_batch's dict(weight=..)."""
    return dict(value) if isinstance(value, dict) else dict(weight=float(value))


def silence_splits(panel, rows):
    """Attach ``silence_split`` (true-silence vs uncharted-music event counts) to every
    free row whose window has zero human notes, from the song's complete mel
    (``panel.features.mel_frames``). None when the features expose no mel."""
    mel_frames = getattr(getattr(panel, 'features', None), 'mel_frames', None)
    songs = {}
    for row in rows:
        if row['condition'] != 'free' or row['truth']:
            continue
        audio = row['audio_sha256']
        if audio not in songs:
            try:
                songs[audio] = D.song_energy(mel_frames(audio)) if mel_frames is not None else None
            except (KeyError, OSError, ValueError):
                songs[audio] = None
        song = songs[audio]
        row['silence_split'] = D.silence_split(row['prediction']['events'], song) if song is not None else None


def classify(prediction, case_role):
    if prediction.get('failure'):
        return 'generation_failure'
    if prediction.get('deadline_reached'):
        return 'watchdog'
    if prediction.get('token_limit_reached'):
        return 'token_budget'
    if not prediction.get('complete'):
        return 'incomplete'
    if case_role == 'positive' and not prediction.get('events'):
        return 'empty_positive'
    return None


def decode_rows(model, rows, *, device, max_tokens, deadline, max_batch_rows=32, vocab=None, log=None):
    """Decode every row; each readout gets its own generate_batch call(s)."""
    device = torch.device(device)
    order = []
    for row in rows:
        if row['readout']['name'] not in order:
            order.append(row['readout']['name'])
    for name in order:
        group = [r for r in rows if r['readout']['name'] == name]
        for first in range(0, len(group), max_batch_rows):
            chunk = group[first:first+max_batch_rows]
            if time.monotonic() >= deadline:
                for row in chunk:
                    row['prediction'] = dict(complete=False, events=[], tokens=[], deadline_reached=True,
                                             token_limit_reached=False, failure=None, not_started=True)
                continue
            dummies = [dict(inputs=r['inputs'], sequence=teacher_sequence(
                encode_events([], r['request_object'], vocab), r['request_object'], vocab)) for r in chunk]
            inp, _ = collate(dummies)
            inp = {k: v.to(device) for k, v in inp.items()}
            unconditional = None
            if chunk[0]['readout'].get('guidance') is not None:
                from beatfcn.train.counterfactual_exposure import unconditional_inputs
                bare = [dict(d, inputs=dict(d['inputs'], **unconditional_inputs(
                    r['request_object'].mode, r['request_object']))) for d, r in zip(dummies, chunk)]
                unconditional, _ = collate(bare)
                unconditional = {k: v.to(device) for k, v in unconditional.items()}
            started = time.monotonic()
            try:
                chunk_seed = _row_seed(chunk[0].get('base_seed', 0), name, first)
                predictions, rng_mode = _call_generate(model, inp, [r['request_object'] for r in chunk], max_tokens,
                    deadline=deadline, readout=chunk[0]['readout'], seeds=[r['seed'] for r in chunk], device=device,
                    chunk_seed=chunk_seed, unconditional=unconditional)
            except ValueError as exc:
                if 'dead end' not in str(exc):
                    raise
                # Legacy generator raises for the whole batch; every row is a failure.
                predictions = [dict(complete=False, events=[], tokens=[], deadline_reached=False,
                                    token_limit_reached=False, failure='grammar_dead_end_batch') for _ in chunk]
                rng_mode = None
            elapsed = time.monotonic()-started
            for row, prediction in zip(chunk, predictions):
                prediction = dict(prediction)
                prediction.setdefault('failure', None)
                row['prediction'] = prediction
                row['rng_mode'] = rng_mode
                row['chunk_generator_seed'] = chunk_seed if rng_mode == 'shared_batch_generator' else None
                row['chunk_elapsed_s'] = elapsed
                row['chunk_rows'] = len(chunk)
            if log:
                log(dict(phase='benchmark_v2_chunk', readout=name, rows=len(chunk), elapsed_s=round(elapsed, 2),
                         complete=sum(bool(r['prediction']['complete']) for r in chunk)))
            del inp


# ---------------------------------------------------------------- scoring
def _anchor_rows(rows):
    by_anchor = {}
    for row in rows:
        by_anchor.setdefault(row['index'], {})[row['condition']] = row
    return by_anchor


def _rate(events, request):
    return len(events)/((request['end_us']-request['start_us'])/1e6)


def score_readout(rows):
    """One decoding rule and one seed: the five benchmark categories."""
    free = [r for r in rows if r['condition'] == 'free']
    positive = [r for r in free if r['case_role'] == 'positive']
    silence = [r for r in free if r['case_role'] == 'natural_silence']
    complete_positive = [r for r in positive if r['failure'] is None or r['failure'] == 'empty_positive']
    full = D.pool_counts([r['scores']['full'] for r in positive])
    prefix = D.pool_counts([r['scores']['prefix'] for r in positive])
    null = D.pool_counts([{k: v for k, v in r['scores']['null_shift'].items()} for r in positive]) if positive else {}
    complete_full = D.pool_counts([r['scores']['full'] for r in complete_positive])
    pick = lambda pooled, key: (pooled.get(key) or {}).get('f1')
    note50, null50 = pick(full, 'note_50ms'), pick(null, 'note_50ms')
    hold = full.get('hold_joint_50ms', dict(tp=0, pred=0, true=0, f1=None))
    accuracy = dict(meaning=MEANINGS['accuracy'], positive_cases=len(positive),
        complete_positive_cases=len(complete_positive),
        complete_fraction=len(complete_positive)/len(positive) if positive else None,
        onset_f1={ms: pick(full, f'onset_{ms}ms') for ms in D.TOLERANCES_MS},
        note_f1={ms: pick(full, f'note_{ms}ms') for ms in D.TOLERANCES_MS},
        note_f1_50ms=note50, null_shift_note_f1_50ms=null50,
        note_f1_50ms_minus_null=(note50-null50) if note50 is not None and null50 is not None else None,
        onset_f1_50ms_null=pick(null, 'onset_50ms'),
        per_type_f1_50ms={k[5:-5]: pick(full, k) for k in full if k.startswith('type_')},
        per_type_counts_50ms={k[5:-5]: {n: full[k][n] for n in ('tp', 'pred', 'true')} for k in full if k.startswith('type_')},
        hold_joint_f1_50ms=hold.get('f1'), hold_joint_f1_100ms=pick(full, 'hold_joint_100ms'),
        hold_joint_counts_50ms=dict(tp=hold['tp'], pred=hold['pred'], true=hold['true'],
            windows_with_true_holds=sum(r['scores']['has_true_holds'] for r in positive),
            windows_with_pred_holds=sum(r['scores']['has_pred_holds'] for r in positive),
            windows_without_any_hold=sum(not r['scores']['has_true_holds'] and not r['scores']['has_pred_holds']
                                         for r in positive)),
        complete_case_note_f1_50ms=pick(complete_full, 'note_50ms'),
        prefix_note_f1_50ms=pick(prefix, 'note_50ms'), prefix_onset_f1_50ms=pick(prefix, 'onset_50ms'),
        mean_coverage_fraction=float(np.mean([r['scores']['coverage_fraction'] for r in positive])) if positive else None,
        failures=dict(Counter(r['failure'] for r in rows if r['failure'] is not None)),
        pooled_counts=full)
    pairs = [(r['prediction']['events'], r['truth'], r['request']) for r in complete_positive]
    human = dict(D.human_likeness(pairs), meaning=MEANINGS['human_likeness'],
                 scope='complete free positive outputs only; incomplete rows excluded and counted',
                 excluded_incomplete=len(positive)-len(complete_positive))
    conditions, grounding = [], []
    for index, group in sorted(_anchor_rows(rows).items()):
        real = group.get('free')
        if real is None:
            continue
        done = lambda r: r is not None and r['prediction'].get('complete')
        entry = dict(index=index, mode=real['mode'])
        rates = {}
        for name in ('free',)+tuple(DENSITY_CONDITIONS):
            r = group.get(name)
            if r is not None:
                rates[name] = dict(produced_events_per_s=_rate(r['prediction']['events'], r['request']),
                                   complete=bool(done(r)))
                if r.get('requested_curve'):
                    m = density_request_metrics(r['prediction']['events'], r['requested_curve'])
                    rates[name].update(requested_events_per_s=float(np.mean([s['density'] for s in r['requested_curve']])),
                                       mae_events_per_s=m['mae_heads_per_s'], profile_pearson=m['pearson'])
                rates[name]['note_f1_50ms_vs_human'] = D.f1_of(r['scores']['full']['note_50ms'])
        entry['rates'] = rates
        low, high, free_row, shuffle = (group.get(k) for k in ('density_low', 'density_high', 'free', 'density_shuffle'))
        if all(done(r) for r in (low, high, free_row)):
            a, b, c = (rates[k]['produced_events_per_s'] for k in ('density_low', 'free', 'density_high'))
            free_f1 = D.f1_of(free_row['scores']['full']['note_50ms']) or 0.
            entry.update(monotonic=a < b < c, high_over_low=(c/a) if a else (math.inf if c else None),
                requested_high_over_low=4.,
                accuracy_cost_high=free_f1-(rates['density_high']['note_f1_50ms_vs_human'] or 0.),
                accuracy_cost_low=free_f1-(rates['density_low']['note_f1_50ms_vs_human'] or 0.),
                low_high_identical=D.same_relative_multiset(low['prediction']['events'], high['prediction']['events'],
                                                            low['request']['start_us'], high['request']['start_us']))
        else:
            entry.update(monotonic=None, censored=True)
        if done(shuffle) and done(high) and high.get('requested_curve'):
            entry['shuffle'] = dict(pearson_with_reversed_request=rates['density_shuffle'].get('profile_pearson'),
                pearson_with_unreversed_request=density_request_metrics(
                    shuffle['prediction']['events'], high['requested_curve'])['pearson'],
                note='follows the request if the first exceeds the second')
        conditions.append(entry)
        for name in AUDIO_CONDITIONS:
            other = group.get(name)
            if other is None:
                continue
            both = done(real) and done(other)
            grounding.append(dict(index=index, mode=real['mode'], intervention=name, complete=bool(both),
                real_note_f1_50ms=D.f1_of(real['scores']['full']['note_50ms']),
                counterfactual_note_f1_50ms=D.f1_of(other['scores']['full']['note_50ms']),
                real_minus_counterfactual_note_f1_50ms=(D.f1_of(real['scores']['full']['note_50ms']) or 0.)-(
                    D.f1_of(other['scores']['full']['note_50ms']) or 0.) if both else None,
                identical_output=D.same_relative_multiset(real['prediction']['events'], other['prediction']['events'],
                    real['request']['start_us'], other['request']['start_us']) if both else None,
                output_similarity=D.relative_similarity(real['prediction']['events'], other['prediction']['events'],
                    real['request']['start_us'], other['request']['start_us'], real['mode']) if both else None))
    monotone = [c['monotonic'] for c in conditions if c.get('monotonic') is not None]
    condition = dict(meaning=MEANINGS['condition_response'], anchors=conditions,
        monotonic_fraction=float(np.mean(monotone)) if monotone else None, measured_anchors=len(monotone),
        mean_high_over_low=float(np.mean([c['high_over_low'] for c in conditions
            if c.get('high_over_low') is not None and math.isfinite(c['high_over_low'])])) if any(
            c.get('high_over_low') is not None and math.isfinite(c['high_over_low']) for c in conditions) else None,
        mean_request_mae_events_per_s=_mean([r.get('mae_events_per_s') for c in conditions
                                             for r in c['rates'].values() if r.get('complete')]))
    # Hacks / degeneracy.
    templates = {}
    for mode in sorted({r['mode'] for r in positive}):
        rows_mode = [r for r in complete_positive if r['mode'] == mode]
        model_t = D.template_similarity([(r['prediction']['events'], r['request']['start_us'], r['audio_sha256'])
                                         for r in rows_mode], mode)
        human_t = D.template_similarity([(r['truth'], r['request']['start_us'], r['audio_sha256'])
                                         for r in rows_mode], mode)
        templates[mode] = dict(model=model_t, human=human_t,
            excess=(model_t['mean']-human_t['mean']) if model_t['mean'] is not None and human_t['mean'] is not None else None)
    silent = []
    for r in silence:
        same_mode = [p for p in complete_positive if p['mode'] == r['mode']]
        split = r.get('silence_split') or {}
        silent.append(dict(index=r['index'], mode=r['mode'], complete=bool(r['prediction'].get('complete')),
            true_silence_events=split.get('true_silence_events'),
            uncharted_music_events=split.get('uncharted_music_events'),
            events=len(r['prediction']['events']), events_per_s=_rate(r['prediction']['events'], r['request']),
            max_similarity_to_music_output=max((D.relative_similarity(r['prediction']['events'], p['prediction']['events'],
                r['request']['start_us'], p['request']['start_us'], r['mode']) for p in same_mode), default=None),
            identical_to_music_outputs=sum(D.same_relative_multiset(r['prediction']['events'], p['prediction']['events'],
                r['request']['start_us'], p['request']['start_us']) for p in same_mode)))
    silence_seconds = sum((r['request']['end_us']-r['request']['start_us'])/1e6 for r in silence)
    split_measured = bool(silent) and all(s['true_silence_events'] is not None for s in silent)
    foreign = [g for g in grounding if g['intervention'] == 'foreign' and g['complete']]
    empty = [r for r in positive if not r['prediction']['events']]
    hacks = dict(meaning=MEANINGS['hacks'],
        empty_positive_rate=len(empty)/len(positive) if positive else None, empty_positive_cases=len(empty),
        all_positive_empty=bool(positive) and len(empty) == len(positive),
        template=templates,
        template_excess_mean=_mean([t['excess'] for t in templates.values()]),
        silence=silent,
        silence_events_per_s=(sum(s['events'] for s in silent)/silence_seconds) if silence_seconds else None,
        true_silence_events_per_s=(sum(s['true_silence_events'] for s in silent)/silence_seconds)
            if silence_seconds and split_measured else None,
        uncharted_music_events_per_s=(sum(s['uncharted_music_events'] for s in silent)/silence_seconds)
            if silence_seconds and split_measured else None,
        silence_split_rule=SILENCE_SPLIT_RULE if split_measured else None,
        silence_all_complete=bool(silent) and all(s['complete'] for s in silent),
        silence_identical_to_music=sum(s['identical_to_music_outputs'] > 0 for s in silent),
        foreign_identical_fraction=float(np.mean([g['identical_output'] for g in foreign])) if foreign else None,
        foreign_output_similarity=_mean([g['output_similarity'] for g in foreign]),
        density_low_high_identical=sum(bool(c.get('low_high_identical')) for c in conditions))
    audio = dict(meaning=MEANINGS['audio_grounding'], generation=grounding,
        real_minus_foreign_note_f1_50ms=_mean([g['real_minus_counterfactual_note_f1_50ms'] for g in grounding
                                               if g['intervention'] == 'foreign']),
        real_minus_zero_note_f1_50ms=_mean([g['real_minus_counterfactual_note_f1_50ms'] for g in grounding
                                            if g['intervention'] == 'zero_local_mel']))
    return dict(accuracy=accuracy, human_likeness=human, condition_response=condition, hacks=hacks,
                audio_grounding=audio)


def _mean(values):
    values = [v for v in values if v is not None and (not isinstance(v, float) or math.isfinite(v))]
    return float(np.mean(values)) if values else None


HEADLINES = (('accuracy', 'note_f1_50ms'), ('accuracy', 'note_f1_50ms_minus_null'), ('accuracy', 'hold_joint_f1_50ms'),
             ('accuracy', 'complete_fraction'), ('accuracy', 'prefix_note_f1_50ms'), ('accuracy', 'mean_coverage_fraction'),
             ('human_likeness', 'ioi_js_nats'), ('human_likeness', 'short_gap_fraction_model'),
             ('human_likeness', 'type_mix_js_nats'), ('human_likeness', 'density_profile_pearson'),
             ('condition_response', 'monotonic_fraction'), ('condition_response', 'mean_high_over_low'),
             ('condition_response', 'mean_request_mae_events_per_s'),
             ('hacks', 'empty_positive_rate'), ('hacks', 'template_excess_mean'), ('hacks', 'silence_events_per_s'),
             ('hacks', 'true_silence_events_per_s'), ('hacks', 'uncharted_music_events_per_s'),
             ('hacks', 'foreign_identical_fraction'),
             ('audio_grounding', 'real_minus_foreign_note_f1_50ms'), ('audio_grounding', 'real_minus_zero_note_f1_50ms'))


def aggregate_seeds(per_seed):
    """mean +- sd over seeds for each headline scalar (one decoding rule)."""
    out = {}
    for category, key in HEADLINES:
        values = [s[category].get(key) for s in per_seed.values()]
        finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
        out[f'{category}.{key}'] = dict(mean=float(np.mean(finite)) if finite else None,
            sd=float(np.std(finite, ddof=1)) if len(finite) > 1 else None, seeds_defined=len(finite),
            seeds=len(values))
    return out


def oracle_best_of_k(rows):
    """Per-case best note F1@50 over seeds. ORACLE: uses the answer to choose; not a capability."""
    best = {}
    for row in rows:
        if row['condition'] != 'free' or row['case_role'] != 'positive':
            continue
        value = D.f1_of(row['scores']['full']['note_50ms']) or 0.
        best[row['case_id']] = max(best.get(row['case_id'], 0.), value)
    return dict(label='ORACLE best-of-K (selected with the human answer; never a capability claim)',
                mean_note_f1_50ms=_mean(list(best.values())), cases=len(best))


def run_generation_benchmark(model, panel, protocol, *, device, deadline, readouts=DEFAULT_READOUTS,
                             token_budget_factor=2., max_batch_rows=32, base_seed=20261101, log=None,
                             charter_timing=None):
    """Decode all cases x readouts, score them, and return complete raw rows + summaries.

    ``charter_timing`` (optional ChartTimingTable): every case also gets its own
    chart's beat features; the result is labelled ``charter_timing_input``."""
    validate_generation_panel(panel, protocol)
    vocab = getattr(model, 'vocab', None)
    device = torch.device(device)
    budget, longest = token_budget(panel, protocol, vocab, token_budget_factor)
    cases = build_cases(panel, protocol)
    prepared = {}
    for case in cases:
        prepared[case['case_id']] = case_inputs(panel, case, charter_timing)
    skipped = []
    if not sampling_supported():
        skipped = [r['name'] for r in readouts if r.get('temperature') is not None]
        readouts = [r for r in readouts if r.get('temperature') is None]
    samples = {s['index']: s for s in protocol['samples']}
    rows = []
    for readout in readouts:
        seeds = readout.get('seeds', (0,)) if readout.get('temperature') is not None else (None,)
        for seed in seeds:
            for case in cases:
                inputs, request_object, truth, curve = prepared[case['case_id']]
                rows.append(dict(case, readout=dict(readout, seeds=list(readout.get('seeds', ()))),
                    readout_seed=seed, seed=_row_seed(base_seed, case['case_id'], seed) if seed is not None else None,
                    base_seed=base_seed,
                    inputs=inputs, request_object=request_object, truth=truth, requested_curve=curve,
                    request=samples[case['index']]['request'], audio_sha256=samples[case['index']]['audio_sha256'],
                    chart_id=samples[case['index']]['chart_id']))
    started = time.monotonic()
    decode_rows(model, rows, device=device, max_tokens=budget, deadline=deadline,
                max_batch_rows=max_batch_rows, vocab=vocab, log=log)
    for row in rows:
        row.pop('inputs')
        row.pop('request_object')
        row['failure'] = classify(row['prediction'], row['case_role'])
        row['scores'] = D.window_accuracy(row['prediction'], row['truth'], row['request'])
        row['diagnostics'] = D.diagnose(row['prediction'], row['truth'], row['request'],
            requested_curve=row['requested_curve'], reference_applicable=not row['condition'].startswith('density_'))
    silence_splits(panel, rows)
    readout_summaries = {}
    for readout in readouts:
        name = readout['name']
        mine = [r for r in rows if r['readout']['name'] == name]
        per_seed = {str(seed): score_readout([r for r in mine if r['readout_seed'] == seed])
                    for seed in sorted({r['readout_seed'] for r in mine}, key=lambda s: (s is None, s))}
        summary = dict(readout={k: v for k, v in readout.items()}, per_seed=per_seed)
        if readout.get('temperature') is not None:
            summary['primary'] = True
            summary['mean_sd_over_seeds'] = aggregate_seeds(per_seed)
            summary['oracle'] = oracle_best_of_k(mine)
            summary['rng_modes'] = dict(Counter(r.get('rng_mode') for r in mine))
        readout_summaries[name] = summary
    extra = {}
    if charter_timing is not None:
        for row in rows:
            row['charter_timing_present'] = charter_timing.segments(row['chart_id']) is not None
        extra['charter_timing_input'] = dict(charter_timing.identity(),
            cases_with_timing=sum(charter_timing.segments(samples[c['index']]['chart_id']) is not None for c in cases),
            cases=len(cases))
    return dict(**extra, rows=rows, readouts=readout_summaries, token_budget=budget, longest_teacher_tokens=longest,
        token_budget_factor=token_budget_factor, readouts_skipped=skipped,
        readouts_skipped_reason=('generate_batch lacks temperature/top_p/generator' if skipped else None),
        elapsed_s=time.monotonic()-started, expected_rows=len(rows),
        complete_rows=sum(bool(r['prediction'].get('complete')) for r in rows),
        failure_counts=dict(Counter(r['failure'] for r in rows if r['failure'])), meanings=MEANINGS)


def teacher_ce_for_samples(model, panel, protocol, *, device, batch_size=8, deadline=None, charter_timing=None):
    """Joint teacher CE per free sample (real audio), for the legacy dashboard keys."""
    device = torch.device(device)
    vocab = getattr(model, 'vocab', None)
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    out = {}
    samples = protocol['samples']
    for first in range(0, len(samples), batch_size):
        if deadline is not None and time.monotonic() >= deadline:
            break
        rows = []
        for sample in samples[first:first+batch_size]:
            inputs, request = base_inputs(panel, sample['index'], cell_us=protocol.get('cell_us', 100_000),
                                          charter_timing=charter_timing)
            rows.append(dict(inputs=inputs, sequence=teacher_target(panel, sample['index'], request, vocab)[1]))
        inp, seq = collate(rows)
        inp = {k: v.to(device) for k, v in inp.items()}
        seq = {k: v.to(device) for k, v in seq.items()}
        with torch.no_grad(), amp:
            losses, _, _ = token_losses(model, inp, seq)
        losses, valid = losses.float().cpu(), seq['valid'].cpu()
        for row, sample in enumerate(samples[first:first+batch_size]):
            out[sample['index']] = float(losses[row][valid[row]].mean())
    return out


def legacy_records(benchmark, protocol, teacher_ce):
    """v1-shaped rows (primary readout, first seed) so old dashboards keep working."""
    primary = next((name for name, s in benchmark['readouts'].items() if s.get('primary')),
                   next(iter(benchmark['readouts']), None))
    if primary is None:
        return [], primary
    seeds = [r['readout_seed'] for r in benchmark['rows'] if r['readout']['name'] == primary]
    first = seeds[0] if seeds else None
    chosen = {r['index']: r for r in benchmark['rows'] if r['readout']['name'] == primary
              and r['readout_seed'] == first and r['condition'] == 'free'}
    records = []
    for sample in protocol['samples']:
        row = chosen.get(sample['index'])
        if row is None:
            continue
        records.append(dict(sample, teacher_joint_ce=teacher_ce.get(sample['index']), truth=row['truth'],
            prediction=row['prediction'], metrics=metrics(row['prediction'], row['truth'], sample['mode']),
            readout=primary, readout_seed=first))
    return records, primary


def evaluate_model(model, panel, protocol, *, device, deadline, per_case_s=None, max_tokens=None,
                   readouts=DEFAULT_READOUTS, token_budget_factor=2., max_batch_rows=32, log=None,
                   charter_timing=None):
    """v2 batched evaluation of one loaded model; returns (legacy_records, benchmark).

    ``charter_timing`` (optional ChartTimingTable) = the charter_timing_input readout."""
    if not protocol.get('anchors') and protocol.get('derive_anchors', True):
        protocol = dict(protocol, anchors=derive_anchors(panel, protocol))
    teacher = teacher_ce_for_samples(model, panel, protocol, device=device, deadline=deadline,
                                     charter_timing=charter_timing)
    benchmark = run_generation_benchmark(model, panel, protocol, device=device, deadline=deadline,
        readouts=readouts, token_budget_factor=token_budget_factor, max_batch_rows=max_batch_rows, log=log,
        charter_timing=charter_timing)
    benchmark['anchors'] = protocol.get('anchors', [])
    records, primary = legacy_records(benchmark, protocol, teacher)
    benchmark['legacy_records_readout'] = primary
    return records, benchmark


def write_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default)+'\n')
    temporary.replace(path)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f'Not JSON serializable: {type(value)}')


def sanitize(value):
    """Replace non-finite floats (e.g. inf ratios) by None so the JSON stays strict."""
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_checkpoint_screen(model, panel, plan, *, root, device):
    """Frozen-checkpoint screen (v1 plan keys) using the batched v2 engine.

    plan['learning_eval'] keys: protocol, protocol_sha256, output, plan_path,
    checkpoints[{path,sha256,exposure_step}], max_wall_s, checkpoint_wall_s,
    optional readouts, token_budget_factor, max_batch_rows. ``per_case_s`` and
    ``max_tokens`` from v1 plans are ignored on purpose (token budget + watchdog).
    """
    settings = plan['learning_eval']
    root = Path(root)
    protocol_path = root/settings['protocol']
    if file_sha(protocol_path) != settings['protocol_sha256']:
        raise ValueError('Learning evaluation protocol identity changed')
    protocol = json.loads(protocol_path.read_text())
    if protocol['tape_sha256'] != plan['inputs_sha256'][plan['tape']]:
        raise ValueError('Evaluation tape differs from training data')
    output = root/settings['output']
    checkpoints = settings['checkpoints']
    current = plan['resume_checkpoint']
    if checkpoints[-1]['path'] != current['path'] or checkpoints[-1]['sha256'] != current['sha256']:
        raise ValueError('Evaluation must end at the exact continuation checkpoint')
    readouts = settings.get('readouts', DEFAULT_READOUTS)
    from beatfcn.train.charter_timing import evaluation_table
    timing = evaluation_table(plan, model, root=root, choice=settings.get('charter_timing'))
    timing_kw = {} if timing is None else dict(charter_timing=timing)
    result = dict(schema='learning-watch-evaluation-v2', protocol_sha256=settings['protocol_sha256'],
        evaluator_sha256=file_sha(__file__), plan_sha256=file_sha(root/settings['plan_path']),
        checkpoints=[], decoding=dict(readouts=list(readouts), primary=next(
            (r['name'] for r in readouts if r.get('temperature') is not None), 'greedy'),
            token_budget_factor=settings.get('token_budget_factor', 2.), watchdog_wall_s=settings['checkpoint_wall_s'],
            ignored_v1_settings={k: settings[k] for k in ('per_case_s', 'max_tokens') if k in settings}),
        unmeasured=['other controls', 'continuation/infill', 'whole-song structure', 'native play/edit'],
        scope='pre-training checkpoint validation; RNG restored; no optimizer updates', meanings=MEANINGS)
    if timing is not None:
        result['charter_timing_input'] = timing.identity()
    total_deadline = time.monotonic()+float(settings['max_wall_s'])
    log = lambda value: print(json.dumps(value), flush=True)
    log(dict(phase='learning_eval_start', checkpoints=len(checkpoints)))
    devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        try:
            model.eval()
            for checkpoint in checkpoints:
                if time.monotonic() >= total_deadline:
                    break
                if file_sha(root/checkpoint['path']) != checkpoint['sha256']:
                    raise ValueError('Evaluation checkpoint identity changed')
                saved = torch.load(root/checkpoint['path'], map_location='cpu', weights_only=False, mmap=True)
                if saved['config'] != asdict(model.c) or saved['updates'] != checkpoint['exposure_step']:
                    raise ValueError('Evaluation checkpoint model or step mismatch')
                vocab = getattr(model, 'vocab', None)
                if 'vocabulary' in saved and vocab is not None and saved['vocabulary'] != asdict(vocab):
                    raise ValueError('Evaluation checkpoint vocabulary differs from the model vocabulary')
                model.load_state_dict(saved['model'], strict=True)
                del saved
                records, benchmark = evaluate_model(model, panel, protocol, device=device,
                    deadline=min(total_deadline, time.monotonic()+float(settings['checkpoint_wall_s'])),
                    readouts=readouts, token_budget_factor=settings.get('token_budget_factor', 2.),
                    max_batch_rows=settings.get('max_batch_rows', 32), log=log, **timing_kw)
                result['checkpoints'].append(sanitize(dict(**checkpoint,
                    optimizer_updates=checkpoint['exposure_step']//plan['accum_steps'],
                    records=records, summary=summarize(records, len(protocol['samples'])), benchmark=benchmark)))
                write_atomic(output, result)
        finally:
            saved = torch.load(root/current['path'], map_location='cpu', weights_only=False, mmap=True)
            model.load_state_dict(saved['model'], strict=True)
            model.train()
    result['all_completed'] = (len(result['checkpoints']) == len(checkpoints) and all(
        c['benchmark']['complete_rows'] == c['benchmark']['expected_rows'] for c in result['checkpoints']))
    write_atomic(output, result)
    log(dict(phase='learning_eval_done', all_completed=result['all_completed'],
             results=[dict(optimizer_updates=c['optimizer_updates'], summary=c['summary']) for c in result['checkpoints']]))
    return dict(path=str(output), all_completed=result['all_completed'], checkpoints=len(result['checkpoints']))
