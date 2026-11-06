"""Whole-song rollout over complete audio, never reading a human target."""
from dataclasses import asdict
from functools import partial
import heapq
import time

import torch

from beatfcn.events.canonical import Event, LaneLoc, LineLoc, lane_conflicts
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest, encode_events, teacher_sequence
from beatfcn.model.full_event_generate import ContinuousDecoder, generate_batch
from beatfcn.train.event_conditions import encode_controls
from beatfcn.train.event_reference import VisibleReference
from beatfcn.train.local_panel import collate
from beatfcn.train.local_request_panel import request_inputs
from beatfcn.train.local_shapes import pad_input_shapes


def canonical(events, mode):
    return [Event(str(i), e.onset_us, e.end_us, e.type,
                  LaneLoc(e.lane) if mode == 'mania_4k' else LineLoc('rhythm', 'above', 0.))
            for i, e in enumerate(events)]


def independent_legality(events, mode, duration_us):
    issues = []
    for i, e in enumerate(events):
        if not 0 <= e.onset_us < duration_us or not e.onset_us <= e.end_us <= duration_us:
            issues.append(dict(code='audio_bounds', index=i))
        if e.type not in (('tap', 'hold') if mode == 'mania_4k' else ('tap', 'hold', 'flick', 'drag')):
            issues.append(dict(code='mode_type', index=i))
        if (e.type == 'hold' and e.end_us <= e.onset_us) or (e.type != 'hold' and e.end_us != e.onset_us):
            issues.append(dict(code='duration', index=i))
    if mode == 'mania_4k':
        issues += [dict(code=code, first=a, second=b) for code, a, b in lane_conflicts(canonical(events, mode))]
    return issues


SCHEDULERS = ('legacy', 'grouped', 'continuous')
# Grouped: rows are sorted within the first 2 x batch_size live rows, so at most that many rows of
# the earliest songs are in flight and a partial run still yields complete consensus for early songs.
GROUP_WINDOW_BATCHES = 2
FIRST_CORE_TOKENS = 270  # median tokens per core, the sort key of a core with no previous core


def _check_rollout_request(model, songs, batch_size):
    if model.training or not songs or batch_size < 1:
        raise ValueError('Whole-song generation requires eval mode and a nonempty song panel')
    allowed = {'chart_id', 'audio_sha256', 'mode', 'duration_us'}
    if any(set(s) != allowed for s in songs):
        raise ValueError('Rollout descriptors cannot include targets, labels, or hidden references')
    if len({s['audio_sha256'] for s in songs}) != len(songs):
        raise ValueError('Duplicate evaluation audio')


def _initial_states(songs, history_reference):
    return [dict(**song, events=[], cores=[], next_start_us=0, complete=False, failure=None,
                 history_reference=history_reference) for song in songs]


@torch.no_grad()
def generate_complete_songs(model, features, songs, *, device, batch_size=4, token_cap=4096,
                            max_wall_s=1200, history_reference=False, on_progress=None,
                            generator=generate_batch, input_shapes=None, intensity_provider=None,
                            on_core_inputs=None, condition_provider=None, timing_provider=None,
                            alternative_provider=None, completed=None):
    """Roll 8 s cores over each whole song.

    ``completed`` (optional): {chart_id: state} finished by an earlier run; kept, not decoded.

    ``alternative_provider`` (optional, default None = historical behaviour) is
    called as ``alternative_provider(state, request)`` per core and returns None
    or an event_reference.VisibleReference with role ``human_alternative``: the
    charter's OWN other chart of this song (difficulty spread / downmapping),
    shown around the core (it may overlap the core; it is not a hidden answer).
    It is the explicit charter-chart channel, labelled by the caller; each core
    records the source id and shown event count under ``alternative``, and
    ``reference_provenance`` is summarized (rows / sources) on that path.

    ``timing_provider`` (optional, default None = historical behaviour) is
    called as ``timing_provider(state, request)`` per core and returns None
    (absent timing) or the chart timing segments [(start_us, beat_us, meter),
    ...] on the model clock; every row then carries ``beat`` features for its
    mel frames (beatfcn.train.charter_timing, all-zero when absent), also in
    ``unconditional_inputs`` (timing is an input, not a guided condition). It is
    the explicit charter-timing channel, labelled by the caller
    (``charter_timing_input``); each core records ``charter_timing`` presence.

    ``condition_provider`` (optional, default None = historical behaviour) is
    called as ``condition_provider(state, request)`` per core and returns None
    (core keeps the unspecified conditions) or ``dict(inputs=update[, record=...])``:
    ``update`` overrides model input keys (e.g. ``controls`` / ``intensity*``
    from counterfactual_exposure.condition_inputs) after ``intensity_provider``;
    ``record`` is stored as that core's ``condition`` entry. When any row of a
    batch is conditioned, ``generator`` also receives ``unconditional_inputs``:
    the same batch WITHOUT the provider updates (the rollout's own unspecified
    conditions), for classifier-free guidance. The provider is the explicit
    charter-intent channel: it may read what the caller chose to specify
    (e.g. a requested density), and it is labelled by the caller.

    ``on_core_inputs`` (optional, default None = historical behaviour) is called
    as ``on_core_inputs(model, inputs, requests, states)`` with the exact
    device inputs passed to ``generator`` for one batch. It may return None or
    one dict per row; each dict is stored as that core's ``probe`` record
    (e.g. onset-head probabilities for a reference-free reranker). It must not
    mutate ``inputs``; it never sees a human target (states hold only the
    descriptor and generated events).
    """
    _check_rollout_request(model, songs, batch_size)
    states = _initial_states(songs, history_reference)
    for state in states:
        state.update((completed or {}).get(state['chart_id'], {}))
    started = time.monotonic()
    _rollout(model, features, states, None, device=device, batch_size=batch_size, token_cap=token_cap,
             deadline=started+max_wall_s, history_reference=history_reference, on_progress=on_progress,
             generator=generator, input_shapes=input_shapes, intensity_provider=intensity_provider,
             on_core_inputs=on_core_inputs, condition_provider=condition_provider, timing_provider=timing_provider,
             alternative_provider=alternative_provider)
    return dict(complete=all(s['complete'] for s in states), songs=states,
                wall_s=time.monotonic()-started, history_reference=history_reference,
                conditioning='unspecified per-core conditions, no hidden human attributes')


@torch.no_grad()
def generate_complete_song_samples(model, features, songs, seeds, *, device, batch_size=64, token_cap=4096,
                                   max_wall_s=1200, history_reference=False, on_progress=None,
                                   generator=generate_batch, input_shapes=None, intensity_provider=None,
                                   on_core_inputs=None, condition_provider=None, timing_provider=None,
                                   alternative_provider=None, scheduler='legacy', completed=None):
    """``generate_complete_songs`` for several sample seeds in ONE rollout.

    Every (seed, song) pair is an independent row; rows of all seeds share
    batches, so N best-of samples cost one rollout of N x songs rows instead
    of N sequential rollouts. ``generator`` is called with an extra keyword
    ``row_seeds`` (the sample seed of each row) and must make each row's draws
    depend only on its own seed and request (per-row torch.Generators), which
    makes every sample identical to a separate ``generate_complete_songs``
    call with that seed. ``max_wall_s`` bounds the whole call. Returns one
    ``generate_complete_songs``-shaped result per seed (``wall_s`` is shared).

    ``scheduler``: 'legacy' fills batches seed by seed. 'grouped' fills them from a window of the
    earliest songs, seeds of one core together, sorted by the previous core's tokens, sharing the
    encoder across seeds in cold mode. 'continuous' refills seats per step (ContinuousDecoder, built
    from ``generator.continuous``). Non-legacy schedulers finish songs in order, so a partial run
    has complete consensus for the songs it finished. All give the same tokens per
    (seed, song, core) because generators are per-row; only fp32 reduction order may differ.

    ``on_progress`` here is ``on_progress(state, seed)``. ``completed``: {(seed, chart_id): state}
    already finished by an earlier run; they are kept as they are and not decoded.
    """
    _check_rollout_request(model, songs, batch_size)
    if scheduler not in SCHEDULERS:
        raise ValueError(f'Unknown scheduler {scheduler!r}')
    seeds = [int(seed) for seed in seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Sample seeds must be unique and nonempty')
    per_seed = [_initial_states(songs, history_reference) for _ in seeds]
    row_seeds = {id(state): seed for seed, group in zip(seeds, per_seed) for state in group}
    for seed, group in zip(seeds, per_seed):
        for state in group:
            state.update((completed or {}).get((seed, state['chart_id']), {}))
    if scheduler == 'legacy':
        states = [state for group in per_seed for state in group]
    else:  # song by song, seeds together
        states = [group[index] for index in range(len(songs)) for group in per_seed]
    progress = None if on_progress is None else lambda state: on_progress(state, row_seeds[id(state)])
    started = time.monotonic()
    run = _rollout_continuous if scheduler == 'continuous' else partial(_rollout, grouped=scheduler == 'grouped')
    run(model, features, states, row_seeds, device=device, batch_size=batch_size, token_cap=token_cap,
        deadline=started+max_wall_s, history_reference=history_reference, on_progress=progress,
        generator=generator, input_shapes=input_shapes, intensity_provider=intensity_provider,
        on_core_inputs=on_core_inputs, condition_provider=condition_provider, timing_provider=timing_provider,
        alternative_provider=alternative_provider)
    wall = time.monotonic()-started
    return [dict(complete=all(s['complete'] for s in group), songs=group, wall_s=wall,
                 history_reference=history_reference, parallel_samples=len(seeds),
                 conditioning='unspecified per-core conditions, no hidden human attributes')
            for group in per_seed]


def _core_request(state):
    start = state['next_start_us']
    end = min(start+8_000_000, state['duration_us'])
    events = [RhythmEvent(**e) for e in state['events']]
    carry = tuple(e for e in events if e.type == 'hold' and e.onset_us < start < e.end_us)
    return EventRequest(state['mode'], start, end, state['duration_us'], known_carry=carry)


def _build_core_row(state, request, features, *, history_reference, intensity_provider, condition_provider, timing_provider,
              alternative_provider):
    """One core's collator row plus what its core record needs."""
    start, end = request.start_us, request.end_us
    sources = ()
    if history_reference and start:
        lower = max(0, start-8_000_000)
        events = [RhythmEvent(**e) for e in state['events']]
        sources = (VisibleReference(source_id='generated:'+state['chart_id'], mode=state['mode'],
            audio_sha256=state['audio_sha256'], role='model_history', regions=((lower, start),),
            events=tuple(canonical([e for e in events if lower <= e.onset_us < start], state['mode']))),)
    built = dict(curve=None, condition=None, timing=None, bare=None)
    if alternative_provider is not None:
        alternative = alternative_provider(state, request)
        if alternative is not None:
            if alternative.role != 'human_alternative':
                raise ValueError('alternative_provider must return a human_alternative reference')
            sources = tuple(sources) + (alternative,)
        built['alternative'] = (None if alternative is None else
                                dict(source_id=alternative.source_id, events=len(alternative.events)))
    controls = encode_controls(state['mode'], {}, (start, end), origin_us=start)
    inputs, source = request_inputs(features, state['audio_sha256'], request, controls, sources=sources)
    if alternative_provider is not None:
        source = dict(rows=len(source), sources=sorted({r['source_id'] for r in source if r['source_id']}))
    if intensity_provider is not None:
        built['curve'] = intensity_provider(state['audio_sha256'], start, end)
        inputs.update(built['curve'])
    if timing_provider is not None:
        from beatfcn.train.charter_timing import window_beat_features
        segments = timing_provider(state, request)
        inputs['beat'] = window_beat_features(inputs, start, segments)
        built['timing'] = dict(present=segments is not None, segments=0 if segments is None else len(segments))
    # Dummy sequence is only a collator adapter and is never passed to generation.
    sequence = teacher_sequence(encode_events([], request), request)
    built['condition'] = condition_provider(state, request) if condition_provider is not None else None
    if condition_provider is not None:
        built['bare'] = dict(inputs=dict(inputs), sequence=sequence)
        if built['condition'] is not None:
            inputs.update(built['condition']['inputs'])
    built.update(row=dict(inputs=inputs, sequence=sequence), source=source)
    return built


def _device_rows(rows, input_shapes, device):
    inputs, _ = collate(rows)
    if input_shapes:
        inputs = pad_input_shapes(inputs, input_shapes)
    return {k: v.to(device) for k, v in inputs.items()}


def _record_core(state, request, prediction, built, probe, wall_s, on_progress):
    """Book one decoded core into its state (identical for every scheduler)."""
    additions = [RhythmEvent(**e) for e in prediction['events']]
    if any(not request.start_us <= e.onset_us < request.end_us for e in additions):
        raise ValueError('Generated event escapes its core')
    state['events'].extend(prediction['events'])
    core = dict(start_us=request.start_us, end_us=request.end_us,
        known_carry=[asdict(e) for e in request.known_carry], reference_provenance=built['source'],
        tokens=prediction['tokens'], event_count=len(additions), complete=prediction['complete'],
        deadline_reached=prediction.get('deadline_reached', False),
        token_limit_reached=prediction.get('token_limit_reached', False),
        batch_wall_s=wall_s)
    if built['curve'] is not None:
        core.update(requested_curve=built['curve']['intensity'].tolist(),
            requested_curve_times=built['curve']['intensity_times'].tolist(),
            requested_curve_valid=built['curve']['intensity_valid'].tolist(),
            curve_start_us=request.start_us)
    if probe is not None:
        core['probe'] = probe
    if built['condition'] is not None and built['condition'].get('record') is not None:
        core['condition'] = built['condition']['record']
    if built['timing'] is not None:
        core['charter_timing'] = built['timing']
    if 'alternative' in built:
        core['alternative'] = built['alternative']
    state['cores'].append(core)
    if not prediction['complete']:
        state['failure'] = 'generation_deadline' if prediction.get('deadline_reached') else 'core_token_limit'
    else:
        state['next_start_us'] = request.end_us
        state['complete'] = request.end_us == state['duration_us']
    state['legality'] = independent_legality([RhythmEvent(**e) for e in state['events']],
                                             state['mode'], state['duration_us'])
    if state['legality']:
        state['complete'] = False
        state['failure'] = 'independent_legality'
    if on_progress:
        on_progress(state)


def _finish_states(states, on_progress):
    for state in states:
        state.setdefault('legality', [])
        if on_progress:
            on_progress(state)


def _is_cold(history_reference, condition_provider, timing_provider, alternative_provider):
    """Cold rows depend only on (audio, core), so all seeds of a core can share one encoder row."""
    return not history_reference and condition_provider is None and timing_provider is None \
        and alternative_provider is None


def _row_key(state, request, cold):
    return (state['audio_sha256'], request.start_us, request.end_us) if cold else id(state)


def _grouped_batch(active, batch_size, previous_tokens):
    """Rows of the earliest songs: seeds of one core together, cores with few previous tokens first.

    Exactly ``batch_size`` rows whenever that many are live (a stable CUDA-graph geometry)."""
    groups = {}
    for state in active[:GROUP_WINDOW_BATCHES*batch_size]:
        groups.setdefault((state['audio_sha256'], state['next_start_us']), []).append(state)
    ordered = sorted(groups.values(), key=lambda group: max(previous_tokens(state) for state in group))
    return [state for group in ordered for state in group][:batch_size]


def _rollout(model, features, states, row_seeds, *, device, batch_size, token_cap, deadline, history_reference,
             on_progress, generator, input_shapes, intensity_provider, on_core_inputs, condition_provider=None,
             timing_provider=None, alternative_provider=None, grouped=False):
    cold = _is_cold(history_reference, condition_provider, timing_provider, alternative_provider)
    row_options = dict(history_reference=history_reference, intensity_provider=intensity_provider,
                       condition_provider=condition_provider, timing_provider=timing_provider,
                       alternative_provider=alternative_provider)
    previous_tokens = lambda state: len(state['cores'][-1]['tokens']) if state['cores'] else FIRST_CORE_TOKENS  # noqa: E731
    while True:
        active = [s for s in states if not s['complete'] and s['failure'] is None]
        if not active:
            break
        if time.monotonic() >= deadline:
            for s in active:
                s['failure'] = 'evaluation_wall_limit'
            break
        batch = _grouped_batch(active, batch_size, previous_tokens) if grouped else active[:batch_size]
        requests = [_core_request(state) for state in batch]
        # Each request row reads ``built[memory_rows[row]]``; the legacy path builds one per row.
        keys, memory_rows = {}, []
        built = []
        for state, request in zip(batch, requests):
            key = _row_key(state, request, cold) if grouped else len(built)
            if key not in keys:
                keys[key] = len(built)
                built.append(dict(_build_core_row(state, request, features, **row_options), state=state, request=request))
            memory_rows.append(keys[key])
        inputs = _device_rows([b['row'] for b in built], input_shapes, device)
        extra = {}
        if any(b['condition'] is not None for b in built):
            extra['unconditional_inputs'] = _device_rows([b['bare'] for b in built], input_shapes, device)
        if grouped:
            extra['memory_index'] = torch.tensor(memory_rows, device=device)
        probes = None
        if on_core_inputs is not None:
            probes = on_core_inputs(model, inputs, [b['request'] for b in built], [b['state'] for b in built])
            if probes is not None and len(probes) != len(built):
                raise ValueError('on_core_inputs returned a different batch size')
        tick = time.monotonic()
        if row_seeds is None:
            predictions = generator(model, inputs, requests, token_cap, deadline=deadline, **extra)
        else:
            predictions = generator(model, inputs, requests, token_cap, deadline=deadline,
                                    row_seeds=[row_seeds[id(state)] for state in batch], **extra)
        if len(predictions) != len(batch):
            raise ValueError('Generator returned a different batch size')
        for state, request, prediction, row in zip(batch, requests, predictions, memory_rows):
            _record_core(state, request, prediction, built[row], None if probes is None else probes[row],
                         time.monotonic()-tick, on_progress)
    _finish_states(states, on_progress)


def _rollout_continuous(model, features, states, row_seeds, *, device, batch_size, token_cap, deadline,
                        history_reference, on_progress, generator, input_shapes, intensity_provider, on_core_inputs,
                        condition_provider=None, timing_provider=None, alternative_provider=None):
    """One seat table over the whole panel. A pair's next core is requested only after its previous
    core ended, so the carry and history_reference of every core are exact; earlier songs always get
    free seats first, so songs finish roughly in order."""
    spec = getattr(generator, 'continuous', None)
    if spec is None:
        raise ValueError("scheduler='continuous' needs a make_generator generator (generator.continuous)")
    guided = spec.get('guidance') is not None and condition_provider is not None
    spec = dict(spec, guidance=spec.get('guidance') if guided else None)
    cold = _is_cold(history_reference, condition_provider, timing_provider, alternative_provider)
    row_options = dict(history_reference=history_reference, intensity_provider=intensity_provider,
                       condition_provider=condition_provider, timing_provider=timing_provider,
                       alternative_provider=alternative_provider)
    ranks = {id(state): rank for rank, state in enumerate(states)}
    by_song = {}
    for state in states:
        by_song.setdefault(state['audio_sha256'], []).append(state)
    waiting = [rank for rank, s in enumerate(states) if not s['complete'] and s['failure'] is None]
    decoder = ContinuousDecoder(model, max(1, min(batch_size, len(waiting))), token_cap, **spec)
    # Seats may wait a few steps so refills encode together; costs about 1% occupancy on saved token counts.
    refill_batch = max(1, decoder.seats//20)
    seated, shared = {}, {}
    while waiting or seated:
        if time.monotonic() >= deadline:
            for row, seat in list(seated.items()):
                _record_core(seat['state'], seat['request'], decoder.release(row, deadline_reached=True), seat['built'],
                             seat['probe'], time.monotonic()-seat['tick'], on_progress)
            for rank in waiting:
                states[rank]['failure'] = 'evaluation_wall_limit'
            break
        free = decoder.free_seats()
        if free and waiting and (len(free) >= min(refill_batch, len(waiting)) or not seated):
            picks = [states[heapq.heappop(waiting)] for _ in range(min(len(free), len(waiting)))]
            requests = [_core_request(state) for state in picks]
            keys = [_row_key(state, request, cold) for state, request in zip(picks, requests)]
            built = {}
            for state, request, key in zip(picks, requests, keys):
                if key not in shared and key not in built:
                    built[key] = dict(_build_core_row(state, request, features, **row_options), state=state, request=request)
            if built:
                new = list(built.values())
                inputs = _device_rows([b['row'] for b in new], input_shapes, device)
                bare = _device_rows([b['bare'] for b in new], input_shapes, device) if guided else None
                probes = (on_core_inputs(model, inputs, [b['request'] for b in new], [b['state'] for b in new])
                          if on_core_inputs is not None else None)
                if probes is not None and len(probes) != len(new):
                    raise ValueError('on_core_inputs returned a different batch size')
                encoded = decoder.encode(inputs, [b['request'] for b in new], bare)
                for row, (key, b) in enumerate(built.items()):
                    del b['row'], b['bare']  # the feature windows are not needed once encoded
                    shared[key] = dict(encoded=encoded[row], built=b, probe=None if probes is None else probes[row])
            generators = ([None]*len(picks) if spec.get('temperature') is None else generator.row_generators(
                [row_seeds[id(state)] for state in picks], requests, decoder.device))
            tick = time.monotonic()
            for row, state, request, key, row_generator in zip(free, picks, requests, keys, generators):
                entry = shared[key]
                decoder.seat(row, request, entry['encoded'], row_generator)
                seated[row] = dict(state=state, request=request, built=entry['built'], probe=entry['probe'], tick=tick)
            # An encoded core stays only while a live pair of that song can still reach it.
            on_seat = {id(seat['state']) for seat in seated.values()}
            shared = {} if not cold else {key: entry for key, entry in shared.items() if any(
                not s['complete'] and s['failure'] is None and (s['next_start_us'] < key[1] or (
                    s['next_start_us'] == key[1] and id(s) not in on_seat)) for s in by_song[key[0]])}
        remaining = len(seated) + len(waiting)
        if decoder.seats >= 16 and remaining <= decoder.seats//2:
            moved = decoder.compact(remaining)
            seated = {moved[row]: seat for row, seat in seated.items()}
            refill_batch = max(1, decoder.seats//20)
        for row in decoder.step():
            seat = seated.pop(row)
            state = seat['state']
            _record_core(state, seat['request'], decoder.release(row), seat['built'], seat['probe'],
                         time.monotonic()-seat['tick'], on_progress)
            if not state['complete'] and state['failure'] is None:
                heapq.heappush(waiting, ranks[id(state)])
    decoder.finish()
    _finish_states(states, on_progress)
