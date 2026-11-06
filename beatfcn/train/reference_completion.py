"""Isolated human-chart masking tasks over the existing full-event interface.

Random removal provides positive notes only: their absence does not mean known
silence. Hidden notes and removal indices never enter input construction.
"""
from collections import Counter
from dataclasses import asdict, replace
import hashlib

import numpy as np

from beatfcn.events.canonical import TYPES
from beatfcn.model.full_event_codec import encode_events, teacher_sequence
from beatfcn.train.event_conditions import encode_controls
from beatfcn.train.event_reference import TIME_SCALE_US, VisibleReference, encode_reference
from beatfcn.train.local_request_panel import event_request, request_inputs
from beatfcn.model.local_song_rollout import canonical

TASKS = ('middle', 'random_groups', 'continuation')


def partition(events, request, task, identity, seed):
    """Return a target request and a lossless, whole-onset-group partition."""
    if task not in TASKS:
        raise ValueError(task)
    events = sorted(events)
    start, end = request.start_us, request.end_us
    if task == 'random_groups':
        onsets = sorted({e.onset_us for e in events})
        digest = hashlib.sha256(f'{seed}:{identity}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
        chosen = rng.permutation(len(onsets))[:(len(onsets) + 1) // 2]
        hidden_onsets = {onsets[int(i)] for i in chosen}
        hidden = [e for e in events if e.onset_us in hidden_onsets]
        visible = [e for e in events if e.onset_us not in hidden_onsets]
        regions = ()  # No complete known-silence regions in random deletion.
    else:
        width = end - start
        gap_start = start + (width // 4 if task == 'middle' else width // 2)
        gap_end = end - width // 4 if task == 'middle' else end
        # The inherited codec requires integer milliseconds, as do source targets.
        gap_start = (gap_start // 1000) * 1000
        gap_end = (gap_end // 1000) * 1000
        hidden = [e for e in events if gap_start <= e.onset_us < gap_end]
        visible = [e for e in events if not gap_start <= e.onset_us < gap_end]
        incoming = tuple(e for e in [*request.known_carry, *visible]
                         if e.type == 'hold' and e.onset_us < gap_start < e.end_us)
        request = replace(request, start_us=gap_start, end_us=gap_end, known_carry=incoming)
        regions = tuple((a, b) for a, b in ((start, gap_start), (gap_end, end)) if a < b)
    signature = lambda e: (e.onset_us, e.end_us, e.type, e.lane)
    if Counter(map(signature, events)) != Counter(map(signature, visible + hidden)):
        raise AssertionError('Masking lost or invented events')
    if {e.onset_us for e in visible} & {e.onset_us for e in hidden}:
        raise AssertionError('An onset group was partially exposed')
    return request, visible, hidden, regions


def sparse_reference(visible, request):
    """Encode only explicitly supplied events, with no access to hidden labels."""
    values = np.zeros((1 + len(visible), 16), np.float32)
    times = np.zeros(1 + len(visible), np.float32)
    values[0, 1] = (request.end_us - request.start_us) / TIME_SCALE_US
    values[0, 15] = 1
    for i, event in enumerate(visible, 1):
        values[i, :2] = [(event.onset_us - request.start_us) / TIME_SCALE_US,
                         (event.end_us - request.start_us) / TIME_SCALE_US]
        values[i, 2 + TYPES.index(event.type)] = 1
        values[i, 10:12] = 1  # event and explicitly authorized human_input
        times[i] = (event.onset_us - request.start_us) / 1e6
    return dict(reference=values, reference_times=times,
                reference_valid=np.ones(len(values), bool))


def completion_row(panel, index, task, *, reference, seed):
    row = panel.records[index]
    if row['mode'] != 'phira':
        raise ValueError('This masking contract is Phira only')
    original = panel.targets(index)
    request, visible, hidden, regions = partition(
        original, event_request(row['request']), task,
        f"{row['chart_id']}:{row['request']['start_us']}", seed)
    controls = encode_controls('phira', {}, (request.start_us, request.end_us), origin_us=request.start_us)
    sources = ()
    if reference and regions:
        sources = (VisibleReference(source_id='completion-visible:' + row['chart_id'], mode='phira',
                    audio_sha256=row['audio_sha256'], role='human_input', regions=regions,
                    events=tuple(canonical(visible, 'phira'))),)
    inputs, provenance = request_inputs(panel.features, row['audio_sha256'], request, controls, sources=sources)
    if reference and not regions:
        inputs.update(sparse_reference(visible, request))
        provenance = [dict(role='human_input_positive_only', event=asdict(e)) for e in visible]
    sequence = teacher_sequence(encode_events(hidden, request), request)
    metadata = dict(index=index, chart_id=row['chart_id'], audio_sha256=row['audio_sha256'],
                    component=row['component'], split=row['split'], stratum=row['stratum'], task=task,
                    request=asdict(request), visible=[asdict(e) for e in visible],
                    hidden=[asdict(e) for e in hidden], original=[asdict(e) for e in original],
                    provenance=provenance, reference=reference)
    return dict(inputs=inputs, sequence=sequence), request, metadata
