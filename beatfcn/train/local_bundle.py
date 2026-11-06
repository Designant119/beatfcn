"""Small-data views and legal, fixed-clock generated-history recovery."""
from collections import Counter
from dataclasses import asdict, replace
import math

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventState, encode_events, teacher_sequence
from beatfcn.model.local_song_rollout import canonical
from beatfcn.train.event_conditions import encode_controls, training_labels
from beatfcn.train.event_reference import VisibleReference
from beatfcn.train.local_panel import collate
from beatfcn.train.local_request_panel import event_request, request_inputs
from beatfcn.train.local_shapes import pad_input_shapes
from beatfcn.train.reference_completion import partition, sparse_reference


def annotated_sequence(events, request):
    tokens = encode_events(events, request)
    sequence = teacher_sequence(tokens, request)
    counts = Counter(e.onset_us for e in events)
    count_valid = np.zeros(len(tokens), bool)
    group_log_count = np.zeros(len(tokens), np.float32)
    state = EventState(request)
    for i, token in enumerate(tokens):
        if state.phase == 'member' and state.group_members == 0:
            count_valid[i] = True
            group_log_count[i] = math.log1p(counts[state.onset])
        state.apply(token)
    return {**sequence, 'count_valid': count_valid, 'group_log_count': group_log_count}


def make_row(panel, index, view='draft', *, seed=20261027, field_index=0,
             generated_history=None, manual_bands=None):
    record = panel.records[index]
    original = panel.targets(index)
    original_request = event_request(record['request'])
    request = original_request
    visible, hidden, regions = [], original, ()
    if view in ('middle', 'random_groups', 'continuation'):
        request, visible, hidden, regions = partition(original, request, view,
            f"{record['chart_id']}:{request.start_us}", seed)
    source_role = 'human_input'
    if generated_history is not None:
        if view != 'continuation':
            raise ValueError('Generated histories only condition fixed-clock continuation')
        visible = [RhythmEvent(**e) for e in generated_history]
        if any(not original_request.start_us <= e.onset_us < request.start_us for e in visible):
            raise ValueError('Generated prefix crosses the fixed continuation boundary')
        request = replace(request, known_carry=tuple(e for e in visible
            if e.type == 'hold' and e.onset_us < request.start_us < e.end_us))
        source_role = 'model_history'
    bands = {} if manual_bands is None else manual_bands
    if view in ('single_condition', 'all_conditions'):
        if record['split'] != 'train':
            raise ValueError('Hidden validation labels cannot construct generation conditions')
        bands, _ = training_labels('phira', canonical(original + list(original_request.known_carry), 'phira'),
                                   (request.start_us, request.end_us))
        if view == 'single_condition' and bands:
            field = sorted(bands)[field_index % len(bands)]
            bands = {field: bands[field]}
    controls = encode_controls('phira', bands, (request.start_us, request.end_us), origin_us=request.start_us)
    sources = ()
    if visible and regions:
        sources = (VisibleReference(source_id=f'local-bundle:{source_role}:{index}', mode='phira',
            audio_sha256=record['audio_sha256'], role=source_role, regions=regions,
            events=tuple(canonical(visible, 'phira'))),)
    inputs, provenance = request_inputs(panel.features, record['audio_sha256'], request, controls, sources=sources)
    if visible and not regions:
        inputs.update(sparse_reference(visible, request))
    return dict(inputs=inputs, sequence=annotated_sequence(hidden, request), request=request,
                index=index, view=view, origin_s=request.start_us / 1e6,
                targets=hidden, visible=visible, provenance=provenance, source_role=source_role)


def batch_rows(rows, device='cpu'):
    inputs, sequence = collate(rows)
    inputs = pad_input_shapes(inputs, dict(local=400, mel=1600, song=1024,
        reference=max(64, ((inputs['reference'].shape[1] + 63) // 64) * 64)))
    inputs['origin_s'] = torch.tensor([row['origin_s'] for row in rows], dtype=torch.float32)
    inputs['end_s'] = torch.tensor([row['request'].end_us / 1e6 for row in rows], dtype=torch.float32)
    inputs['duration_s'] = torch.tensor([row['request'].duration_us / 1e6 for row in rows], dtype=torch.float32)
    return ({k: v.to(device) for k, v in inputs.items()}, {k: v.to(device) for k, v in sequence.items()})


def prefix_row(panel, index):
    original = event_request(panel.records[index]['request'])
    mid = ((original.start_us + original.end_us) // 2000) * 1000
    request = replace(original, end_us=mid, known_carry=())
    controls = encode_controls('phira', {}, (request.start_us, request.end_us), origin_us=request.start_us)
    inputs, _ = request_inputs(panel.features, panel.records[index]['audio_sha256'], request, controls)
    return dict(inputs=inputs, sequence=annotated_sequence([], request), request=request,
                origin_s=request.start_us / 1e6)


def tensor_audit(inputs, sequence):
    report = {}
    for section, values in (('inputs', inputs), ('sequence', sequence)):
        report[section] = {}
        for name, value in values.items():
            numeric = value.float()
            finite = bool(torch.isfinite(numeric).all())
            if not finite:
                raise ValueError(f'Non-finite real tensor {section}.{name}')
            report[section][name] = dict(shape=list(value.shape), finite=finite,
                min=float(numeric.min()), max=float(numeric.max()), mean=float(numeric.mean()))
    return report
