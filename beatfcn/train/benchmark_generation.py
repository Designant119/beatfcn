"""Generate one benchmark request with optional visible reference and density curve."""
import numpy as np
import torch

from beatfcn.events.canonical import Event, LaneLoc, LineLoc
from beatfcn.model.full_event_codec import EventRequest, encode_events, teacher_sequence
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.train.density_curve import grid_density_curve
from beatfcn.train.event_reference import VisibleReference
from beatfcn.train.local_panel import collate
from beatfcn.train.local_request_panel import request_inputs


def canonical(events, mode, role='reference'):
    return tuple(Event(f'{role}:{index}', event.onset_us, event.end_us, event.type,
        LaneLoc(event.lane) if mode == 'mania_4k' else LineLoc('ref', 'above', 0.))
        for index, event in enumerate(events))


def source(audio, mode, label, regions, events):
    # Only human_input/model_history/human_alternative are legal provenance roles.
    # The label keeps the two visible regions as distinct sources.
    return VisibleReference(source_id=f'{label}:{audio[:12]}', mode=mode,
        audio_sha256=audio, role='human_input', regions=tuple(regions),
        events=canonical(events, mode, label))


def generate_request(model, features, audio, request, *, device, curve_segments=(),
                     controls=None, sources=(), max_tokens=512):
    if controls is None:  # unconditional, exactly as a fully dropped training sample
        from beatfcn.train.counterfactual_exposure import unconditional_inputs
        controls = unconditional_inputs(request.mode, request)['controls']
    controls = np.asarray(controls, np.float32)
    inputs, _ = request_inputs(features, audio, request, controls, sources=sources)
    inputs.update(grid_density_curve(list(curve_segments), request.start_us,
        request.end_us, cell_us=100_000, scale=16.))
    sequence = teacher_sequence(encode_events([], request), request)
    batch_inputs, _ = collate([dict(inputs=inputs, sequence=sequence)])
    batch_inputs = {key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in batch_inputs.items()}
    return generate_batch(model, batch_inputs, [request], max_tokens=max_tokens)[0]
