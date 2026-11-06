"""Temperature-one ancestral readout for the existing event-package model.

Independent of its frozen greedy training/generation implementation. Counts,
each member onset and legal joint hold tails are sampled from model laws. No
count/type quota, threshold tuning or duplicate removal is applied.
"""
import time

import numpy as np
import torch

from .canvas_diffusion import _sample_tail
from .event_canvas import RhythmEvent
from .local_event_package import GRU_WIDTH, _allowed_kinds, _single_request
from .local_typed_canvas import CELL_US, KINDS


def categorical(probabilities, rng):
    probability = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(probability).all() or (probability < 0).any() or probability.sum() <= 0:
        raise ValueError('Invalid categorical probability')
    return int(rng.choice(len(probability), p=probability / probability.sum()))


@torch.no_grad()
def sample_package(model, inputs, request, *, seed, output=None,
                   frame_permutation=None, deadline=None):
    _single_request(inputs, [request])
    model.eval()
    if output is None:
        output = model(inputs)
    rng = np.random.default_rng(seed)
    valid = inputs['canvas_valid'][0].cpu().numpy()
    frames = np.flatnonzero(valid)
    permutation = np.arange(len(valid)) if frame_permutation is None else np.asarray(frame_permutation)
    if (permutation.shape != (len(valid),) or not np.array_equal(np.sort(permutation), np.arange(len(valid)))
            or not np.array_equal(np.sort(permutation[frames]), frames)):
        raise ValueError('Frame permutation must preserve the valid address support')
    probability = output['count'][0].float().cpu().numpy()
    heads = inputs['canvas_head_us'][0]
    events = []
    for frame in frames:
        source = int(permutation[frame])
        base_us = request.start_us + int(frame) * CELL_US
        allowed = inputs['canvas_head_allowed'][0, frame]
        for kind in _allowed_kinds(request):
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError('Ancestral package readout deadline')
            count = categorical(probability[source, kind], rng)
            if len(events) + count > request.max_events:
                raise ValueError('Ancestral package event resource overflow')
            if count == 0:
                continue
            if not bool(allowed.any()):
                raise ValueError('No legal event onset address')
            static = (output['x'][0, source] + model.kind_ids.weight[kind]
                      + model.count_ids.weight[count])
            previous = model.member_start
            hidden = static.new_zeros(GRU_WIDTH)
            for _ in range(count):
                hidden = model.member_gru(torch.cat((static, previous), -1)[None], hidden[None])[0]
                logits = model.onset_read(hidden).masked_fill(~allowed, -1e9)
                offset = categorical(logits.float().cpu().double().softmax(-1).numpy(), rng)
                head = int(heads[frame, offset])
                onset_embedding = model.onset_ids.weight[offset]
                end = head
                if kind == KINDS.index('hold'):
                    condition = model.tail_condition(torch.cat((hidden, onset_embedding), -1))
                    fields = (model.tail_block(condition), model.tail_cell(condition), model.tail_ms(condition))
                    lo = (head - base_us) // 1000 + 1
                    hi = (request.duration_us - base_us) // 1000
                    if hi < lo:
                        raise ValueError('No legal native-ms hold tail')
                    delta = _sample_tail(*(f.float().cpu().double().numpy() for f in fields), lo, hi, rng)
                    end = base_us + delta * 1000
                    block, rest = divmod(delta, 2000)
                    cell, ms = divmod(rest, 40)
                    tail_embedding = (model.tail_block_ids.weight[block]
                                      + model.tail_cell_ids.weight[cell] + model.tail_ms_ids.weight[ms])
                else:
                    tail_embedding = model.no_tail
                previous = model.member_read(torch.cat((onset_embedding, tail_embedding), -1))
                events.append(RhythmEvent(head, end, KINDS[kind]))
    return sorted(events)
