"""An opt-in likelihood contrast against a contradictory request condition.

A negative condition is not a second positive human chart. CPU preparation
must establish actual contradiction and feasible supported request values.
"""
import math

import numpy as np
import torch
from torch.nn import functional as F

from beatfcn.events.controls import BAND_EDGES, MODE_FIELDS
from beatfcn.model.full_event_model import event_loss
from beatfcn.train.event_conditions import CONTROL_FIELDS, encode_controls, in_band, request_log_probability


def validated_countercondition(mode, field, actual, positive, negative, scope, correct):
    """Validate a CPU single-field training view; preserve all unrelated bytes."""
    if field not in MODE_FIELDS.get(mode, ()):
        raise ValueError('Inapplicable field')
    if not in_band(field, actual, positive) or in_band(field, actual, negative):
        raise ValueError('Positive must contain the real label and negative must contradict it')
    correct = np.asarray(correct, np.float32)
    expected = encode_controls(mode, {field: positive}, scope, origin_us=scope[0])
    if not np.array_equal(correct, expected):
        raise ValueError('Expected exactly one specified true condition at the same scope')
    opposing = encode_controls(mode, {field: negative}, scope, origin_us=scope[0])
    i = CONTROL_FIELDS.index(field)
    if not np.array_equal(correct[:, 2:], opposing[:, 2:]):
        raise ValueError('Condition metadata changed')
    if not np.array_equal(np.delete(correct, i, 0), np.delete(opposing, i, 0)):
        raise ValueError('A second field changed')
    return opposing


def countercondition_objective(model, inputs, sequence, *, opposing_controls=None, weight=0., margin=.1):
    if not math.isfinite(weight) or weight < 0 or not math.isfinite(margin) or margin < 0:
        raise ValueError('Finite nonnegative objective parameters required')
    memory = model.encode(inputs)
    keys = ('tokens', 'fields', 'times', 'valid')
    correct = model.decode(*(sequence[k] for k in keys), memory)
    ce = event_loss(correct, sequence)
    if weight == 0 or opposing_controls is None:
        return ce, dict(event_ce=ce.detach(), contrast=None, gap=None)
    if opposing_controls.shape != inputs['controls'].shape:
        raise ValueError('Countercondition shape differs from the CPU-validated view')
    altered = model.replace_controls(memory, opposing_controls)
    opposing = model.decode(*(sequence[k] for k in keys), altered)
    gap = request_log_probability(correct, sequence) - request_log_probability(opposing, sequence)
    rank = F.softplus(margin-gap).mean()
    return ce + weight*rank, dict(event_ce=ce.detach(), contrast=rank.detach(), gap=gap.detach())
