"""Shared candidate input preparation and memory-bounded auxiliary gradients."""
import torch

from beatfcn.train.condition_interaction import condition_interaction_loss
from beatfcn.train.local_shapes import pad_input_shapes


def prepare_inputs(inputs, normalizer, shapes):
    # A single CPU transform used for training, validation and free generation.
    return pad_input_shapes(normalizer(inputs), shapes)


def padded_common_prefix(sequence, fork, length=384):
    if fork < 0 or fork >= length or not bool(sequence['valid'][0, fork]):
        raise ValueError('Invalid common prefix position or shape bucket')
    result = {}
    for key, value in sequence.items():
        out = value.new_zeros((len(value), length, *value.shape[2:]))
        out[:, :fork+1] = value[:, :fork+1]
        result[key] = out
    result['allowed'][..., 0] |= ~result['valid']
    return result


def backward_interaction(row_logits, *, verify=False):
    """Exact first-order chain rule, retaining only one model graph at a time.

    row_logits(i) must be deterministic at unchanged parameters: dropout and
    augmentations are disabled by the experiment contract. The small detached
    2x2 scores determine dL/dz; recomputation propagates those same coefficients.
    No optimizer update is allowed between either pass or either row.
    """
    # Keep the same autograd-enabled execution path as recomputation. Inference
    # kernels may round differently; detach each row immediately to release its
    # model graph before evaluating the next row.
    with torch.enable_grad():
        scores = torch.stack([row_logits(i).detach() for i in (0, 1)])
    scores = scores.detach().requires_grad_(True)
    loss, gap = condition_interaction_loss(scores)
    coefficients = torch.autograd.grad(loss, scores)[0]
    errors = []
    for index in (0, 1):
        values = row_logits(index)
        if verify:
            errors.append(float((values.detach()-scores[index].detach()).abs().max()))
        (values * coefficients[index].detach()).sum().backward()
    if verify and max(errors) > 1e-5:
        raise ValueError('Auxiliary recomputation changed logits; exact chain rule is invalid')
    return loss.detach(), gap.detach(), errors
