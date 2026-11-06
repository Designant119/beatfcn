"""Opt-in log-odds interaction for verified same-prefix real chart pairs.

The caller certifies identical visible context and two distinct legal targets.
Rows are conditions A/B and columns their real next targets a/b. This objective
does not guarantee either row's absolute argmax; ordinary CE is still required.
"""
import torch
from torch.nn import functional as F


def condition_interaction_loss(scores):
    if scores.ndim < 2 or scores.shape[-2:] != (2, 2) or not scores.is_floating_point():
        raise ValueError('Expected floating condition-by-target 2x2 score matrices')
    if scores.numel() == 0:
        raise ValueError('At least one verified real pair is required')
    # Preserve double precision in algebra tests; avoid low-precision subtraction.
    values = scores if scores.dtype == torch.float64 else scores.float()
    interaction = values[..., 0, 0] - values[..., 0, 1] - values[..., 1, 0] + values[..., 1, 1]
    return F.softplus(1. - interaction).mean(), interaction
