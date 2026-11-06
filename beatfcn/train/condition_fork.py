"""Opt-in weight on a CPU-verified first divergent real paired-chart target."""
import math

import torch
from torch.nn import functional as F

from beatfcn.model.full_event_model import event_loss


def condition_fork_loss(logits, sequence, positions=None, *, weight=0.):
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('Finite nonnegative fork weight required')
    ce = event_loss(logits, sequence)
    if weight == 0.:
        return ce, dict(event_ce=ce.detach(), fork_ce=None)
    if positions is None or positions.shape != logits.shape[:1] or positions.dtype != torch.long:
        raise ValueError('One CPU-validated fork position per request required; -1 means no pair')
    indices = positions.clamp_min(0)[:, None]
    gather = indices[:, :, None].expand(-1, -1, logits.shape[-1])
    selected = logits.gather(1, gather).squeeze(1)
    allowed = sequence['allowed'].gather(1, gather).squeeze(1)
    targets = sequence['targets'].gather(1, indices).squeeze(1)
    fork = F.cross_entropy(selected.float().masked_fill(~allowed, -1e9), targets, reduction='none')
    # Keep equal request weighting, including requests without a verified pair.
    extra = (fork * (positions >= 0)).mean()
    return ce + weight * extra, dict(event_ce=ce.detach(), fork_ce=extra.detach())
