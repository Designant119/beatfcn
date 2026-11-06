"""Opt-in fixed-normalizer sequence likelihood experiment; not a default loss.

The constants come only from the frozen training exposure. Unlike a per-request
token mean or a batch-token mean, the weight of a decision does not depend on its
target sequence length or on other requests in its physical microbatch.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ModeSequenceNLL(nn.Module):
    def __init__(self, *, mania_mean_tokens, phira_mean_tokens):
        super().__init__()
        values = (float(mania_mean_tokens), float(phira_mean_tokens))
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError('Frozen mode normalizers must be finite and positive')
        self.register_buffer('mean_tokens', torch.tensor(values, dtype=torch.float32))

    def forward(self, logits, sequence, modes):
        if modes.ndim != 1 or len(modes) != len(logits) or modes.dtype != torch.int64:
            raise ValueError('One int64 mode index is required per request')
        allowed, valid, targets = sequence['allowed'], sequence['valid'], sequence['targets']
        selected = logits.float().masked_fill(~allowed, -1e9)
        losses = F.cross_entropy(selected.flatten(0, 1), targets.flatten(), reduction='none').view_as(targets)
        per_request = (losses * valid).sum(-1) / self.mean_tokens.index_select(0, modes)
        # A real empty music window still has a valid END token and contributes.
        return per_request.sum() / (valid.sum(-1) > 0).sum().clamp_min(1)
