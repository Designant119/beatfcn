"""C08 isolated ablation: keep BOS and acoustic recurrence, gate mark feedback."""
import torch
from torch import nn
from torch.nn import functional as F

from .anchor_mark_chain import AnchorMarkChain


class MarkFeedback(nn.Module):
    def __init__(self, embedding, gate):
        super().__init__()
        if gate not in (0, 1):
            raise ValueError('Only preregistered binary feedback gates are supported')
        self.weight = embedding.weight
        self.gate = gate

    def forward(self, ids):
        value = F.embedding(ids, self.weight)
        scale = torch.where(ids == 4, 1, self.gate).to(value.dtype)
        return value * scale[..., None]


class NoMarkHistory(AnchorMarkChain):
    def __init__(self, d=128, *, gate=0):
        super().__init__(d)
        # Reuse the same Parameter; no random draw, name, shape or order changes.
        self.types = MarkFeedback(self.types, gate)
