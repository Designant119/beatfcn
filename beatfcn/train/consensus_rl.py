"""Frozen-teacher agreement surrogate and leave-one-out policy gradient.

Agreement alone is not an admitted music-quality or request-following reward.
"""
import numpy as np
import torch
from torch.nn import functional as F

from beatfcn.model.full_event_codec import FIELDS

FIELD_TEMPERATURE = {'kind': 1., 'head_block': .7, 'head_cell': .7, 'head_ms': .7}


def event_f1(predicted, target, *, tolerance_us=20_000):
    """Symmetric one-to-one event matching; duplicates and missing holds cost recall/precision."""
    if not predicted and not target:
        return 1.
    if tolerance_us < 0:
        raise ValueError('Tolerance must be nonnegative')
    edges = [[j for j, t in enumerate(target)
              if kind == t[2] and lane == t[3] and abs(onset-t[0]) <= tolerance_us
              and (kind != 'hold' or abs(end-t[1]) <= tolerance_us)]
             for onset, end, kind, lane in predicted]
    owners = {}
    def augment(i, visited):
        for j in edges[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in owners or augment(owners[j], visited):
                owners[j] = i
                return True
        return False
    # Maximum matching avoids nearest-first stealing another note's sole match.
    matches = sum(augment(i, set()) for i in range(len(predicted)))
    return 2 * matches / max(len(predicted) + len(target), 1)


def leave_one_out(rewards):
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 1 or len(rewards) < 2 or not np.isfinite(rewards).all():
        raise ValueError('Need at least two finite rewards from one request')
    return (rewards - (rewards.sum() - rewards) / (len(rewards) - 1)).tolist()


def policy_terms(logits, reference_logits, sequence, advantages):
    allowed, valid = sequence['allowed'], sequence['valid']
    temperatures = torch.ones_like(sequence['fields'], dtype=torch.float32)
    for field, temperature in FIELD_TEMPERATURE.items():
        temperatures = torch.where(sequence['fields'] == FIELDS.index(field), temperature, temperatures)
    def log_probs(x):
        return F.log_softmax((x.float() / temperatures[..., None]).masked_fill(~allowed, -1e9), -1)
    logp, ref = log_probs(logits), log_probs(reference_logits.detach())
    selected = logp.gather(-1, sequence['targets'][..., None]).squeeze(-1)
    decision = valid & (allowed.sum(-1) > 1)
    # A fixed scalar rescales the exact sequence score; per-sequence length normalization would change the objective.
    pg = -(advantages * (selected * decision).sum(-1) / 128.).mean()
    kl = (ref.exp() * (ref - logp)).sum(-1)
    kl = (kl * decision).sum() / decision.sum().clamp_min(1)
    return pg, kl
