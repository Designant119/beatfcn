"""Isolated SCST mechanics; no experiment or production runner enables this.

Scores come from actual outcomes in condition_outcome. This module only samples
at temperature one and replays that sample's own prefix with gradients. It also
retains capped prefixes, so a failed rollout cannot disappear from the loss.
"""
from dataclasses import asdict
import math
import time

import numpy as np
import torch

from beatfcn.model.full_event_codec import BOS, EventState


def _versions(model):
    return tuple((name, id(p), p._version) for name, p in model.named_parameters())


def _check_model(model, inputs):
    if model.training or len(inputs['mode']) != 1:
        raise ValueError('SCST requires eval mode and one actual request')


def prefix_sequence(tokens, request, vocab):
    """Replay complete or capped samples without appending a synthetic END."""
    if not tokens:
        raise ValueError('No sampled actions to replay')
    state = EventState(request, vocab)
    previous, fields, times, masks = [], [], [], []
    for token in tokens:
        if state.done:
            raise ValueError('Actions after END are not a trajectory')
        field, clock = state.query()
        previous.append(state.tokens[-1] if state.tokens else BOS)
        fields.append(field)
        times.append(clock)
        mask = np.zeros(vocab.size, dtype=bool)
        mask[list(state.allowed_tokens())] = True
        masks.append(mask)
        state.apply(token)
    return dict(tokens=np.asarray(previous, np.int64), fields=np.asarray(fields, np.int64),
                times=np.asarray(times, np.float32), targets=np.asarray(tokens, np.int64),
                allowed=np.asarray(masks), valid=np.ones(len(tokens), bool)), state


@torch.no_grad()
def sample_trajectory(model, inputs, request, max_tokens, *, deadline=None):
    """Unmodified legal T=1 policy, with online probabilities for replay audit."""
    _check_model(model, inputs)
    if type(max_tokens) is not int or not 0 < max_tokens <= request.max_tokens:
        raise ValueError('Invalid rollout token limit')
    versions = _versions(model)
    state = EventState(request, model.vocab)
    memory = model.encode(inputs)
    cache = model.start_cache(memory)
    device = inputs['mode'].device
    log_probabilities = []
    deadline_reached = False
    for _ in range(max_tokens):
        if state.done:
            break
        if deadline is not None and time.monotonic() >= deadline:
            deadline_reached = True
            break
        field, clock = state.query()
        previous = state.tokens[-1] if state.tokens else BOS
        logits = model.decode(torch.tensor([[previous]], device=device),
            torch.tensor([[field]], device=device),
            torch.tensor([[clock]], dtype=torch.float32, device=device),
            torch.ones(1, 1, dtype=torch.bool, device=device), memory, cache=cache)[0, 0].float()
        legal = torch.tensor(state.allowed_tokens(), device=device)
        if not len(legal) or not torch.isfinite(logits[legal]).all():
            raise ValueError('Invalid legal sampling support')
        # No repeated legal IDs for support padding: that changes probabilities.
        log_probs = logits[legal].log_softmax(0)
        choice = torch.multinomial(log_probs.exp(), 1)
        token = int(legal[choice].item())
        log_probabilities.append(float(log_probs[choice].item()))
        state.apply(token)
    if _versions(model) != versions:
        raise ValueError('Policy changed during sampling')
    return dict(tokens=list(state.tokens), events=[asdict(e) for e in sorted(state.events)],
                complete=state.done, deadline_reached=deadline_reached,
                token_limit_reached=not state.done and not deadline_reached,
                online_log_probabilities=log_probabilities, parameter_versions=versions)


def replay_log_probabilities(model, inputs, request, trajectory):
    """Differentiable log probabilities under the exact sampled policy state."""
    _check_model(model, inputs)
    if trajectory['parameter_versions'] != _versions(model):
        raise ValueError('Policy changed between sampling and replay')
    arrays, state = prefix_sequence(trajectory['tokens'], request, model.vocab)
    if state.done != trajectory['complete']:
        raise ValueError('Trajectory completion does not match its own actions')
    device = inputs['mode'].device
    sequence = {k: torch.from_numpy(v)[None].to(device) for k, v in arrays.items()}
    logits = model(inputs, sequence).float()
    legal_logits = logits.masked_fill(~sequence['allowed'], -torch.inf)
    if not torch.isfinite(logits[sequence['allowed']]).all():
        raise ValueError('Non-finite policy on legal actions')
    return legal_logits.log_softmax(-1).gather(-1, sequence['targets'][..., None])[0, :, 0]


def scst_loss(log_probabilities, sampled_reward, greedy_reward, *, normalizer):
    """One trajectory, fixed global C; rewards/baseline never receive gradients."""
    if (isinstance(normalizer, bool) or not math.isfinite(normalizer) or normalizer <= 0
            or log_probabilities.ndim != 1 or not log_probabilities.numel()
            or not torch.isfinite(log_probabilities).all()):
        raise ValueError('SCST requires finite actions and a frozen positive global C')
    sampled = torch.as_tensor(sampled_reward, device=log_probabilities.device).detach()
    greedy = torch.as_tensor(greedy_reward, device=log_probabilities.device).detach()
    if sampled.numel() != 1 or greedy.numel() != 1 or not torch.isfinite(sampled).all() or not torch.isfinite(greedy).all():
        raise ValueError('Expected finite scalar outcome rewards')
    return -(sampled.reshape(()) - greedy.reshape(())) * log_probabilities.sum() / normalizer
