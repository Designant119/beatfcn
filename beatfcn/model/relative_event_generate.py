"""Isolated D88 cached generator, with the original greedy decoding algorithm.

Only the state class differs from full_event_generate. Keep this experimental
entry separate until whole-song quality supports promotion.
"""
from dataclasses import asdict
import time

import torch

from .full_event_codec import BOS
from .relative_event_codec import RelativeHeadState


@torch.no_grad()
def generate_batch(model, inputs, requests, max_tokens, *, deadline=None):
    if model.training or len(requests) != len(inputs['mode']) or not requests or max_tokens < 1:
        raise ValueError('Generation requires eval mode and one request per input row')
    states = [RelativeHeadState(request, model.vocab) for request in requests]
    device = inputs['local'].device
    memory = model.encode(inputs)
    cache = model.start_cache(memory)
    deadline_reached = False
    for _ in range(max_tokens):
        if deadline is not None and time.monotonic() >= deadline:
            deadline_reached = True
            break
        active = [not state.done for state in states]
        if not any(active):
            break
        queries = [state.query() if enabled else (0, 0.) for state, enabled in zip(states, active)]
        previous = [state.tokens[-1] if state.tokens else BOS for state in states]
        logits = model.decode(torch.tensor(previous, device=device)[:, None],
            torch.tensor([q[0] for q in queries], device=device)[:, None],
            torch.tensor([q[1] for q in queries], device=device, dtype=torch.float32)[:, None],
            torch.ones(len(states), 1, device=device, dtype=torch.bool), memory, cache=cache)[:, 0]
        legal = [state.allowed_tokens() if enabled else [BOS] for state, enabled in zip(states, active)]
        if any(not len(s) for s in legal):
            raise ValueError('Grammar reached a dead end')
        longest = max(map(len, legal))
        indices = torch.tensor([list(s) + [s[0]] * (longest-len(s)) for s in legal], device=device)
        selected = indices.gather(1, logits.gather(1, indices).argmax(-1, keepdim=True)).squeeze(1).tolist()
        for state, token, enabled in zip(states, selected, active):
            if enabled:
                state.apply(token)
    return [dict(complete=state.done, events=[asdict(e) for e in sorted(state.events)], tokens=state.tokens,
                 deadline_reached=deadline_reached and not state.done,
                 token_limit_reached=not state.done and not deadline_reached) for state in states]
