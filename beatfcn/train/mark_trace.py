"""Read-only traces of the original T1 sampler and fixed greedy secondary track."""
import time

import numpy as np
import torch

from beatfcn.model.anchor_mark_chain import KINDS, sampled_marks
from beatfcn.model.local_typed_canvas import best_tail


@torch.no_grad()
def sampled_trace(model, x, heads, duration, *, index, seeds, labels=None, allowed_types=None, deadline=None):
    raw = []
    hook = model.kind.register_forward_hook(lambda module, args, output: raw.append(output.detach().cpu().numpy()))
    try:
        events = sampled_marks(model, x, heads, duration, index=index, view_enum=0,
            seeds=seeds, allowed_types=allowed_types, deadline=deadline)
    finally:
        hook.remove()
    logits = np.stack(raw, axis=1).astype(np.float64) if raw else np.empty((len(seeds), 0, 4))
    for j, head in enumerate(heads):
        if duration // 1000 - head // 1000 < 1: logits[:, j, 1] = -np.inf
    if allowed_types is not None:
        logits[:, :, [j for j, k in enumerate(KINDS) if k not in allowed_types]] = -np.inf
    normal = logits - np.max(logits, axis=-1, keepdims=True) if len(heads) else logits
    logp = normal - np.log(np.exp(normal).sum(-1, keepdims=True)) if len(heads) else normal
    result = dict(events=events, probabilities=np.exp(logp).tolist())
    if labels is not None:
        labels = np.asarray(labels, np.int64)
        nll = -logp[:, np.arange(len(heads)), labels]
        result.update(node_nll=nll.tolist(), nll=float(nll.mean()) if nll.size else 0,
                      truth=labels.tolist(), heads=heads)
    return result


@torch.no_grad()
def greedy_marks(model, x, heads, duration, *, allowed_types=None, deadline=None):
    model.eval(); features, hidden = model.encode(x)
    previous = torch.full((1, 1), 4, dtype=torch.long, device=features.device)
    events = []; probabilities = []
    for j, head in enumerate(heads):
        if deadline is not None and time.monotonic() > deadline: raise TimeoutError('Greedy diagnostic cap')
        h, hidden = model.gru(torch.cat((features[:, j:j+1], model.types(previous)), -1), hidden)
        logits = model.kind(h[:, 0]).cpu().numpy()[0].astype(np.float64)
        if duration // 1000 - head // 1000 < 1: logits[1] = -np.inf
        if allowed_types is not None:
            logits[[j for j, k in enumerate(KINDS) if k not in allowed_types]] = -np.inf
        p = np.exp(logits - logits.max()); p /= p.sum(); kind = int(p.argmax())
        probabilities.append(p.tolist()); end = head
        if kind == 1:
            fields = [layer(h[:, 0]).cpu().numpy()[0] for layer in (model.tail_block, model.tail_cell, model.tail_ms)]
            end = head // 1000 * 1000 + best_tail(*fields, 0, (duration // 1000 - head // 1000) * 1000)
        events.append(dict(onset_us=int(head), end_us=int(end), type=KINDS[kind], lane=-1))
        previous = torch.tensor([[kind]], dtype=torch.long, device=features.device)
    return dict(events=events, probabilities=probabilities)
