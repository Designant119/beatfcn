"""Score each legal millisecond with its own audio query.

The production decoder asks for every millisecond in a 40ms cell with one
cross-attention time: the start of that cell. This module keeps that decoder
for every other decision. At head_ms positions only, a side branch rereads
audio. The control branch rereads at the cell start. The candidate branch
rereads at the cell start plus that millisecond.

Both branches attend only to earlier tokens. The production token also attends
to itself, so these scores are not a drop-in replacement for production logits.
The probe compares the two branches to each other, which share that restriction.
"""
import torch
from torch.nn import functional as F

from .full_event_codec import FIELDS


def _replay(model, tokens, fields, times, valid, memory):
    """Ordinary decoder, also returning each layer's history keys and values."""
    b, n = tokens.shape
    positions = torch.arange(n, device=tokens.device)[None].expand(b, -1)
    x = (model.token_embedding(tokens) + model.field_embedding(fields)
         + model.mode_ids(memory['mode'])[:, None])
    memory_kv = model.memory_kv(memory)
    causal = torch.arange(n, device=x.device)[None, :] <= torch.arange(n, device=x.device)[:, None]
    history = []
    for i, layer in enumerate(model.decoder):
        z = layer.norm1(x)
        k, v = layer.self_attn.project_kv(z, times, positions)
        mask = causal[None, None] & valid[:, None, None, :]
        x = x + layer.drop(layer.self_attn(z, times, positions, (k, v), mask))
        x = x + layer.drop(layer.cross_attn(
            layer.norm2(x), times, positions, memory_kv[i], memory['valid'][:, None, None, :]))
        x = x + layer.drop(layer.ffn(layer.norm3(x)))
        history.append((k, v))
    return model.output(model.final_norm(x)), history


def head_ms_logits(model, inputs, sequence, *, per_candidate_time):
    """Full logits with head_ms rows replaced by a shared side branch.

    ``per_candidate_time`` is the only behavioral switch. False keeps every
    legal millisecond on the cell-start query. True moves millisecond ``v``
    to ``cell_start + v ms``.
    """
    if sequence['tokens'].shape[0] != 1:
        raise ValueError('Candidate-time scoring is implemented for one request')
    if model.c.history_mixer != 'attention':
        raise ValueError('Candidate-time scoring uses the attention history mixer')
    memory = model.encode(inputs)
    tokens, fields, times, valid = (sequence[k] for k in ('tokens', 'fields', 'times', 'valid'))
    base, history = _replay(model, tokens, fields, times, valid, memory)
    field = FIELDS.index('head_ms')
    pos = ((fields[0] == field) & valid[0]).nonzero(as_tuple=False).flatten()
    if pos.numel() == 0:
        raise ValueError('Request has no head_ms decision')
    allowed = sequence['allowed'][0, pos]
    base_token = model.vocab.token('head_ms', 0)
    legal = allowed[:, base_token:base_token + 40]
    if not bool(legal.any(1).all()):
        raise ValueError('A head_ms decision has no legal millisecond')
    slot, value = legal.nonzero(as_tuple=True)
    origin = pos[slot]
    prefix = times[0, origin]
    query_time = prefix + value.float() * 0.001 if per_candidate_time else prefix
    # The input token is the previous decision, shared by every candidate.
    h = (model.token_embedding(tokens[0, origin]) + model.field_embedding(fields[0, origin])
         + model.mode_ids(memory['mode'])[0])[None]
    key_index = torch.arange(tokens.shape[1], device=tokens.device)
    attend = key_index[None, :] < origin[:, None]
    if not bool(attend.any(1).all()):
        raise ValueError('head_ms has no earlier token to condition on')
    query_time = query_time[None]
    query_pos = origin[None].to(query_time.dtype)
    memory_kv = model.memory_kv(memory)
    mem_mask = memory['valid'][:, None, None, :]
    pair_mask = attend[None, None]
    for i, layer in enumerate(model.decoder):
        k, v = history[i]
        z = layer.norm1(h)
        h = h + layer.drop(layer.self_attn(z, prefix[None], query_pos, (k, v), pair_mask))
        h = h + layer.drop(layer.cross_attn(
            layer.norm2(h), query_time, query_pos, memory_kv[i], mem_mask))
        h = h + layer.drop(layer.ffn(layer.norm3(h)))
    scores = model.output(model.final_norm(h))[0, torch.arange(value.shape[0], device=value.device),
                                                base_token + value]
    covered = torch.zeros_like(base, dtype=torch.bool)
    covered[0, pos, base_token:base_token + 40] = True
    addition = torch.zeros_like(base)
    addition[0, origin, base_token + value] = scores
    # Legal millisecond columns come only from the side branch. Every other
    # decision keeps the ordinary decoder logit and its gradient.
    return torch.where(covered, addition, base)


def head_ms_accuracy(logits, sequence, vocab):
    field = FIELDS.index('head_ms')
    valid = sequence['valid'][0]
    pos = ((sequence['fields'][0] == field) & valid).nonzero(as_tuple=False).flatten()
    base = vocab.token('head_ms', 0)
    chosen, target, correct = [], [], 0
    for index in pos.tolist():
        legal = sequence['allowed'][0, index, base:base + 40]
        values = legal.nonzero(as_tuple=False).flatten()
        pick = values[logits[0, index, base + values].argmax()]
        truth = int(sequence['targets'][0, index]) - base
        chosen.append(int(pick))
        target.append(truth)
        correct += int(pick) == truth
    return dict(count=len(target), correct=correct,
                accuracy=correct / max(len(target), 1),
                chosen=chosen, target=target)
