"""Fine-tune objective: the round-5 terms plus optional token weights and distillation.

With no weights and no distillation this is exactly training_objective.exposure_group_loss.
"""
import torch
from torch.nn import functional as F

from beatfcn.events.canonical import TYPES
from beatfcn.model.full_event_codec import FIELDS
from beatfcn.train.training_objective import objective_terms


def token_weight_table(vocab, field_weights=None, kind_class_weights=None):
    """Weight per target token [V], or None when nothing is weighted.

    ``field_weights`` {field: w}; ``kind_class_weights`` {type: w}, on top of the field weight. Only Phira emits
    ``kind`` tokens, so kind classes are Phira type weights. Token weights shift the learned prior: at convergence
    a class weight w adds log w to that class's logit."""
    field_weights, kind_class_weights = dict(field_weights or {}), dict(kind_class_weights or {})
    if not field_weights and not kind_class_weights:
        return None
    if set(field_weights)-set(FIELDS) or set(kind_class_weights)-set(TYPES):
        raise ValueError('Unknown loss field or kind class')
    if any(w < 0 for w in (*field_weights.values(), *kind_class_weights.values())):
        raise ValueError('Token weights must be nonnegative')
    table = torch.ones(vocab.size)
    offset = 2
    for field, size in zip(FIELDS, vocab.sizes):
        table[offset:offset+size] = float(field_weights.get(field, 1.))
        offset += size
    for kind, weight in kind_class_weights.items():
        table[vocab.token('kind', TYPES.index(kind))] *= float(weight)
    return table


def distillation_loss(logits, teacher_logits, sequence, temperature):
    """Hinton T^2 KL(teacher || student) over the legal support, mean per decision token, equal weight per request."""
    allowed, valid = sequence['allowed'], sequence['valid']
    keep = valid & (allowed.sum(-1) > 1)
    student = F.log_softmax(logits.float().masked_fill(~allowed, -1e9)/temperature, -1)
    teacher = F.softmax(teacher_logits.float().masked_fill(~allowed, -1e9)/temperature, -1)
    kl = F.kl_div(student, teacher, reduction='none').sum(-1)*temperature**2
    count = keep.sum(-1)
    return ((kl*keep).sum(-1)/count.clamp_min(1)).sum()/(count > 0).sum().clamp_min(1)


def _weighted_task(logits, sequence, settings, token_weights, aux):
    terms = objective_terms(logits, sequence, aux=aux, **settings)
    if token_weights is None:
        return terms['total']
    valid, targets = sequence['valid'], sequence['targets']
    legal = logits.float().masked_fill(~sequence['allowed'], -1e9)
    ce = F.cross_entropy(legal.flatten(0, 1), targets.flatten(), reduction='none').view_as(valid)
    # The count stays the normalizer so a weight raises that token's share of the objective.
    weighted = ce*token_weights.to(ce.device)[targets]
    count = valid.sum(-1)
    main = ((weighted*valid).sum(-1)/count.clamp_min(1)).sum()/(count > 0).sum().clamp_min(1)
    kind = valid & (sequence['fields'] == FIELDS.index('kind'))
    count = kind.sum(-1)
    type_ce = ((weighted*kind).sum(-1)/count.clamp_min(1)).sum()/(count > 0).sum().clamp_min(1)
    type_weight = settings.get('type_weight', 0.)
    return terms['total']-terms['joint_ce']-type_weight*terms['type_ce']+main+type_weight*type_ce


def evolve_loss(logits, sequence, settings, *, aux=None, token_weights=None, teacher_logits=None, kd=None,
                group_size=4):
    """Same four-request auxiliary normalization as exposure_group_loss, then (1-alpha) task + alpha KD."""
    batch = logits.shape[0]
    if batch % group_size and batch > group_size:
        raise ValueError('Physical batch must contain complete exposure groups')
    if batch <= group_size:
        task = _weighted_task(logits, sequence, settings, token_weights, aux)
    else:
        def rows(mapping, start):
            return None if mapping is None else {k: v[start:start+group_size] for k, v in mapping.items()}
        task = sum(_weighted_task(logits[s:s+group_size], rows(sequence, s), settings, token_weights, rows(aux, s))
                   for s in range(0, batch, group_size))/(batch//group_size)
    if kd is None:
        return task
    alpha = float(kd['alpha'])
    return (1-alpha)*task+alpha*distillation_loss(logits, teacher_logits, sequence, float(kd['temperature']))


@torch.no_grad()
def field_ce_sums(logits, sequence):
    """(sum, count) of CE per grammar field over valid decision tokens (forced tokens excluded), for logs."""
    allowed, valid, rows = sequence['allowed'], sequence['valid'], sequence['fields']
    legal = logits.float().masked_fill(~allowed, -1e9)
    ce = F.cross_entropy(legal.flatten(0, 1), sequence['targets'].flatten(), reduction='none').view_as(valid)
    keep = valid & (allowed.sum(-1) > 1)
    return (torch.bincount(rows[keep], weights=ce[keep], minlength=len(FIELDS)),
            torch.bincount(rows[keep], minlength=len(FIELDS)))
