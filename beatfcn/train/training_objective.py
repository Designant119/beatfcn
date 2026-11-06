"""Explicit event objective and optimizer-update learning rate contract."""
import math

import numpy as np
import torch
from torch.nn import functional as F

from beatfcn.events.canonical import TYPES
from beatfcn.model.full_event_codec import FIELDS
from beatfcn.model.full_event_model import ONSET_CHANNELS, event_loss


def onset_frame_targets(onsets_us, frame_times_s, request_start_us, *, onset_types=None,
                        frame_valid=None, sigma_s=None):
    """Soft per-frame onset targets for the ``onset_head`` auxiliary.

    onsets_us: chart onset times (absolute song microseconds). Pass every
        onset of the song that may fall near the frames (core AND audio context);
        a hold counts once at its head, tails are not onsets. Duplicates are fine.
    frame_times_s: [T] frame times in seconds relative to ``request_start_us``,
        exactly the model's frame clock: ``mel_times`` when the config has
        ``mel_bins`` > 0, else ``local_times`` (one row per MERT frame).
    onset_types: optional per-onset canonical type names (TYPES); fills the
        per-type channels 1+k. Without it only channel 0 (any onset) is set.
    frame_valid: optional [T] bool audio validity (padding/out-of-song = False).
    sigma_s: Gaussian width in seconds; default one median frame hop.

    Returns dict(onset_targets=float32 [T, ONSET_CHANNELS],
    onset_valid=bool [T]). Each onset gives exp(-dt^2/2sigma^2) (cut at 4 sigma), its nearest
    frame (if within one hop) is set to 1; overlapping onsets take the max.
    The collator pads targets with 0 and onset_valid with False to the model's
    padded frame length and stacks them into sequence['onset_targets'] [B,T,C]
    and sequence['onset_valid'] [B,T].
    """
    times = np.asarray(frame_times_s, np.float64).reshape(-1)
    targets = np.zeros((len(times), ONSET_CHANNELS), np.float32)
    valid = (np.ones(len(times), bool) if frame_valid is None
             else np.asarray(frame_valid, bool).reshape(-1).copy())
    if len(valid) != len(times):
        raise ValueError('frame_valid must align with frame_times_s')
    onsets = (np.asarray(onsets_us, np.int64).reshape(-1) - int(request_start_us)) / 1e6
    if onset_types is not None:
        onset_types = list(onset_types)
        if len(onset_types) != len(onsets) or not set(onset_types) <= set(TYPES):
            raise ValueError('onset_types must align with onsets_us and use canonical TYPES')
    if not len(times) or not len(onsets):
        return dict(onset_targets=targets, onset_valid=valid)
    hop = float(np.median(np.diff(times))) if len(times) > 1 else .01
    if not hop > 0:
        raise ValueError('frame_times_s must be increasing')
    sigma = hop if sigma_s is None else float(sigma_s)
    if not sigma > 0:
        raise ValueError('sigma_s must be positive')
    near = (onsets >= times[0] - 4 * sigma - hop) & (onsets <= times[-1] + 4 * sigma + hop)
    index = np.flatnonzero(near)
    if not len(index):
        return dict(onset_targets=targets, onset_valid=valid)
    delta = times[:, None] - onsets[None, index]
    soft = np.where(np.abs(delta) <= 4 * sigma, np.exp(-.5 * (delta / sigma) ** 2), 0.)
    nearest = np.abs(delta).argmin(0)
    hit = np.abs(delta[nearest, np.arange(len(index))]) <= hop
    soft[nearest[hit], np.flatnonzero(hit)] = 1.
    channels = [(0, np.ones(len(index), bool))]
    if onset_types is not None:
        kinds = np.asarray([onset_types[i] for i in index])
        channels += [(1 + k, kinds == kind) for k, kind in enumerate(TYPES)]
    for channel, members in channels:
        if members.any():
            targets[:, channel] = soft[:, members].max(1)
    return dict(onset_targets=targets, onset_valid=valid)


def onset_loss(aux, sequence, *, pos_weight=None, max_pos_weight=100.):
    """Positive-weighted soft BCE of the onset head, equal weight per request.

    Returns None when the model produced no onset head or the batch carries no
    ``onset_targets``/``onset_valid`` (the term is then skipped, value 0).
    Targets may carry fewer channels than the head (e.g. presence only); extra
    head channels are then unsupervised. ``pos_weight=None`` balances each
    channel by the batch ratio sum(1-y)/sum(y) (detached, clamped to
    [1, max_pos_weight]) so sparse onsets are not ignored.
    """
    if not aux or 'onset_logits' not in aux or 'onset_targets' not in sequence or 'onset_valid' not in sequence:
        return None
    logits, targets = aux['onset_logits'].float(), sequence['onset_targets'].float()
    if targets.ndim == 2:
        targets = targets[..., None]
    if targets.shape[:2] != logits.shape[:2] or targets.shape[-1] > logits.shape[-1]:
        raise ValueError(f'Onset targets {tuple(targets.shape)} do not align with head {tuple(logits.shape)}')
    logits = logits[..., :targets.shape[-1]]
    valid = sequence['onset_valid'].bool() & aux['onset_valid'].bool()
    weight = valid[..., None].float()
    if pos_weight is None:
        positive = (targets * weight).sum((0, 1))
        negative = ((1 - targets) * weight).sum((0, 1))
        pos_weight = (negative / positive.clamp_min(1e-6)).clamp(1., max_pos_weight).detach()
    else:
        pos_weight = torch.as_tensor(float(pos_weight), device=logits.device)
    bce = pos_weight * targets * F.softplus(-logits) + (1 - targets) * F.softplus(logits)
    count = weight.sum((1, 2))
    per_request = (bce * weight).sum((1, 2)) / count.clamp_min(1)
    return per_request.sum() / (count > 0).sum().clamp_min(1)


@torch.no_grad()
def onset_presence_auc(aux, sequence, *, threshold=.5):
    """Cheap 'does the encoder hear onsets' metric: ROC-AUC of channel 0 over
    valid frames, positives = target >= threshold. None if undefined."""
    if not aux or 'onset_logits' not in aux or 'onset_targets' not in sequence:
        return None
    targets = sequence['onset_targets'].float()
    targets = targets[..., 0] if targets.ndim == 3 else targets
    valid = sequence['onset_valid'].bool() & aux['onset_valid'].bool()
    scores, labels = aux['onset_logits'][..., 0].float()[valid], targets[valid] >= threshold
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return None
    ranks = torch.empty_like(scores)
    order = scores.argsort()
    ranks[order] = torch.arange(1, len(scores) + 1, device=scores.device, dtype=scores.dtype)
    # Average ranks over ties (Mann-Whitney U).
    unique, inverse = scores.unique(return_inverse=True)
    if len(unique) < len(scores):
        sums = torch.zeros(len(unique), device=scores.device).index_add_(0, inverse, ranks)
        counts = torch.zeros(len(unique), device=scores.device).index_add_(0, inverse, torch.ones_like(ranks))
        ranks = (sums / counts)[inverse]
    u = ranks[labels].sum() - positives * (positives + 1) / 2
    return float(u / (positives * negatives))


def objective_terms(logits, sequence, *, type_weight=0., z_weight=0., first_token_weight=0.,
                    onset_weight=0., onset_pos_weight=None, aux=None):
    """All objective terms, each reported separately (raw and weighted).

    ``aux`` is the second value of ``model(inputs, sequence, return_aux=True)``.
    The onset term is skipped (0, onset_active=0) if aux or the sequence's
    onset tensors are absent.
    """
    if type_weight < 0 or z_weight < 0 or first_token_weight < 0 or onset_weight < 0:
        raise ValueError('Loss weights must be nonnegative')
    main = event_loss(logits, sequence)
    valid = sequence['valid']
    legal = logits.float().masked_fill(~sequence['allowed'], -1e9)
    ce = F.cross_entropy(legal.flatten(0, 1), sequence['targets'].flatten(),
        reduction='none').view_as(valid)
    kind = valid & (sequence['fields'] == FIELDS.index('kind'))
    count = kind.sum(-1)
    type_ce = ((ce*kind).sum(-1)/count.clamp_min(1)).sum()/(count > 0).sum().clamp_min(1)
    z = legal.logsumexp(-1).square()
    count = valid.sum(-1)
    z_loss = ((z*valid).sum(-1)/count.clamp_min(1)).sum()/(count > 0).sum().clamp_min(1)
    first = (ce*valid)[:, 0].mean()
    onset = onset_loss(aux, sequence, pos_weight=onset_pos_weight) if onset_weight or aux else None
    onset_active = onset is not None
    if onset is None:
        onset = main.new_zeros(())
    weighted = dict(weighted_type_ce=type_weight*type_ce, weighted_z_loss=z_weight*z_loss,
        weighted_first_token_ce=first_token_weight*first, weighted_onset_bce=onset_weight*onset)
    # Same left-to-right order as the historical sum (bitwise-identical when the
    # onset term is off: it adds an exact 0).
    total = (main + weighted['weighted_type_ce'] + weighted['weighted_z_loss']
             + weighted['weighted_first_token_ce'] + weighted['weighted_onset_bce'])
    return dict(total=total, joint_ce=main, type_ce=type_ce, z_loss=z_loss, first_token_ce=first,
        onset_bce=onset, onset_active=main.new_tensor(float(onset_active)), **weighted)


def training_loss(logits, sequence, settings=None, aux=None):
    if not settings:
        return event_loss(logits, sequence)
    return objective_terms(logits, sequence, aux=aux, **settings)['total']


def update_learning_rate(optimizer, completed_updates, total_updates, *, base_lr,
                         warmup_updates=0, minimum_lr=0.):
    if (total_updates < 1 or not 0 <= warmup_updates < total_updates
            or base_lr <= 0 or not 0 <= minimum_lr <= base_lr):
        raise ValueError('Invalid optimizer-update learning rate schedule')
    if completed_updates < warmup_updates:
        lr = base_lr*(completed_updates+1)/warmup_updates
    else:
        progress = min(1., (completed_updates-warmup_updates)/max(1, total_updates-warmup_updates-1))
        lr = minimum_lr+(base_lr-minimum_lr)*(1+math.cos(math.pi*progress))/2
    for group in optimizer.param_groups:
        group['lr'] = lr
    return lr


def gradient_norms(model):
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            key = '.'.join(name.split('.')[:2])
            value = parameter.grad.detach().float().square().sum()
            groups[key] = groups.get(key, 0)+value
    return {key: float(value.sqrt()) for key, value in groups.items()}


def exposure_group_loss(logits, sequence, settings=None, group_size=4, aux=None):
    """Retain the original four-request auxiliary-loss normalization."""
    batch = logits.shape[0]
    if batch <= group_size:
        return training_loss(logits, sequence, settings, aux)
    if batch % group_size:
        raise ValueError('Physical batch must contain complete exposure groups')
    def rows(mapping, start):
        return None if mapping is None else {key: value[start:start+group_size] for key, value in mapping.items()}
    losses = [training_loss(logits[start:start+group_size], rows(sequence, start), settings,
        rows(aux, start)) for start in range(0, batch, group_size)]
    return sum(losses)/len(losses)
