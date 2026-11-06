"""Fixed post-onset mel window. The label's end time is never an input."""
import numpy as np
import torch
from torch.nn import functional as F


def _frame_index(times, onset_s, valid):
    distance = np.abs(np.asarray(times, np.float64) - onset_s)
    if valid is not None:
        distance = np.where(np.asarray(valid, bool), distance, np.inf)
    if not np.isfinite(distance).any():
        raise ValueError('No valid mel frame at the onset')
    return int(np.argmin(distance))


def onset_and_window(mel, times, onset_s, n_frames, valid=None):
    """Onset frame and the mean of the fixed run of frames that starts there.

    ``n_frames`` is a constant. A hold's annotated end does not lengthen it.
    """
    if type(n_frames) is not int or n_frames < 1:
        raise ValueError('n_frames must be a positive integer')
    mel = np.asarray(mel, np.float32)
    start = _frame_index(times, onset_s, valid)
    frame = mel[start].copy()
    window = mel[start:start + n_frames]
    if len(window) < 1:
        raise ValueError('Onset window is empty')
    return frame, window.mean(axis=0).astype(np.float32)


def roll_mel(mel, fraction):
    """Move mel content along time and leave the caller-supplied clock unchanged."""
    mel = np.asarray(mel)
    if mel.ndim != 2 or not 0 < fraction < 1:
        raise ValueError('roll_mel expects a 2-d mel and a fraction inside (0, 1)')
    shift = int(mel.shape[0] * fraction)
    if shift <= 0:
        shift = 1 if mel.shape[0] > 1 else 0
    return np.roll(mel, shift, axis=0)


def fit_linear(features, labels, train, n_classes, steps, seed, lr):
    """Multinomial linear classifier. Returns weight [C, D] and bias [C]."""
    if steps < 1 or n_classes < 2:
        raise ValueError('fit_linear needs at least one step and two classes')
    x = torch.as_tensor(np.asarray(features, np.float32))
    y = torch.as_tensor(np.asarray(labels, np.int64))
    train = np.asarray(train, bool)
    if int(train.sum()) < n_classes:
        raise ValueError('Training split does not cover the classifier')
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(n_classes, x.shape[1], generator=generator) * 0.01
    bias = torch.zeros(n_classes)
    weight.requires_grad_(True)
    bias.requires_grad_(True)
    optimizer = torch.optim.AdamW((weight, bias), lr=lr, weight_decay=0.)
    train_x, train_y = x[torch.as_tensor(train)], y[torch.as_tensor(train)]
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(train_x @ weight.T + bias, train_y)
        loss.backward()
        optimizer.step()
    return weight.detach(), bias.detach(), float(loss.detach())


def recall_of(features, labels, test, weight, bias, positive):
    x = torch.as_tensor(np.asarray(features, np.float32))
    y = np.asarray(labels)
    test = np.asarray(test, bool)
    with torch.no_grad():
        pred = (x @ weight.T + bias).argmax(1).numpy()
    truth = y[test]
    chosen = pred[test]
    mask = truth == positive
    support = int(mask.sum())
    if support == 0:
        return dict(support=0, recall=None)
    return dict(support=support, recall=float((chosen[mask] == positive).mean()))
