"""Isolated input interventions; no optimizer or production defaults."""
import torch

from beatfcn.model.full_event_codec import FIELDS


def audio_intervention(inputs, name):
    out = dict(inputs)
    if name.startswith('local_'):
        sign = 1 if name == 'local_plus200' else -1
        for key, valid_key, axis, shift in (
                ('local', 'local_valid', 2, sign * 5),
                ('mel', 'mel_valid', 1, sign * 20)):
            value = inputs[key].clone()
            for b, mask in enumerate(inputs[valid_key]):
                ids = mask.nonzero().flatten()
                if key == 'local':
                    value[b, :, ids] = inputs[key][b, :, ids].roll(shift, 1)
                else:
                    value[b, ids] = inputs[key][b, ids].roll(shift, 0)
            out[key] = value
    elif name.startswith('remote_'):
        value = inputs['song'].clone()
        for b in range(len(value)):
            local_times = inputs['mel_times'][b][inputs['mel_valid'][b]]
            remote = (inputs['song_valid'][b]
                      & ((inputs['song_times'][b] < local_times.min() - .5)
                         | (inputs['song_times'][b] > local_times.max() + .5)))
            ids = remote.nonzero().flatten()
            if len(ids) < 2:
                raise ValueError('Insufficient remote song context')
            permuted = ids.flip(0) if name == 'remote_reverse' else ids.roll(len(ids) // 2)
            value[b, ids] = inputs['song'][b, permuted]
        out['song'] = value
    elif name != 'correct':
        raise ValueError(name)
    return out


def legal_probabilities(logits, sequence):
    return logits.float().masked_fill(~sequence['allowed'], -1e9).softmax(-1)


def decision_rows(logits, sequence, vocab, baseline=None):
    probs = legal_probabilities(logits, sequence)[0]
    predicted = probs.argmax(-1)
    records = {}
    for field_id, name in enumerate(FIELDS):
        mask = sequence['valid'][0] & (sequence['fields'][0] == field_id)
        ids = mask.nonzero().flatten()
        if not len(ids):
            continue
        truth = sequence['targets'][0, ids]
        chosen = predicted[ids]
        p = probs[ids].gather(1, truth[:, None]).squeeze(1)
        support = sequence['allowed'][0, ids].sum(-1)
        values = [vocab.decision(int(v))[1] for v in truth]
        guessed = [vocab.decision(int(v))[1] for v in chosen]
        row = dict(count=len(ids), correct=int((truth == chosen).sum()),
                   nonforced_count=int((support > 1).sum()),
                   nonforced_correct=int(((support > 1) & (truth == chosen)).sum()),
                   nll=(-p.clamp_min(1e-30).log()).tolist(),
                   target_probability=p.tolist(), truth=values, predicted=guessed)
        if name in ('head_cell', 'head_ms'):
            unit = 40 if name == 'head_cell' else 1
            row['conditional_error_ms'] = [(b - a) * unit for a, b in zip(values, guessed)]
        if baseline is not None:
            row['tv'] = (.5 * (probs[ids] - baseline[ids]).abs().sum(-1)).tolist()
        records[name] = row
    return records
