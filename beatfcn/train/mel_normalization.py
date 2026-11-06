"""Train-only mel channel statistics; raw immutable feature caches stay intact."""
import hashlib
import json

import numpy as np


def fit_mel_statistics(features, train_audios, forbidden_audios=()):
    audios = sorted(set(train_audios))
    if not audios or set(audios) & set(forbidden_audios):
        raise ValueError('Mel statistics need disjoint training audio only')
    total, square, count = np.zeros(128), np.zeros(128), 0
    for audio in audios:
        frames = features.mel_frames(audio)
        for start in range(0, len(frames), 8192):
            block = np.asarray(frames[start:start+8192], np.float64)
            if block.ndim != 2 or block.shape[1] != 128 or not np.isfinite(block).all():
                raise ValueError('Invalid raw mel statistics input')
            total += block.sum(0)
            square += np.square(block).sum(0)
            count += len(block)
    if not count:
        raise ValueError('Empty mel training frames')
    mean = total/count
    std = np.sqrt(np.maximum(square/count-mean*mean, 0)).clip(min=1e-5)
    return dict(schema='train-mel-channel-v1', train_audios=audios, frames=count,
        audio_set_sha256=hashlib.sha256(json.dumps(audios).encode()).hexdigest(),
        mean=mean.tolist(), std=std.tolist())


class MelNormalizedFeatures:
    def __init__(self, base, statistics, *, train_audios, forbidden_audios):
        audios = set(statistics['train_audios'])
        if (statistics['schema'] != 'train-mel-channel-v1' or not audios
                or not audios <= set(train_audios) or audios & set(forbidden_audios)):
            raise ValueError('Normalization statistics include unauthorized/nontraining audio')
        self.base = base
        self.mean = np.asarray(statistics['mean'], np.float32)
        self.std = np.asarray(statistics['std'], np.float32)
        if (self.mean.shape != (128,) or self.std.shape != (128,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or (self.std <= 0).any()):
            raise ValueError('Invalid mel normalization coefficients')

    def __getattr__(self, name):
        base = self.__dict__.get('base')
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def window(self, audio, start_us, end_us):
        row = dict(self.base.window(audio, start_us, end_us))
        values = (row['mel']-self.mean)/self.std
        values[~row['mel_valid']] = 0
        row['mel'] = values
        return row
