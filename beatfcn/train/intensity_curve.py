"""Audio-derived intensity envelopes usable at inference (no target leakage)."""
import math

import numpy as np


def audio_intensity_curve(features, audio, start_us, end_us, *, cell_us=500_000,
                          scale=16.0, mode='reference'):
    if mode not in ('reference', 'shuffle', 'zero'):
        raise ValueError('Unknown intensity curve mode')
    cells = max(1, math.ceil((end_us - start_us) / cell_us))
    values = np.zeros(cells, np.float64)
    if mode != 'zero':
        window = features.window(audio, start_us, end_us)
        mel = np.asarray(window['mel'], np.float32)
        times, valid = np.asarray(window['mel_times']), np.asarray(window['mel_valid'], bool)
        for cell in range(cells):
            lo, hi = cell*cell_us/1e6, min((cell+1)*cell_us, end_us-start_us)/1e6
            block = mel[valid & (times >= lo) & (times < hi)]
            if len(block):
                values[cell] = float(np.abs(block).mean())
        peak = float(values.max())
        if peak > 0:
            values = values / peak
        if mode == 'shuffle':
            values = values[::-1].copy()
    lo = np.arange(cells)*cell_us
    times = (lo+np.minimum(lo+cell_us, end_us-start_us))/2e6
    return dict(intensity=values[:, None].astype(np.float32),
                intensity_times=times.astype(np.float32),
                intensity_valid=np.ones(cells, bool))
