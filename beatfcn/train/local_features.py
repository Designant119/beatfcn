"""Bounded local feature windows with distinct Pupu and complete-mel coverage."""
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

from beatfcn.audio.preprocess import SR, HOP, CELL_SAMPLES
from beatfcn.train.worker_cache import WorkerCacheMixin


def song_summary(layer24, scale):
    """Keep the exact D65 half-second pooling recipe; no chart labels are used."""
    bins = ((np.arange(len(layer24))*40_000+20_000)//500_000).astype(int)
    counts = np.bincount(bins)
    values = np.zeros((len(counts), 1024), np.float32)
    for block in range(len(counts)):
        lo, hi = np.searchsorted(bins, (block, block+1))
        values[block] = np.asarray(layer24[lo:hi], np.float32).mean((0, 1))*scale
    times = np.bincount(bins, weights=np.arange(len(layer24))*.04+.02)/counts
    return dict(values=values, times=times.astype(np.float32))


def feature_window(layers, scales, mel, summary, samples, start_us, end_us):
    if not 0 <= start_us < end_us or end_us*SR > samples*1_000_000:
        raise ValueError('Requested target is outside decoded music')
    legacy_cells = samples//CELL_SAMPLES
    if any(x.shape != (legacy_cells, 8, 1024) for x in layers):
        raise ValueError('Pupu arrays do not match full legacy cells')
    frames = (samples+CELL_SAMPLES-1)//CELL_SAMPLES*4
    if mel.shape != (frames, 128):
        raise ValueError('Complete mel does not cover the padded final cell')
    lo, nominal_hi = max(0, start_us//40_000-100), start_us//40_000+300
    hi = min(legacy_cells, nominal_hi)
    if hi <= lo or end_us > nominal_hi*40_000:
        raise ValueError('Request exceeds the fixed audio context')
    local = np.stack([np.asarray(x[lo:hi], np.float32)*scales[f'scale_{layer}']
                      for x, layer in zip(layers, (9, 16, 24))])
    mlo, mhi = lo*4, min(frames, nominal_hi*4)
    indices = np.arange(mlo, mhi)
    valid = indices*HOP < samples
    times = indices*.01+.005-start_us/1e6
    partial = valid & ((indices+1)*HOP > samples)
    times[partial] = (indices[partial]*HOP+samples)/(2*SR)-start_us/1e6
    return dict(local=local, local_times=(np.arange(lo, hi)*.04+.02-start_us/1e6).astype(np.float32),
        local_valid=np.ones(hi-lo, bool), song=summary['values'],
        song_times=(summary['times']-start_us/1e6).astype(np.float32),
        song_valid=np.ones(len(summary['values']), bool),
        mel=np.ascontiguousarray(mel[mlo:mhi], dtype=np.float32),
        mel_times=times.astype(np.float32), mel_valid=valid)


class LocalFeatureStore(WorkerCacheMixin):
    """Loads only audited songs; incomplete audits are usable only when explicit."""
    worker_cache_specs = (('_open', '_read_song', 2), ('window', '_window', 6))
    def __init__(self, root, report, *, allow_partial=False):
        self.root = Path(root)
        self.report = json.loads((self.root/report).read_text())
        if (not self.report['complete'] and not allow_partial) or self.report.get('failures'):
            raise ValueError('Feature admission audit incomplete or failed')
        with np.load(self.root/self.report['scales'], allow_pickle=False) as archive:
            self.scales = {k: archive[k].copy() for k in archive.files}
        self.records = {r['audio_sha256']: r for r in self.report['records'] if r['passed']}
        # Per-instance caches do not keep other experiments alive.
        self._open = lru_cache(maxsize=2)(self._read_song)
        self.window = lru_cache(maxsize=6)(self._window)

    def _read_song(self, audio):
        record = self.records[audio]
        layers = [np.load(self.root/record['paths'][f'L{l}.npy'], mmap_mode='r', allow_pickle=False)
                  for l in (9, 16, 24)]
        mel = np.load(self.root/record['complete_mel'], mmap_mode='r', allow_pickle=False)
        with np.load(self.root/record['song_summary'], allow_pickle=False) as archive:
            summary = {k: archive[k].copy() for k in archive.files}
        return layers, mel, summary

    def mel_frames(self, audio):
        return self._open(audio)[1][:math.ceil(self.records[audio]['decoded_samples']/240)]

    def _window(self, audio, start_us, end_us):
        layers, mel, summary = self._open(audio)
        return feature_window(layers, self.scales, mel, summary,
                              self.records[audio]['decoded_samples'], start_us, end_us)
