"""MERT receipt windows for the event model; same contract as LocalFeatureStore.

Reads the immutable per-song MERT receipts (L12/L23 float16, per-frame clock,
mel10_complete) straight from the mounted volume.  No copy into the image and no
local download: windows are sliced from mmapped arrays.
"""
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

from beatfcn.audio.mert2 import CELL_SAMPLES, SR
from beatfcn.train.worker_cache import WorkerCacheMixin


FRAME_US = 40_000
MEL_HOP_US = 10_000
SUMMARY_US = 500_000


class MertFeatureStore(WorkerCacheMixin):
    """Panel-facing MERT windows: local [2,T,1,1024], song [S,1024], mel [T2,128]."""
    worker_cache_specs = (('_cache', '_open', 128), ('_summary', '_build_summary', 384))

    def __init__(self, root, features_root):
        self.root = Path(root)
        self.features_root = Path(features_root)
        self.records = {}
        self._cache = lru_cache(maxsize=128)(self._open)
        self._summary = lru_cache(maxsize=384)(self._build_summary)

    def ensure(self, audio):
        if audio in self.records:
            return self.records[audio]
        folder = self.features_root / audio
        receipt = json.loads((folder / 'receipt.json').read_text())
        if receipt.get('complete') is not True:
            raise ValueError('MERT receipt is incomplete: ' + audio)
        samples = int(receipt['metadata']['real_samples'])
        self.records[audio] = dict(duration_us=samples * 1_000_000 // SR, real_samples=samples)
        return self.records[audio]

    def _open(self, audio):
        folder = self.features_root / audio
        self.ensure(audio)
        l12 = np.load(folder / 'L12.npy', mmap_mode='r', allow_pickle=False)
        l23 = np.load(folder / 'L23.npy', mmap_mode='r', allow_pickle=False)
        times = np.load(folder / 'times_s.npy', allow_pickle=False)
        mel = np.load(folder / 'mel10_complete.npy', mmap_mode='r', allow_pickle=False)
        if l12.shape != l23.shape or l12.ndim != 2 or l12.shape[1] != 1024:
            raise ValueError('MERT retained-layer shape changed: ' + audio)
        frames = (self.records[audio]['real_samples']+CELL_SAMPLES-1)//CELL_SAMPLES
        if (len(l12) != frames or times.shape != (frames,) or not np.isfinite(times).all()
                or (np.diff(times) <= 0).any() or mel.shape != (frames*4, 128)):
            raise ValueError('MERT/mel frame clock or coverage changed: ' + audio)
        return l12, l23, times, mel

    def mel_frames(self, audio):
        return self._cache(audio)[3][:math.ceil(self.records[audio]['real_samples']/240)]

    def _build_summary(self, audio):
        _, l23, _, _ = self._cache(audio)
        step = SUMMARY_US // FRAME_US
        frames = l23.shape[0]
        # Bound conversion to chunks instead of one Python allocation per cell
        # or a full-song FP32 temporary. Preserve the original 12-frame recipe.
        cells = np.empty(((frames + step - 1)//step, l23.shape[1]), np.float32)
        block_cells = 64
        for first in range(0, frames//step, block_cells):
            count = min(block_cells, frames//step-first)
            block = np.asarray(l23[first*step:(first+count)*step], np.float32)
            cells[first:first+count] = block.reshape(count, step, -1).mean(axis=1)
        if frames % step:
            cells[-1] = np.asarray(l23[frames//step*step:], np.float32).mean(axis=0)
        return cells

    def window(self, audio, start_us, end_us):
        l12, l23, times, mel = self._cache(audio)
        record = self.records[audio]
        if not 0 <= start_us < end_us <= record['duration_us']:
            raise ValueError('Request lies outside the decoded song')
        frames = len(times)
        lo = max(0, int(start_us // FRAME_US) - 100)
        hi = min(frames, int(start_us // FRAME_US) + 300)
        if hi <= lo:
            raise ValueError('Empty local window')
        local = np.stack([l12[lo:hi], l23[lo:hi]]).astype(np.float32, copy=False)[:, :, None, :]
        local_times = np.asarray(times[lo:hi], np.float64) - start_us/1e6
        mlo, mhi = lo * 4, min(len(mel), hi * 4)
        mel_values = np.asarray(mel[mlo:mhi], np.float32)
        indices = np.arange(mlo, mhi)
        samples = record['real_samples']
        mel_valid = indices*240 < samples
        mel_times = (indices*240+np.minimum((indices+1)*240, samples))/(2*SR)-start_us/1e6
        summary = self._summary(audio)
        step = SUMMARY_US//FRAME_US
        summary_times = np.asarray([times[i:i+step].mean() for i in range(0, len(times), step)])-start_us/1e6
        return dict(
            local=local,
            local_times=local_times.astype(np.float32),
            local_valid=np.ones(hi - lo, bool),
            song=summary,
            song_times=np.asarray(summary_times, np.float32),
            song_valid=np.ones(len(summary), bool),
            mel=mel_values,
            mel_times=mel_times.astype(np.float32),
            mel_valid=mel_valid)
