"""Versioned density controls: explicit requests or training-only derived labels.

v2 uses non-saturating log density and an explicit presence mask. Hidden evaluation
targets are never read unless oracle_validation is selected for diagnostics.
"""
from functools import lru_cache
import hashlib
import math

import numpy as np

from beatfcn.train.local_panel import collate
from beatfcn.train.worker_cache import WorkerCacheMixin
from beatfcn.train.density_curve import (absent_curve, target_density_curve, requested_density_curve,
    grid_density_curve)


class IntensityPanel(WorkerCacheMixin):
    """Inject 'intensity' / 'intensity_times' / 'intensity_valid' into panel rows."""
    worker_cache_specs = (('_curves', '_curve', 4096),)

    def __init__(self, panel, *, cell_us=500_000, scale=16.0, hide_probability=0.0, seed=0,
                 version=2, oracle_validation=False, requested_curves=None):
        if cell_us < 1000 or not math.isfinite(scale) or scale <= 0 or not 0 <= hide_probability <= 1:
            raise ValueError('Invalid intensity panel contract')
        if version not in (1, 2):
            raise ValueError('Unknown curve version')
        self.version = version
        self.oracle_validation = bool(oracle_validation)
        self.requested_curves = requested_curves or {}
        self.base = panel
        self.cell_us = cell_us
        self.scale = float(scale)
        self.hide_probability = float(hide_probability)
        self.seed = int(seed)
        self._curves = lru_cache(maxsize=4096)(self._curve)

    def __getattr__(self, name):
        base = self.__dict__.get('base')
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def _curve(self, index):
        row = self.records[index]
        start = int(row['request']['start_us'])
        end = int(row['request']['end_us'])
        curve_key = None
        if str(index) in self.requested_curves:
            curve_key = str(index)
        elif row.get('chart_id') in self.requested_curves:
            curve_key = row['chart_id']
        if self.version == 2 and curve_key is not None:
            return grid_density_curve(self.requested_curves[curve_key], start, end,
                cell_us=self.cell_us, scale=self.scale)
        if self.version == 2 and row['split'] != 'train' and not self.oracle_validation:
            return absent_curve(start, end, self.cell_us)
        events = self.base.targets(index)
        if self.version == 2:
            return target_density_curve([e.onset_us for e in events], start, end,
                cell_us=self.cell_us, scale=self.scale)
        cells = max(1, math.ceil((end - start) / self.cell_us))
        counts = np.zeros(cells, np.float32)
        for event in events:
            onset = int(event.onset_us)
            if start <= onset < end:
                counts[(onset - start) // self.cell_us] += 1
        values = np.clip(counts / (self.cell_us / 1_000_000.0) / self.scale, 0.0, 1.0)
        times = (np.arange(cells) * self.cell_us + self.cell_us / 2) / 1e6
        return dict(intensity=values[:, None].astype(np.float32),
                    intensity_times=times.astype(np.float32),
                    intensity_valid=np.ones(cells, bool))

    def _hidden(self, index):
        if not self.hide_probability or self.records[index].get('split') != 'train':
            return False
        digest = hashlib.blake2b(f'{self.seed}:{index}'.encode(), digest_size=8).digest()
        return int.from_bytes(digest, 'big') / 2**64 < self.hide_probability

    def _row_curve(self, index):
        curve = dict(self._curves(index))
        if self._hidden(index):
            curve['intensity'] = np.zeros_like(curve['intensity'])
            if self.version == 2:
                curve['intensity_valid'] = np.zeros_like(curve['intensity_valid'])
        return curve

    def row(self, index, *, controls=None):
        record = self.base.row(index, controls=controls)
        record['inputs'].update(self._row_curve(index))
        return record

    def batch(self, indices, *, controls=None, device='cpu'):
        inputs, sequences = collate([self.row(i, controls=None if controls is None
            else controls[j]) for j, i in enumerate(indices)])
        return ({k: v.to(device) for k, v in inputs.items()},
                {k: v.to(device) for k, v in sequences.items()})

    def curve_for(self, index):
        return self._row_curve(index)
