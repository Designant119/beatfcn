"""Density from explicitly authorized reference charts; absent if none is supplied.

Training-target-derived density is valid supervision. Evaluation requests must be
declared independently; audio energy fallback is confined to legacy v1 replay.
"""
from collections import defaultdict
import math

import numpy as np

from beatfcn.train.intensity_panel import IntensityPanel
from beatfcn.train.density_curve import absent_curve, density_grid, requested_density_curve


class ReferenceIntensityPanel(IntensityPanel):
    def __init__(self, panel, *, source='chart', fallback=None,
                 authorized_chart_ids=(), **settings):
        if fallback is None:
            fallback = 'audio' if settings.get('version', 2) == 1 else 'unspecified'
        if source not in ('chart', 'audio'):
            raise ValueError('Reference intensity source must be chart or audio')
        if fallback not in ('audio', 'zero', 'unspecified'):
            raise ValueError('Reference intensity fallback must be audio or zero')
        if settings.get('version', 2) == 2 and (source == 'audio' or fallback != 'unspecified'):
            raise ValueError('v2 density cannot substitute audio energy or implicit zero; use unspecified')
        self.source = source
        self.fallback = fallback
        self.authorized_chart_ids = frozenset(authorized_chart_ids)
        self.donors = self._build_donors(panel, self.authorized_chart_ids
            if settings.get('version', 2) == 2 else None)
        super().__init__(panel, **settings)
        self.sources = {}

    @staticmethod
    def _build_donors(panel, allowed=None):
        groups = defaultdict(list)
        seen = set()
        for index, row in enumerate(panel.records):
            key = (row['audio_sha256'], row['chart_id'])
            if key not in seen:
                groups[row['audio_sha256']].append(index)
                seen.add(key)
        donors = {}
        choices = {}
        for index, row in enumerate(panel.records):
            key = (row['audio_sha256'], row['chart_id'], row['mode'], row['stratum'])
            if key in choices:
                if choices[key] is not None:
                    donors[index] = choices[key]
                continue
            candidates = [other for other in groups[row['audio_sha256']]
                          if panel.records[other]['chart_id'] != row['chart_id']
                          and (allowed is None or panel.records[other]['chart_id'] in allowed)]
            if not candidates:
                choices[key] = None
                continue
            # Prefer a different mode/stratum, then the nearest chart id.
            candidates.sort(key=lambda other: (
                panel.records[other]['mode'] == row['mode'],
                panel.records[other]['stratum'] == row['stratum'],
                panel.records[other]['chart_id']))
            donors[index] = candidates[0]
            choices[key] = candidates[0]
        return donors

    def _chart_curve(self, donor_index, start, end, cells):
        row = self.records[donor_index]
        try:
            _, values = self.base._arrays(row['chart_index'])
            onsets = values['onset_us']
        except (AttributeError, KeyError):
            events = self.base.targets(donor_index)
            onsets = np.asarray([event.onset_us for event in events], np.int64)
        counts = np.zeros(cells, np.float64)
        for onset in np.asarray(onsets):
            onset = int(onset)
            if start <= onset < end:
                counts[(onset - start) // self.cell_us] += 1
        return counts

    def _audio_curve(self, row, start, end, cells):
        try:
            window = self.base.features.window(row['audio_sha256'], start, end)
        except (FileNotFoundError, KeyError):
            return None
        mel = np.asarray(window['mel'], np.float32)
        # Feature windows include left/right context. Use their relative clocks,
        # rather than assuming frame zero is the start of the requested region.
        times = np.asarray(window['mel_times'])
        valid = np.asarray(window['mel_valid'], bool)
        energy = np.zeros(cells, np.float64)
        for cell in range(cells):
            lo, hi = cell * self.cell_us / 1e6, (cell+1) * self.cell_us / 1e6
            block = mel[valid & (times >= lo) & (times < min(hi, (end-start)/1e6))]
            if len(block):
                energy[cell] = float(np.abs(block).mean())
        peak = float(energy.max())
        return energy / peak if peak > 0 else energy

    def _curve(self, index):
        row = self.records[index]
        start = int(row['request']['start_us'])
        end = int(row['request']['end_us'])
        cells = max(1, math.ceil((end - start) / self.cell_us))
        counts = None
        if self.source == 'chart' and index in self.donors:
            counts = self._chart_curve(self.donors[index], start, end, cells)
            if self.version == 2:
                edges = density_grid(start, end, self.cell_us)
                self.sources[index] = f"chart:{self.records[self.donors[index]]['chart_id']}"
                segments = [dict(start_us=int(a), end_us=int(b), density=float(n/((b-a)/1e6)))
                    for a, b, n in zip(edges[:-1], edges[1:], counts)]
                return requested_density_curve(segments, start, end, scale=self.scale)
            values = np.clip(counts / (self.cell_us / 1_000_000.0) / self.scale, 0.0, 1.0)
            self.sources[index] = f"chart:{self.records[self.donors[index]]['chart_id']}"
        else:
            if self.version == 2:
                self.sources[index] = 'unspecified'
                return absent_curve(start, end, self.cell_us)
            energy = self._audio_curve(row, start, end, cells)
            if energy is not None and self.fallback == 'audio':
                values = np.clip(energy, 0.0, 1.0)
                self.sources[index] = 'audio'
            else:
                values = np.zeros(cells, np.float64)
                self.sources[index] = 'zero'
        times = (np.arange(cells) * self.cell_us + self.cell_us / 2) / 1e6
        return dict(intensity=values[:, None].astype(np.float32),
                    intensity_times=times.astype(np.float32),
                    intensity_valid=np.ones(cells, bool))
