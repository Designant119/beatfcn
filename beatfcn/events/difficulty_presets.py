"""Simple difficulty presets: Phira level (HD 8 .. AT 16) or osu!mania star rating -> a song-level density request.

The model's difficulty control is a density level (onset groups = "heads" per second); with the round-5
level-only training a constant request sets the overall level and the song keeps its own structure.
The tables come from the database (scripts/build_difficulty_presets.py): the median heads/s of the charts at
each Phira catalog difficulty / approximate mania star, made monotonic (pool-adjacent-violators) and
interpolated. Star rating here is an APPROXIMATION of osu!mania's legacy strain algorithm (individual +
overall strain, 400 ms sections, 0.9 peak weighting, x0.018), not the live osu! value.
"""
import json
from pathlib import Path

import numpy as np

INDIVIDUAL_DECAY, OVERALL_DECAY = 0.125, 0.30
STRAIN_STEP_MS, DECAY_WEIGHT, STAR_SCALING = 400.0, 0.9, 0.018


def mania_star(notes, keys=4):
    """Approximate osu!mania (legacy) star rating of [(onset_us, end_us, type, lane), ...]."""
    notes = sorted((int(a) / 1000.0, int(b) / 1000.0, int(l)) for a, b, _, l in notes if 0 <= int(l) < keys)
    if len(notes) < 2:
        return 0.0
    hold_end = [0.0] * keys
    indiv = [0.0] * keys
    overall = 1.0
    prev = notes[0][0]
    strains = []
    for start, end, col in notes[1:]:
        dt = max(start - prev, 0.0)
        hold_factor, hold_add = 1.0, 0.0
        for i in range(keys):
            if start < hold_end[i] < end:
                hold_add = 1.0
            if end == hold_end[i]:
                hold_add = 0.0
            if hold_end[i] > end:
                hold_factor = 1.25
        hold_end[col] = end
        decay_i = INDIVIDUAL_DECAY ** (dt / 1000.0)
        indiv = [v * decay_i for v in indiv]
        indiv[col] += 2.0 * hold_factor
        overall = overall * OVERALL_DECAY ** (dt / 1000.0) + (1.0 + hold_add) * hold_factor
        strains.append((start, indiv[col] + overall))
        prev = start
    t0 = notes[0][0]
    peaks = {}
    for t, s in strains:
        k = int((t - t0) // STRAIN_STEP_MS)
        peaks[k] = max(peaks.get(k, 0.0), s)
    vals = sorted(peaks.values(), reverse=True)
    difficulty = sum(v * DECAY_WEIGHT ** i for i, v in enumerate(vals))
    return difficulty * STAR_SCALING


def monotone(xs, ys, ws):
    """Weighted pool-adjacent-violators (non-decreasing)."""
    blocks = [[x, y, w] for x, y, w in zip(xs, ys, ws)]
    out = []
    for b in blocks:
        out.append([b[0], b[0], b[1], b[2]])
        while len(out) > 1 and out[-2][2] > out[-1][2]:
            a, c = out[-2], out[-1]
            w = a[3] + c[3]
            out[-2:] = [[a[0], c[1], (a[2] * a[3] + c[2] * c[3]) / w, w]]
    fx, fy = [], []
    for lo, hi, y, _ in out:
        for x in xs:
            if lo <= x <= hi:
                fx.append(x); fy.append(y)
    return fx, fy


class DifficultyPresets:
    def __init__(self, table):
        self.table = table

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    @classmethod
    def bundled(cls):
        from beatfcn.events.difficulty_presets_table import TABLE
        return cls(TABLE)

    def source(self, mode):
        key = 'phira_level' if mode == 'phira' else 'mania_star'
        how = ('weighted log-linear fit of the per-level median' if key == 'phira_level'
               else 'monotone (PAVA) per-0.5-star median; approximate legacy star')
        return f"{self.table['schema']} {key}: {how} heads/s of {self.table['source']}"

    def heads_per_s(self, mode, value):
        key = 'phira_level' if mode == 'phira' else 'mania_star'
        xs, ys = self.table[key]['x'], self.table[key]['heads_per_s']
        return float(np.interp(float(value), xs, ys))
