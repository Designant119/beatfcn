"""Density controls in heads/second, shared by training and caller requests."""
import math

import numpy as np


def encode_density(values, scale=16.):
    values = np.asarray(values, np.float64)
    if not math.isfinite(scale) or scale <= 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Density must be finite and nonnegative; scale must be positive')
    return (np.log1p(values)/math.log1p(scale)).astype(np.float32)


def density_grid(start_us, end_us, cell_us):
    if not 0 <= start_us < end_us or type(cell_us) is not int or cell_us < 1000:
        raise ValueError('Invalid density curve clock')
    edges = np.arange(start_us, end_us, cell_us, dtype=np.int64)
    return np.append(edges, end_us)


def curve_values(values, times, valid):
    values = np.asarray(values, np.float32)
    if values.ndim == 1:
        values = values[:, None]
    return dict(intensity=values,
        intensity_times=np.asarray(times, np.float32), intensity_valid=np.asarray(valid, bool))


def absent_curve(start_us, end_us, cell_us=500_000):
    edges = density_grid(start_us, end_us, cell_us)
    values = np.stack((np.zeros(len(edges)-1), (edges[:-1]-start_us)/1e6,
        (edges[1:]-start_us)/1e6), axis=-1)
    return curve_values(values, (edges[:-1]+edges[1:])/2e6-start_us/1e6,
        np.zeros(len(edges)-1, bool))


def target_density_curve(onsets, start_us, end_us, *, cell_us=500_000, scale=16.):
    edges = density_grid(start_us, end_us, cell_us)
    onsets = np.asarray(onsets, np.int64)
    inside = onsets[(onsets >= start_us) & (onsets < end_us)]
    counts = np.histogram(inside, bins=edges)[0]
    rates = counts/(np.diff(edges)/1e6)
    segments = [dict(start_us=int(a), end_us=int(b), density=float(rate))
        for a, b, rate in zip(edges[:-1], edges[1:], rates)]
    return requested_density_curve(segments, start_us, end_us, scale=scale)


def requested_density_curve(segments, start_us, end_us, *, scale=16.):
    """Absolute half-open caller segments; gaps mean unspecified, zero means silence.

    Segments are not inferred from a hidden target. Keep each user boundary rather
    than resampling it onto a coarse grid. Overlapping requests are rejected.
    """
    density_grid(start_us, end_us, 500_000)
    values, times = [], []
    previous_end = -1
    for segment in segments:
        lo, hi = segment['start_us'], segment['end_us']
        if type(lo) is not int or type(hi) is not int or lo < 0 or hi <= lo or lo < previous_end:
            raise ValueError('Density segments must be ordered, nonoverlapping integer intervals')
        previous_end = hi
        value = float(encode_density([segment['density']], scale)[0])
        a, b = max(lo, start_us), min(hi, end_us)
        if a < b:
            values.append((value, (a-start_us)/1e6, (b-start_us)/1e6))
            times.append(((a+b)/2-start_us)/1e6)
    if not values:
        return absent_curve(start_us, end_us)
    return curve_values(values, times, np.ones(len(values), bool))

def grid_density_curve(segments, start_us, end_us, *, cell_us=100_000, scale=16.):
    """Resample caller segments to a fixed model grid.

    The model needs one token per cell; keeping a constant request as a single
    token makes the condition much weaker than the 100 ms training curve.
    Gaps stay explicitly unspecified (valid=False), never implicit zero.
    """
    edges = density_grid(start_us, end_us, cell_us)
    values, times, valid = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        center = (int(a)+int(b))/2
        segment = next((segment for segment in segments
                        if segment['start_us'] <= center < segment['end_us']), None)
        if segment is None:
            values.append((0., (a-start_us)/1e6, (b-start_us)/1e6))
            valid.append(False)
        else:
            values.append((float(encode_density([segment['density']], scale)[0]),
                           (a-start_us)/1e6, (b-start_us)/1e6))
            valid.append(True)
        times.append(((a+b)/2-start_us)/1e6)
    return curve_values(values, times, valid)
