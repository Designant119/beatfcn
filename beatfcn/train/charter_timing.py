"""Optional charter-timing (beat grid) input: table, per-frame beat features, dropout.

A charter sets BPM/offset (red lines) before placing notes, so the chart's own
timing points are a legitimate, OPTIONAL model input (FullEventConfig.beat_features).
Every readout that uses it is labelled ``charter_timing_input``; the model must
still work without it (absent state = all-zero features incl. presence 0).

Table (scripts/build_chart_timing.py): chart_id -> segments [(start_us, beat_us,
meter), ...] already on the MODEL clock (raw chart grid + the target's
model-clock shift), plus validity. Only ``valid`` charts are ever served; an
invalid / missing chart gets the absent state.

Per 10 ms mel frame (absolute song time t), segment s governing t (the first
segment extends backwards, osu rule), local beat position x = (t - start_s) /
beat_s:
  0-1  sin/cos(2 pi x)            beat phase
  2-3  sin/cos(2 pi x / meter)    bar phase (bars anchored at each red line, osu rule)
  4-5  sin/cos(4 pi x)            2x beat frequency (1/2)
  6-7  sin/cos(8 pi x)            4x beat frequency (1/4)
  8    signed offset (ms) from the nearest 1/4 point, clipped to +-25 ms, / 25
  9    signed offset (ms) from the nearest 1/3 point, clipped to +-25 ms, / 25
       (positive = the frame is after the grid point)
  10   log(beat_ms / 500) clipped to [-3, 3]   (beat length; 0 = 120 BPM)
  11   presence flag (1)
Frames whose mel row is invalid (padding / outside the audio) are all zeros.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from beatfcn.model.full_event_model import BEAT_FEATURE_DIM

SCHEMA = 'chart-timing-table-v1'
BEAT_FEATURE_NAMES = ('beat_sin', 'beat_cos', 'bar_sin', 'bar_cos', 'beat2_sin', 'beat2_cos',
                      'beat4_sin', 'beat4_cos', 'offset_quarter', 'offset_third', 'log_beat', 'present')
OFFSET_CLIP_MS = 25.
LOG_BEAT_CLIP = 3.
DEFAULT_DROPOUT = .3
LABEL = 'charter_timing_input'
if len(BEAT_FEATURE_NAMES) != BEAT_FEATURE_DIM:
    raise ImportError('Beat feature names and model input width disagree')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def segment_array(segments):
    """[(start_us, beat_us, meter), ...] -> float64 [S, 3] sorted by start (stable), or None."""
    if segments is None:
        return None
    seg = np.asarray(segments, np.float64).reshape(-1, 3)
    if not len(seg):
        return None
    if not np.all(np.isfinite(seg)) or np.any(seg[:, 1] <= 0) or np.any(seg[:, 2] < 1):
        raise ValueError('Charter timing segments need finite start, beat_us > 0 and meter >= 1')
    return seg[np.argsort(seg[:, 0], kind='stable')]


def beat_features(times_us, segments):
    """float32 [T, BEAT_FEATURE_DIM] for absolute song times (us); all zeros when segments is None."""
    t = np.asarray(times_us, np.float64).reshape(-1)
    out = np.zeros((len(t), BEAT_FEATURE_DIM), np.float32)
    seg = segment_array(segments)
    if seg is None or not len(t):
        return out
    starts, beat, meter = seg[:, 0], seg[:, 1], seg[:, 2]
    i = np.clip(np.searchsorted(starts, t, 'right')-1, 0, len(seg)-1)
    b = beat[i]
    x = (t-starts[i])/b
    two_pi = 2*np.pi
    columns = []
    for angle in (two_pi*x, two_pi*x/meter[i], 2*two_pi*x, 4*two_pi*x):
        columns += [np.sin(angle), np.cos(angle)]
    for k in (4, 3):
        offset_ms = (x*k-np.round(x*k))*b/k/1000.
        columns.append(np.clip(offset_ms, -OFFSET_CLIP_MS, OFFSET_CLIP_MS)/OFFSET_CLIP_MS)
    columns.append(np.clip(np.log(b/500_000.), -LOG_BEAT_CLIP, LOG_BEAT_CLIP))
    columns.append(np.ones(len(t)))
    out[:] = np.stack(columns, 1)
    return out


def window_beat_features(inputs, start_us, segments):
    """``beat`` [T_mel, F] for one request's inputs, aligned to ``mel_times`` (s, relative to start_us)."""
    times = np.asarray(inputs['mel_times'], np.float64)*1e6+float(start_us)
    beat = beat_features(times, segments)
    if 'mel_valid' in inputs:
        beat[~np.asarray(inputs['mel_valid'], bool)] = 0
    return beat


def timing_dropped(seed, request_index, key, rate):
    """Deterministic per (seed, request, exposure epoch/occurrence) Bernoulli(rate) draw."""
    if not rate:
        return False
    digest = hashlib.sha256(f'charter-timing:{int(seed)}:{int(request_index)}:{key}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big')/2**64 < float(rate)


def normalize_charter_timing(settings):
    """Plan key ``charter_timing`` = {path, sha256 (optional), dropout (default .3)} -> normalized dict or None."""
    if not settings:
        return None
    if isinstance(settings, (str, Path)):
        settings = dict(path=str(settings))
    settings = dict(settings)
    unknown = set(settings) - {'path', 'sha256', 'dropout'}
    if unknown or not settings.get('path'):
        raise ValueError(f'charter_timing needs a path; unknown keys {sorted(unknown)}')
    dropout = float(settings.get('dropout', DEFAULT_DROPOUT))
    if not 0 <= dropout <= 1:
        raise ValueError('charter_timing dropout must be within [0, 1]')
    return dict(path=str(settings['path']), sha256=settings.get('sha256'), dropout=dropout)


class ChartTimingTable:
    """Read-only chart_id -> valid segments lookup over a build_chart_timing.py table."""

    def __init__(self, document, *, path=None, sha256=None):
        if document.get('schema') != SCHEMA:
            raise ValueError(f'Not a {SCHEMA} table')
        self.document = document
        self.path, self.sha256 = (None if path is None else str(path)), sha256
        self.charts = document['charts']
        self._cache = {}

    @classmethod
    def load(cls, path, *, sha256=None, root=None):
        path = Path(path)
        if root is not None and not path.is_absolute():
            path = Path(root)/path
        digest = file_sha(path)
        if sha256 and digest != sha256:
            raise ValueError(f'Charter timing table identity changed: {path}')
        return cls(json.loads(path.read_text()), path=path, sha256=digest)

    @classmethod
    def from_settings(cls, settings, *, root=None):
        settings = normalize_charter_timing(settings)
        return None if settings is None else cls.load(settings['path'], sha256=settings['sha256'], root=root)

    def status(self, chart_id):
        entry = self.charts.get(chart_id)
        return 'missing' if entry is None else entry['status']

    def segments(self, chart_id):
        """float64 [S, 3] for a VALID chart, else None (invalid / missing -> absent state)."""
        if chart_id not in self._cache:
            entry = self.charts.get(chart_id)
            self._cache[chart_id] = (segment_array(entry['segments'])
                                     if entry is not None and entry.get('valid') else None)
        return self._cache[chart_id]

    def identity(self):
        return dict(path=self.path, sha256=self.sha256, schema=SCHEMA, label=LABEL,
                    coverage=self.document.get('coverage', {}).get('overall'))


def attach_window_beat(inputs, table, chart_id, start_us, *, dropped=False):
    """Set ``inputs['beat']`` for one request; returns the audit record."""
    status = table.status(chart_id) if table is not None else 'no_table'
    segments = None if dropped or table is None else table.segments(chart_id)
    inputs['beat'] = window_beat_features(inputs, start_us, segments)
    reason = 'present' if segments is not None else 'dropout' if dropped and status == 'valid' else status
    return dict(label=LABEL, present=segments is not None, reason=reason, chart_id=chart_id,
                segments=0 if segments is None else int(len(segments)))


def evaluation_table(plan, model, *, root=None, choice=None):
    """Table for in-training evaluation screens, or None.

    Used only when the plan has ``charter_timing`` AND the model has beat_features;
    ``choice`` ('with' | 'without', default 'with') lets a screen run without it."""
    choice = choice or 'with'
    if choice not in ('with', 'without'):
        raise ValueError("charter_timing screen choice must be 'with' or 'without'")
    if choice == 'without' or not plan.get('charter_timing') or not getattr(model.c, 'beat_features', False):
        return None
    settings = normalize_charter_timing(plan['charter_timing'])
    key = (str(root), settings['path'], settings['sha256'])
    if key not in _EVALUATION_TABLES:  # screens repeat every N updates; parse the table once
        _EVALUATION_TABLES[key] = ChartTimingTable.from_settings(settings, root=root)
    return _EVALUATION_TABLES[key]


_EVALUATION_TABLES = {}
