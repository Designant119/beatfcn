"""Another human chart of the same song as a visible encoder reference (difficulty spread).

Training contract (plan key ``alternative_context``, OFF unless set): for a native
(rate None) training request whose song has another TRAIN chart of the same mode
(a different chart id and a different canonical chart), with probability ``p``
the encoder reference stream also shows that sibling chart's events with onsets in
[start - pad, end + pad) (clipped to the song; holds with their full tails, as
event_reference.VisibleReference requires) under role ``human_alternative``.
With probability ``denser`` the sibling is drawn among the charts with more
distinct onsets than the target (downmapping: the charter's harder chart -> this
one), else among the others; when only one side exists, that side is used.
Within the side, with probability ``nearest`` the chart closest in distinct onsets
is shown (the adjacent difficulty: the realistic spread case and the clearest
derive-this-from-that signal), else one drawn uniformly.

The target, its density curve / controls and their frozen dropout are unchanged,
so the model learns "derive THIS chart (what the conditions ask for) from that
chart". At inference the charter's own chart is the sibling (spread benchmark /
draft tool), never a hidden human answer.

Leakage contract: the sibling is a different human chart, not hidden truth of the
target, so it is the only reference role that may overlap the target window
(event_reference.encode_reference). Exact duplicate charts (same canonical sha, or
identical arrays) are never shown. Edited-density targets never get an
alternative. Draws are a pure function of (seed, request index, draw key =
exposure epoch/occurrence), independent of history_context draws.
"""
import math

import numpy as np

from beatfcn.train.history_context import chart_arrays, visible_source

ALTERNATIVE_TAG = 707
FORMAT = 'alternative-context-v1'
ROLE = 'human_alternative'
DEFAULT_ALTERNATIVE_CONTEXT = dict(p=.35, denser=.7, nearest=.6, pad_s=2.0, max_reference_rows=2048)


def normalize_alternative_context(settings):
    """None/False -> None (OFF). True or a dict -> validated settings (defaults merged)."""
    if settings is None or settings is False:
        return None
    if settings is True:
        settings = {}
    if not isinstance(settings, dict):
        raise ValueError('alternative_context must be a dict, true, or absent')
    unknown = set(settings) - set(DEFAULT_ALTERNATIVE_CONTEXT)
    if unknown:
        raise ValueError(f'Unknown alternative_context keys {sorted(unknown)}')
    out = dict(DEFAULT_ALTERNATIVE_CONTEXT, **settings)
    for key in ('p', 'denser', 'nearest'):
        if not (isinstance(out[key], (int, float)) and 0 <= out[key] <= 1):
            raise ValueError(f'alternative_context.{key} must lie in [0, 1]')
        out[key] = float(out[key])
    if not (isinstance(out['pad_s'], (int, float)) and math.isfinite(out['pad_s']) and 0 <= out['pad_s'] <= 30):
        raise ValueError('alternative_context.pad_s must be finite seconds in [0, 30]')
    out['pad_s'] = float(out['pad_s'])
    if type(out['max_reference_rows']) is not int or out['max_reference_rows'] < 4:
        raise ValueError('alternative_context.max_reference_rows must be an integer >= 4')
    return out


def draw_alternative(settings, seed, request, key):
    """Deterministic choice for one (request, draw key). Always consumes 3 uniforms."""
    u = np.random.default_rng([int(seed) & 0xFFFFFFFF, ALTERNATIVE_TAG,
                               int(request) & 0xFFFFFFFF, int(key) & 0xFFFFFFFF]).random(3)
    return dict(applied=bool(u[0] < settings['p']), denser=bool(u[1] < settings['denser']), pick=float(u[2]))


class SiblingIndex:
    """(audio, mode) -> chart indices of ONE split of a panel (tape order; training uses 'train'),
    with lazy per-chart head counts."""

    def __init__(self, panel, split='train'):
        self.panel = panel
        self.split = split
        groups = {}
        for index, chart in enumerate(panel.charts):
            if chart.get('split') == split:
                groups.setdefault((chart['audio_sha256'], chart['mode']), []).append(index)
        self.groups = {k: tuple(v) for k, v in groups.items() if len(v) >= 2}
        self._heads = {}

    def songs_with_siblings(self):
        return len(self.groups)

    def heads(self, chart_index):
        if chart_index not in self._heads:
            self._heads[chart_index] = int(len(np.unique(chart_arrays(self.panel, chart_index)['onset_us'])))
        return self._heads[chart_index]

    def _duplicate(self, a, b):
        ca, cb = (self.panel.charts[i].get('canonical_sha256') for i in (a, b))
        if ca is not None and cb is not None:
            return ca == cb
        x, y = chart_arrays(self.panel, a), chart_arrays(self.panel, b)
        return all(len(x[k]) == len(y[k]) and np.array_equal(x[k], y[k]) for k in ('onset_us', 'end_us', 'kind', 'lane'))

    def siblings(self, chart_index):
        chart = self.panel.charts[chart_index]
        group = self.groups.get((chart['audio_sha256'], chart['mode']), ())
        return [j for j in group if j != chart_index and self.panel.charts[j]['chart_id'] != chart['chart_id']
                and not self._duplicate(chart_index, j)]


def alternative_source(index, request, record, draw, settings):
    """(VisibleReference or None, diagnostics record) for one drawn alternative choice."""
    info = dict(format=FORMAT, drawn=dict(draw), applied=False, reason=None, sibling=None, direction=None,
                region_us=None, events=0, window_events=0, target_heads=None, sibling_heads=None)
    if not draw['applied']:
        info['reason'] = 'not_drawn'
        return None, info
    chart_index = record['chart_index']
    siblings = index.siblings(chart_index)
    if not siblings:
        info['reason'] = 'no_sibling'
        return None, info
    mine = index.heads(chart_index)
    denser = [j for j in siblings if index.heads(j) > mine]
    other = [j for j in siblings if index.heads(j) <= mine]
    side, direction = ((denser, 'denser') if (draw['denser'] and denser) or not other else (other, 'sparser_or_equal'))
    near, pick = settings['nearest'], draw['pick']
    if pick < near:
        j = min(side, key=lambda c: (abs(index.heads(c) - mine), c))
        info['choice'] = 'nearest'
    else:
        u = (pick - near) / (1 - near) if near < 1 else 0.
        j = side[min(len(side) - 1, int(u * len(side)))]
        info['choice'] = 'uniform'
    pad = int(round(settings['pad_s'] * 1e6))
    region = (max(0, request.start_us - pad), min(request.duration_us, request.end_us + pad))
    sibling_id = index.panel.charts[j]['chart_id']
    source = visible_source(chart_arrays(index.panel, j), mode=request.mode, audio_sha256=record['audio_sha256'],
                            role=ROLE, region=region, source_id=f'alternative:{sibling_id}')
    if len(source.events) + 2 > settings['max_reference_rows']:
        info.update(reason='over_reference_capacity', sibling=sibling_id, direction=direction)
        return None, info
    info.update(applied=True, sibling=sibling_id, direction=direction, region_us=list(region),
                events=len(source.events), target_heads=mine, sibling_heads=index.heads(j),
                window_events=sum(request.start_us <= e.onset_us < request.end_us for e in source.events))
    return source, info


def check_alternative(sources, record):
    """Raise unless every alternative source is a human_alternative view of ANOTHER chart."""
    own = f"alternative:{record.get('chart_id')}"
    for source in sources:
        if source.role != ROLE or not source.source_id.startswith('alternative:') or source.source_id == own:
            raise ValueError('Alternative reference must be another human chart under role human_alternative')


def reference_inputs_with_alternative(request, record, history_sources, alternative_sources, *, max_rows=4096):
    """One reference input from same-chart history (outside the window, history_context
    leak rules) plus sibling views (role human_alternative, may overlap the window)."""
    from beatfcn.train.event_reference import encode_reference
    from beatfcn.train.history_context import check_no_leak
    check_no_leak(history_sources, request)
    check_alternative(alternative_sources, record)
    sources = tuple(history_sources) + tuple(alternative_sources)
    reference, provenance = encode_reference(mode=request.mode, audio_sha256=record['audio_sha256'],
        duration_us=request.duration_us, origin_us=request.start_us,
        unknown_regions=((request.start_us, request.end_us),), sources=sources, max_rows=max_rows)
    for row in provenance:
        if (row['event_id'] is not None and request.start_us <= row['interval_us'][0] < request.end_us
                and row['role'] != ROLE):
            raise ValueError('Encoded reference contains a target-window event of the target chart')
    return reference, provenance


def chart_arrays_from_notes(notes):
    """[(onset_us, end_us, type, lane), ...] (workflow benchmark reference notes) -> chart arrays."""
    from beatfcn.events.canonical import TYPES
    rows = sorted((int(a), int(b), TYPES.index(t), int(l)) for a, b, t, l in notes)
    return {k: np.asarray([r[i] for r in rows], np.int64) for i, k in enumerate(('onset_us', 'end_us', 'kind', 'lane'))}


def sibling_provider(charts, *, pad_s=DEFAULT_ALTERNATIVE_CONTEXT['pad_s']):
    """local_song_rollout alternative_provider: per core, the charter's own chart of that song
    (``charts[state chart_id] = (source chart id, chart arrays on the model clock)``) in
    [start - pad, end + pad) under role human_alternative; None for songs without one."""
    pad = int(round(float(pad_s) * 1e6))

    def provider(state, request):
        entry = charts.get(state['chart_id'])
        if entry is None:
            return None
        chart_id, arrays = entry
        region = (max(0, request.start_us - pad), min(request.duration_us, request.end_us + pad))
        return visible_source(arrays, mode=request.mode, audio_sha256=state['audio_sha256'], role=ROLE,
                              region=region, source_id=f'alternative:{chart_id}')
    return provider
