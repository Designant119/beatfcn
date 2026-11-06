"""Visible human context (previous/next chart regions) as encoder references.

Training contract (plan key ``history_context``, OFF unless set): for a native
(rate None) training request, with probability ``p`` the encoder reference
stream also shows the SAME chart's events in a prefix region
[start - L, start) (L drawn from ``lengths_s``, clipped at song start; no
prefix when start == 0) under a role drawn from ``roles`` (default
human_input .7 / model_history .3, so whole-song rollout's own-output history
is in distribution). With probability ``p_infill`` of those samples a suffix
region [end, end + L') (role ``suffix_role``) is shown too.

Leakage contract: only events whose onset lies OUTSIDE the target window are
shown. A shown hold may cross into the window only if it is exactly one of the
request's ``known_carry`` holds (which the request already authorizes); its
full tail is shown, as event_reference.VisibleReference requires. Draws are a
pure function of (seed, request index, draw key = exposure epoch/occurrence).
Edited-density targets never get history (a native history next to a density
edited target would teach a mismatched continuation).
"""
import math

import numpy as np

from beatfcn.events.canonical import Event, LaneLoc, LineLoc, TYPES
from beatfcn.train.event_reference import SOURCE_COLUMN, encode_reference

HISTORY_TAG = 606
FORMAT = 'history-context-v1'
DEFAULT_HISTORY_CONTEXT = dict(p=.5, p_infill=.25, lengths_s=(4, 8, 16, 32),
                               suffix_lengths_s=None, roles={'human_input': .7, 'model_history': .3},
                               suffix_role='human_input', max_reference_rows=2048)


def normalize_history_context(settings):
    """None/False -> None (OFF). True or a dict -> validated settings (defaults merged)."""
    if settings is None or settings is False:
        return None
    if settings is True:
        settings = {}
    if not isinstance(settings, dict):
        raise ValueError('history_context must be a dict, true, or absent')
    unknown = set(settings) - set(DEFAULT_HISTORY_CONTEXT)
    if unknown:
        raise ValueError(f'Unknown history_context keys {sorted(unknown)}')
    out = dict(DEFAULT_HISTORY_CONTEXT, **settings)
    for key in ('p', 'p_infill'):
        if not (isinstance(out[key], (int, float)) and 0 <= out[key] <= 1):
            raise ValueError(f'history_context.{key} must lie in [0, 1]')
        out[key] = float(out[key])
    for key in ('lengths_s', 'suffix_lengths_s'):
        if out[key] is None:
            continue
        values = [float(v) for v in out[key]]
        if not values or any(not math.isfinite(v) or v <= 0 for v in values) or len(set(values)) != len(values):
            raise ValueError(f'history_context.{key} must be unique positive seconds')
        out[key] = sorted(values)
    if out['suffix_lengths_s'] is None:
        out['suffix_lengths_s'] = list(out['lengths_s'])
    roles = {str(k): float(v) for k, v in dict(out['roles']).items()}
    if (not roles or set(roles) - set(SOURCE_COLUMN) or any(v < 0 for v in roles.values())
            or abs(sum(roles.values()) - 1) > 1e-9):
        raise ValueError('history_context.roles must be reference roles with probabilities summing to one')
    out['roles'] = dict(sorted(roles.items()))
    if out['suffix_role'] not in SOURCE_COLUMN:
        raise ValueError('history_context.suffix_role must be a reference role')
    if type(out['max_reference_rows']) is not int or out['max_reference_rows'] < 4:
        raise ValueError('history_context.max_reference_rows must be an integer >= 4')
    return out


def _rng(seed, request, key):
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, HISTORY_TAG,
                                  int(request) & 0xFFFFFFFF, int(key) & 0xFFFFFFFF])


def draw_history(settings, seed, request, key):
    """Deterministic choice for one (request, draw key). Always consumes 5 uniforms."""
    u = _rng(seed, request, key).random(5)
    lengths, suffix = settings['lengths_s'], settings['suffix_lengths_s']
    roles = list(settings['roles'].items())
    cumulative, role = 0., roles[-1][0]
    for name, probability in roles:
        cumulative += probability
        if u[2] < cumulative:
            role = name
            break
    return dict(applied=bool(u[0] < settings['p']),
                prefix_s=lengths[min(len(lengths) - 1, int(u[1] * len(lengths)))],
                role=role, infill=bool(u[3] < settings['p_infill']),
                suffix_s=suffix[min(len(suffix) - 1, int(u[4] * len(suffix)))])


def chart_arrays(panel, chart_index):
    """Whole native chart arrays (onset_us/end_us/kind/lane) of a training or tape panel."""
    value = panel._arrays(chart_index)
    values = value[1] if isinstance(value, tuple) else value
    return {key: np.asarray(values[key]) for key in ('onset_us', 'end_us', 'kind', 'lane')}


def _canonical(arrays, mask, mode, prefix):
    out = []
    for index in np.flatnonzero(mask):
        lane = int(arrays['lane'][index])
        out.append(Event(f'{prefix}{int(index)}', int(arrays['onset_us'][index]), int(arrays['end_us'][index]),
                         TYPES[int(arrays['kind'][index])],
                         LaneLoc(lane) if mode == 'mania_4k' else LineLoc('rhythm', 'above', 0.)))
    return tuple(out)


def region_events(arrays, mode, region, prefix=''):
    lo, hi = region
    onset = arrays['onset_us']
    return _canonical(arrays, (onset >= lo) & (onset < hi), mode, prefix)


def shift_arrays(arrays, offset_us):
    """Retime a (foreign) chart onto another clock: every time moves by offset_us."""
    return dict(arrays, onset_us=arrays['onset_us'] + int(offset_us), end_us=arrays['end_us'] + int(offset_us))


def visible_source(arrays, *, mode, audio_sha256, role, region, source_id):
    from beatfcn.train.event_reference import VisibleReference
    events = region_events(arrays, mode, region, prefix=source_id + ':')
    return VisibleReference(source_id=source_id, mode=mode, audio_sha256=audio_sha256, role=role,
                            regions=(tuple(int(t) for t in region),), events=events)


def check_no_leak(sources, request):
    """Raise unless every shown event lies outside [start, end), and every shown
    event overlapping the window is an authorized incoming carry hold."""
    start, end = request.start_us, request.end_us
    carry = {(e.onset_us, e.end_us, e.lane if request.mode == 'mania_4k' else -1)
             for e in request.known_carry}
    crossing = 0
    for source in sources:
        for lo, hi in source.regions:
            if lo < end and start < hi:
                raise ValueError('History region overlaps the target window')
        for event in source.events:
            if start <= event.onset_us < end:
                raise ValueError('History shows a target-window event')
            if event.onset_us < start < event.end_us:
                lane = event.location.lane if request.mode == 'mania_4k' else -1
                if (event.onset_us, event.end_us, lane) not in carry:
                    raise ValueError('History hold crosses into the window but is not an authorized carry')
                crossing += 1
    return crossing


def _rows(sources):
    return 1 + sum(len(s.regions) + len(s.events) for s in sources)


def history_sources(arrays, request, draw, settings, *, chart_id, audio_sha256):
    """Sources plus a diagnostics record for one drawn history choice.

    Over-capacity references are never truncated: the region falls back to the
    next shorter configured length (recorded), else history is dropped.
    """
    info = dict(format=FORMAT, drawn=dict(draw), applied=False, role=None, prefix_s=None,
                prefix_region_us=None, prefix_events=0, infill=False, suffix_s=None,
                suffix_region_us=None, suffix_events=0, crossing_holds=0, shrunk=False, reason=None)
    if not draw['applied']:
        info['reason'] = 'not_drawn'
        return (), info
    start, end, duration = request.start_us, request.end_us, request.duration_us
    lengths = [v for v in settings['lengths_s'] if v <= draw['prefix_s']][::-1]
    suffixes = [v for v in settings['suffix_lengths_s'] if v <= draw['suffix_s']][::-1]
    use_prefix, use_suffix = start > 0, bool(draw['infill'] and end < duration)
    if not use_prefix and not use_suffix:
        info['reason'] = 'song_start_no_prefix'
        return (), info
    prefix_options = lengths if use_prefix else [None]
    suffix_options = suffixes if use_suffix else [None]
    for p_index, prefix_s in enumerate(prefix_options):
        for s_index, suffix_s in enumerate(suffix_options):
            sources = []
            if prefix_s is not None:
                region = (max(0, start - int(round(prefix_s * 1e6))), start)
                sources.append(visible_source(arrays, mode=request.mode, audio_sha256=audio_sha256,
                    role=draw['role'], region=region, source_id=f'native-prefix:{chart_id}'))
            if suffix_s is not None:
                region = (end, min(duration, end + int(round(suffix_s * 1e6))))
                sources.append(visible_source(arrays, mode=request.mode, audio_sha256=audio_sha256,
                    role=settings['suffix_role'], region=region, source_id=f'native-suffix:{chart_id}'))
            if _rows(sources) > settings['max_reference_rows']:
                continue
            info.update(applied=True, crossing_holds=check_no_leak(sources, request),
                        shrunk=bool(p_index or s_index), reason=None)
            if prefix_s is not None:
                info.update(role=draw['role'], prefix_s=prefix_s, prefix_region_us=list(sources[0].regions[0]),
                            prefix_events=len(sources[0].events))
            if suffix_s is not None:
                info.update(infill=True, suffix_s=suffix_s, suffix_region_us=list(sources[-1].regions[0]),
                            suffix_events=len(sources[-1].events))
            return tuple(sources), info
    info['reason'] = 'over_reference_capacity'
    return (), info


def reference_inputs(request, audio_sha256, sources):
    """The reference keys request_inputs would produce for these sources (target never read)."""
    check_no_leak(sources, request)
    reference, provenance = encode_reference(mode=request.mode, audio_sha256=audio_sha256,
        duration_us=request.duration_us, origin_us=request.start_us,
        unknown_regions=((request.start_us, request.end_us),), sources=sources)
    for row in provenance:
        if row['event_id'] is not None and request.start_us <= row['interval_us'][0] < request.end_us:
            raise ValueError('Encoded reference contains a target-window event')
    return reference, provenance
