"""Permission projection for context-only Stage A. No hidden targets are accepted.

The request-local memory is built once from authorized observations/templates;
it survives core boundaries. Generated rhythm-only prefixes always stay unplaced.
"""
from dataclasses import dataclass, field
from collections import Counter
import numpy as np
from ..events.canonical import Event
from .rpe import digest

POINTS = 8
MAX_MEMORY = 256
STATUS = {'unplaced': 0, 'sampled': 1, 'unsupported': 2, 'masked': 3, 'truncated': 4}


@dataclass
class Packet:
    points: np.ndarray = field(default_factory=lambda: np.zeros((POINTS, 7), np.float64))
    valid: np.ndarray = field(default_factory=lambda: np.zeros(POINTS, bool))
    status: int = STATUS['unplaced']
    reason: str = 'rhythm_only_generated_prefix'
    source_kind: int = 0  # 0 unknown, 1 event, 2 reference, 3 retained scene, 4 template, 5 observed line fragments

    def relative(self, c0_us):
        points = self.points.copy()
        points[:, 0] = (points[:, 0] - c0_us) / 1e6
        points[~self.valid] = 0
        return points.astype(np.float32)

    def key(self):
        # No native line IDs, event IDs, or hidden inventory in memory identity/order.
        return digest({'points': self.points.tolist(), 'valid': self.valid.tolist(),
                       'status': self.status, 'source_kind': self.source_kind})


def _sample(scene, line, x, interval, shift_us, width=1.0, source_kind=1):
    a, b = map(int, interval)
    if b < a:
        raise ValueError('inverted authorized interval')
    try:
        cuts = scene.breakpoints(line, a - shift_us, b - shift_us)
        mandatory = {a, b}
        for native in cuts:
            t = native + shift_us
            mandatory.update(q for q in (t - 1, t, t + 1) if a <= q <= b)
        candidates = sorted(mandatory)
        truncated = len(candidates) > POINTS
        if truncated:
            candidates = [candidates[i] for i in np.linspace(0, len(candidates) - 1, POINTS).round().astype(int)]
        else:
            for t in np.linspace(a, b, POINTS).round().astype(int):
                if len(candidates) == POINTS:
                    break
                if t not in candidates:
                    candidates.append(int(t))
            candidates.sort()
        packet = Packet(status=STATUS['truncated' if truncated else 'sampled'],
                        reason='sampled_native_curve_no_continuous_error_certificate', source_kind=source_kind)
        for i, t in enumerate(candidates):
            geom = scene.region(line, x, t - shift_us, judge_area=width)
            # Stage A deliberately excludes visual anchors. Their zeros are never advertised as valid.
            packet.points[i] = [t, *geom, 0, 0]
            packet.valid[i] = True
        return packet
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        return Packet(status=STATUS['unsupported'], reason=str(exc), source_kind=source_kind)


@dataclass
class SpatialContext:
    known: dict
    references: dict
    memory: tuple
    cache_key: str
    audit: dict
    request_fingerprint: str

    def validate(self, request, evidence):
        if self.request_fingerprint != digest({'request': request, 'evidence': list(evidence)}):
            raise ValueError('stale spatial projection: request/permissions/reference changed')

    def event_packet(self, event, *, reference_id=None, generated=False):
        if generated:
            return Packet()
        key = (reference_id, event.event_id) if reference_id is not None else event.event_id
        source = self.references if reference_id is not None else self.known
        record = source.get(key)
        # Event IDs alone never authorize recovering hidden or changed geometry.
        if record is None or record[0] != event.signature():
            return Packet(status=STATUS['masked'], reason='event_not_in_authorized_projection')
        return record[1]


def project(scene, request, evidence=(), *, shift_us=0, reference_scenes=None, max_memory=MAX_MEMORY,
            memory_overflow='error'):
    """Project only explicit O/R/grants. Opaque legacy scene_ref grants no permission.

    known_event_spans use a conservative judgement query envelope; reference
    geometry is clipped to its explicitly selected span and never enters the
    target binding memory. Retained-line grants must name source, lines and span.
    """
    if request['mode'] != 'phira':
        raise ValueError('operation projection is Phira-only')
    if not 0 <= max_memory <= MAX_MEMORY:
        raise ValueError('invalid request memory capacity')
    if memory_overflow not in ('error', 'truncate_reported'):
        raise ValueError('unknown memory overflow policy')
    observed = [Event.from_json(e) for e in request['observed_events']]
    known, refs, pool, observed_lines = {}, {}, [], {}
    audits = []

    def event_packet(source, event, kind, clip=None):
        if source is None:
            return Packet(status=STATUS['unsupported'], reason='source_geometry_unavailable', source_kind=kind)
        record = source.notes.get(event.event_id)
        if record is None:
            return Packet(status=STATUS['masked'], reason='event_missing_from_source', source_kind=kind)
        li, note = record
        from .rpe import beat
        native_head = round((source.seconds(beat(note['startTime'])) + source.offset) * 1e6) + shift_us
        native_end = round((source.seconds(beat(note['endTime'])) + source.offset) * 1e6) + shift_us if event.is_hold else native_head
        native_type = {1: 'tap', 2: 'hold', 3: 'flick', 4: 'drag'}.get(note['type'])
        if (abs(native_head - event.onset_us) > 1 or abs(native_end - event.end_us) > 1
                or native_type != event.type or str(li) != event.location.line_id
                or float(note['positionX']) != event.location.native_x):
            return Packet(status=STATUS['masked'], reason='source_event_or_clock_mismatch', source_kind=kind)
        if note.get('yOffset', 0) != 0:
            return Packet(status=STATUS['unsupported'], reason='nonzero_yOffset_unvalidated', source_kind=kind)
        a = max(0, event.onset_us - 220_000)
        b = min(request['audio_duration_us'], max(event.onset_us + 290_000, event.end_us - 220_000))
        if clip is not None:
            a, b = max(a, clip[0]), min(b, clip[1] - 1)
        return _sample(source, li, note['positionX'], (a, b), shift_us,
                       note.get('judgeArea', 1), kind)

    for event in observed:
        packet = event_packet(scene, event, 1)
        known[event.event_id] = (event.signature(), packet)
        if packet.valid.any():
            # One memory summary per authorized line/width, not one slot per note.
            # Only sampled points inside observed-event envelopes enter it. There
            # is no interpolation through the gaps and no future line grant.
            points = packet.points[packet.valid].copy()
            points[:, 3] -= event.location.native_x / 675
            key = (event.location.line_id, float(points[0, 4]))
            observed_lines.setdefault(key, []).extend(points.tolist())
        audits.append(packet.reason)
    for (line_id, width), samples in observed_lines.items():
        samples = np.unique(np.asarray(samples, np.float64), axis=0)
        samples = samples[np.argsort(samples[:, 0], kind='stable')]
        redundant = any(g['kind'] == 'retained_line_scene' and scene is not None and
                        g.get('source_sha256') == scene.source_sha256 and line_id in map(str, g['line_ids']) and
                        g['span_us'][0] <= samples[0, 0] and g['span_us'][1] >= samples[-1, 0] and
                        abs(width - .21 / scene.profile.aspect * 2) < 1e-10
                        for g in request.get('operation_space', {}).get('grants', []))
        if redundant:
            continue
        compressed = len(samples) > POINTS
        if compressed:
            samples = samples[np.linspace(0, len(samples) - 1, POINTS).round().astype(int)]
        packet = Packet(status=STATUS['truncated' if compressed else 'sampled'], source_kind=5,
                        reason='authorized_line_fragment_summary_not_an_executable_bind')
        packet.points[:len(samples)], packet.valid[:len(samples)] = samples, True
        pool.append(packet)
    sources = dict(reference_scenes or {})
    if scene is not None:
        sources[scene.source_sha256] = scene
    public = {ref['reference_id']: ref for ref in request.get('references', [])}
    for ref in evidence:
        grant = public.get(ref['reference_id'], {})
        source = sources.get(grant.get('spatial_source_sha256'))
        allowed = grant.get('space_attributes', False)
        if allowed and (grant.get('span') != {'start_us': ref['span'][0], 'end_us': ref['span'][1]}
                        or grant.get('coverage') != ref['coverage']):
            raise ValueError('reference geometry span or coverage differs from its permission')
        granted = {e['event_id']: Event.from_json(e).signature() for e in grant.get('events', [])}
        for raw in ref['selected']:
            event = Event.from_json(raw)
            packet = (event_packet(source, event, 2, ref['span']) if allowed and granted.get(event.event_id) == event.signature()
                      else Packet(status=STATUS['masked'], reason='reference_attributes_not_authorized', source_kind=2))
            refs[ref['reference_id'], event.event_id] = (event.signature(), packet)
    for grant in request.get('operation_space', {}).get('grants', []):
        if grant['kind'] == 'retained_line_scene':
            if scene is None or grant['source_sha256'] != scene.source_sha256:
                raise ValueError('retained scene source mismatch')
            a, b = grant['span_us']
            if not 0 <= a < b <= request['audio_duration_us']:
                raise ValueError('retained scene grant outside audio')
            for line in grant['line_ids']:
                pool.append(_sample(scene, line, 0, (a, b), shift_us, source_kind=3))
        elif grant['kind'] == 'explicit_template':
            a, b = grant['span_us']
            normal = np.asarray(grant['normal'], float)
            if (not 0 <= a < b <= request['audio_duration_us'] or normal.shape != (2,)
                    or not np.isclose(np.linalg.norm(normal), 1) or grant['half_width'] <= 0):
                raise ValueError('invalid explicit static template')
            packet = Packet(status=STATUS['sampled'], reason='explicit_static_template', source_kind=4)
            for i, t in enumerate(np.linspace(a, b, POINTS)):
                packet.points[i] = [t, *normal, grant['offset'], grant['half_width'], 0, 0]
            if not np.isfinite(packet.points).all():
                raise ValueError('nonfinite template')
            packet.valid[:] = True
            pool.append(packet)
        else:
            raise ValueError('unknown operation-space grant')
    # Deduplicate only memory definitions, never event rows or event multiplicity.
    unique = {p.key(): p for p in pool if p.valid.any()}
    ordered = [unique[k] for k in sorted(unique)]
    total_memory = len(ordered)
    if total_memory > max_memory and memory_overflow == 'error':
        raise ValueError(f'operation memory {len(ordered)} exceeds {max_memory}; no truncation applied')
    # Optional, explicit representation budget only: event rows and labels remain intact.
    ordered = ordered[:max_memory]
    identity = {'compiler_cache': None if scene is None else scene.cache_key, 'request': request,
                'evidence': list(evidence), 'shift_us': shift_us, 'max_memory': max_memory,
                'memory_overflow': memory_overflow,
                'reference_sources': {k: v.cache_key for k, v in sorted(sources.items())}}
    def coverage(packets):
        packets = list(packets)
        resolved = sum(bool(p.valid.any()) for p in packets)
        return {'rows': len(packets), 'resolved_rows': resolved,
                'resolved_fraction': resolved / len(packets) if packets else None,
                'status_counts': dict(Counter(next(k for k, v in STATUS.items() if v == p.status) for p in packets)),
                'reason_counts': dict(Counter(p.reason for p in packets))}
    coverage_audit = {'known': coverage(p for _, p in known.values()),
                      'references': coverage(p for _, p in refs.values()),
                      'memory_before_deduplication': coverage(pool)}
    all_rows = sum(v['rows'] for v in coverage_audit.values())
    resolved_rows = sum(v['resolved_rows'] for v in coverage_audit.values())
    stratum = ('no_spatial_input' if not all_rows else 'fully_supported' if resolved_rows == all_rows
               else 'partially_supported' if resolved_rows else 'unsupported')
    return SpatialContext(known, refs, tuple(ordered), digest(identity),
                          {'schema': 'operation_context_v1', 'observed_events': len(observed),
                           'known_resolved_rows': sum(p.valid.any().item() for _, p in known.values()),
                           'reference_rows': len(refs), 'memory_rows': len(ordered),
                           'memory_requested': total_memory, 'memory_truncated': total_memory - len(ordered),
                           'memory_point_budget_limited': sum(p.status == STATUS['truncated'] for p in ordered),
                           'memory_semantics': 'sampled authorized fragments; not an executable BIND pool',
                           'coverage': coverage_audit, 'geometry_stratum': stratum,
                           'reasons': {r: audits.count(r) for r in sorted(set(audits))},
                           'geometry_samples_are_continuous_certificates': False},
                          digest({'request': request, 'evidence': list(evidence)}))
