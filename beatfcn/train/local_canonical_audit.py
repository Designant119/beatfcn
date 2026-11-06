"""Canonical integrity checks for local candidates, not native admission.

The cached document can prove its own consistency, but cannot establish that the
native package or audio decoder has the same revision and clock. No notes are
deleted or corrected by this audit.
"""
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import re

from beatfcn.events.canonical import (
    BpmSegment, Event, LaneLoc, LineLoc, MODE_TYPES, TimingMap,
    build_model_view, sec_to_us,
)
from beatfcn.train.data import chart_from_canon, model_shift_us


def rhythm_fingerprint(mode, events):
    """Ignore Phira geometry, keep mania keys and every repeated event."""
    values = sorted((e.onset_us, e.end_us, e.type,
                     e.location.lane if mode == 'mania_4k' else -1) for e in events)
    payload = json.dumps([mode, values], separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def internal_clock(doc, events):
    raw = doc.get('timing')
    if not raw:
        return dict(status='pending_no_common_timing_map', checked_events=0)
    segments = tuple(BpmSegment(Fraction(s['start_beat']), Fraction(s['bpm']),
                                Fraction(s['start_sec'])) for s in raw['segments'])
    if (not segments or any(s.bpm <= 0 for s in segments)
            or any(b.start_beat <= a.start_beat for a, b in zip(segments, segments[1:]))):
        raise ValueError('Invalid canonical BPM segments')
    timing = TimingMap(segments, Fraction(raw['offset_sec']))
    if timing.offset_sec != Fraction(doc['offset_sec']):
        raise ValueError('Canonical offset disagrees with timing map')
    # Serialized rational coordinates can contain insignificant decimal roundoff
    # from the legacy ingest path. Use the same one-microsecond storage precision
    # as the event check, and expose the residual instead of demanding exact
    # equality of potentially enormous rational denominators.
    segment_errors = [abs(a.start_sec + (b.start_beat-a.start_beat)*60/a.bpm - b.start_sec)*1_000_000
                      for a, b in zip(segments, segments[1:])]
    discontinuities = [i for i, error in enumerate(segment_errors, 1) if error > 1]
    missing, mismatches, maximum = [], [], 0
    for event in events:
        beats = doc.get('native_beats', {}).get(event.event_id)
        if beats is None:
            missing.append(event.event_id)
            continue
        if len(beats) != 2:
            raise ValueError('Malformed native beat pair')
        actual = (event.onset_us, event.end_us)
        expected = tuple(sec_to_us(timing.audio_sec(Fraction(b))) for b in beats)
        error = max(abs(a-b) for a, b in zip(actual, expected))
        maximum = max(maximum, error)
        if error > 1:
            mismatches.append(dict(event_id=event.event_id, error_us=error))
    return dict(status='consistent' if not (missing or mismatches or discontinuities) else 'inconsistent',
                checked_events=len(events)-len(missing), missing_count=len(missing),
                missing_examples=missing[:10], mismatch_count=len(mismatches),
                mismatch_examples=mismatches[:10], max_error_us=maximum,
                discontinuities=discontinuities,
                max_segment_residual_us=float(max(segment_errors, default=0)))


def audit_document(doc, row):
    """Return diagnostics and the complete, quantized target only if checks pass."""
    for key in ('chart_id', 'source', 'native_id', 'mode', 'audio_sha256'):
        if doc.get(key) != row.get(key):
            raise ValueError(f'Canonical/manifest identity mismatch: {key}')
    if doc.get('schema') != 'beatfcn.canon.v1':
        raise ValueError('Unexpected canonical schema')
    if not re.fullmatch('[0-9a-f]{64}', doc.get('source_sha256', '')):
        raise ValueError('Missing source content identity')
    duration = row['audio_sec']
    if (not math.isfinite(duration) or duration <= 0
            or abs(doc['audio_sec']-duration) > 1e-6):
        raise ValueError('Canonical/manifest audio duration mismatch')
    if row.get('offset_decision', {}).get('decision_version') != 2:
        raise ValueError('Frozen offset-v2 decision required')
    chart = chart_from_canon(doc)
    fakes = []
    for raw in doc.get('fakes', []):
        if type(raw.get('fake')) not in (int, bool) or raw['fake'] != 1:
            raise ValueError('Fake sidecar must be explicitly non-playable')
        fakes.append(Event.from_json(raw))
    ids = [e.event_id for e in (*chart.events, *fakes)]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate event identity across playable/fake sets')
    issues = []
    mode = row['mode']
    for event in chart.events:
        if (event.type not in MODE_TYPES[mode]
                or not isinstance(event.location, LaneLoc if mode == 'mania_4k' else LineLoc)):
            issues.append('mode_type_or_location')
        if (event.is_hold and event.end_us <= event.onset_us
                or not event.is_hold and event.end_us != event.onset_us):
            issues.append('invalid_duration')
    # Legacy native-duplicate quarantines are not playability evidence. Preserve
    # multiplicity; retain actual representation/timing failures below.
    issues.extend('source_issue:'+i['code'] for i in doc.get('issues', [])
                  if i['severity'] == 'quarantine' and not (mode == 'phira' and
                     i['code'] in {'duplicate_object', 'duplicate_object_in_source'}))
    clock = internal_clock(doc, (*chart.events, *fakes))
    if clock['status'] == 'inconsistent':
        issues.append('internal_clock_inconsistent')
    shift = model_shift_us(row, True)
    shifted = chart_from_canon(doc, shift).events
    duration_us = math.floor(duration*1_000_000)
    out_of_bounds = [e.event_id for e in shifted if e.onset_us < 0
                     or e.onset_us >= duration_us or e.end_us > duration_us]
    if out_of_bounds:
        issues.append('model_clock_outside_audio')
    view = build_model_view(mode, (), shifted)
    issues.extend('model_view:'+i.code for i in view.quarantine)
    targets = tuple(e for group in view.target_groups for e in group.events)
    if len(targets) != len(chart.events):
        issues.append('model_view_changed_event_count')
    rounded_outside = [e.event_id for e in targets if e.onset_us < 0
                       or e.onset_us >= duration_us or e.end_us > duration_us]
    if rounded_outside:
        issues.append('quantized_clock_outside_audio')
    report = dict(chart_id=row['chart_id'], mode=mode, split=row['split'],
                  audio_sha256=row['audio_sha256'], source_sha256=doc['source_sha256'],
                  playable_notes=len(chart.events), fake_notes=len(fakes),
                  playable_types=dict(Counter(e.type for e in chart.events)),
                  fake_types=dict(Counter(e.type for e in fakes)),
                  model_shift_us=shift, internal_clock=clock,
                  out_of_bounds_count=len(out_of_bounds), out_of_bounds_examples=out_of_bounds[:10],
                  quantized_out_of_bounds_count=len(rounded_outside),
                  model_view_notes=len(targets), issues=dict(Counter(issues)),
                  canonical_integrity_passed=not issues,
                  source_rhythm_sha256=rhythm_fingerprint(mode, chart.events),
                  model_rhythm_sha256=rhythm_fingerprint(mode, targets),
                  native_source_verified=False, audio_clock_verified=False, training_ready=False)
    return report, targets if not issues else ()
