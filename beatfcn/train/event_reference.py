"""CPU encoding of explicitly authorized event references.

Version 1 covers same-audio prefixes and visible regions. It does not align
different songs or infer permission from a target chart. This encoder is opt-in;
the legacy local training loader still uses its frozen blank-reference contract.
"""
from dataclasses import dataclass
import re

import numpy as np

from beatfcn.events.canonical import Event, LaneLoc, LineLoc, MODE_TYPES, TYPES

TIME_SCALE_US = 1_200_000_000
REFERENCE_COLUMNS = (
    'start_relative', 'end_relative', 'tap', 'hold', 'flick', 'drag', 'event',
    'human_input', 'model_history', 'human_alternative', 'known_region', 'unknown_region',
)
SOURCE_COLUMN = {'human_input': 7, 'model_history': 8, 'human_alternative': 9}


@dataclass(frozen=True)
class VisibleReference:
    """A caller-authorized complete view of the given regions, not hidden truth.

    Known holds expose their full tails, including tails outside the region.
    Empty regions still convey known silence. Repeated notes remain repeated.
    ``source_id`` names the human chart or prior generation artifact for auditing.
    """
    source_id: str
    mode: str
    audio_sha256: str
    role: str
    regions: tuple[tuple[int, int], ...]
    events: tuple[Event, ...]


def _regions(regions, duration_us):
    result = tuple(regions)
    for interval in result:
        if (len(interval) != 2 or any(type(t) is not int for t in interval)
                or not 0 <= interval[0] < interval[1] <= duration_us):
            raise ValueError('Reference regions require valid integer audio-clock bounds')
    ordered = sorted(result)
    if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
        raise ValueError('Reference regions overlap')
    return tuple(ordered)


def encode_reference(*, mode, audio_sha256, duration_us, origin_us,
                     unknown_regions, sources=(), max_rows=4096):
    """Build the existing [R,16] input without consulting any hidden target.

    Unknown and visible spans must be disjoint for same-chart roles; only a
    ``human_alternative`` source (another human chart of the same audio) may
    overlap the unknown window. A cross-song exemplar needs a separate explicit
    clock contract and is rejected here, rather than being assigned a false
    shared audio clock.
    """
    if mode not in MODE_TYPES or not re.fullmatch('[0-9a-f]{64}', audio_sha256):
        raise ValueError('A supported mode and exact audio identity are required')
    if (type(duration_us) is not int or type(origin_us) is not int
            or not 0 <= origin_us <= duration_us <= TIME_SCALE_US
            or type(max_rows) is not int or max_rows < 1):
        raise ValueError('Invalid reference time or capacity contract')
    unknown = _regions(unknown_regions, duration_us)
    rows, times, provenance = [], [], []

    def append(start, end, *, column, source=None, event=None):
        if len(rows) >= max_rows:
            raise ValueError('Reference capacity exceeded; never truncate authorized inputs')
        row = np.zeros(len(REFERENCE_COLUMNS), np.float32)
        row[:2] = [(start-origin_us)/TIME_SCALE_US, (end-origin_us)/TIME_SCALE_US]
        row[column] = 1
        if source is not None:
            row[SOURCE_COLUMN[source.role]] = 1
        if event is not None:
            row[2+TYPES.index(event.type)] = 1
        rows.append(row)
        times.append((start-origin_us)/1_000_000)
        provenance.append(dict(source_id=source.source_id if source else None,
                               role=source.role if source else 'unknown',
                               event_id=event.event_id if event else None,
                               interval_us=[start, end]))

    for start, end in unknown:
        append(start, end, column=15)
    seen_sources = set()
    for source in sources:
        if (source.mode != mode or source.audio_sha256 != audio_sha256
                or source.role not in SOURCE_COLUMN or not source.source_id):
            raise ValueError('Reference mode, audio identity or provenance differs')
        if source.source_id in seen_sources:
            raise ValueError('Repeated reference source identity')
        seen_sources.add(source.source_id)
        visible = _regions(source.regions, duration_us)
        # Same-chart views (human_input / model_history) must not show the unknown target. A
        # human_alternative is ANOTHER human chart of the same audio (difficulty spread), not hidden
        # truth of this target, so it is the one role allowed to overlap the unknown window.
        if not visible or (source.role != 'human_alternative'
                           and any(a < d and c < b for a, b in visible for c, d in unknown)):
            raise ValueError('Visible reference overlaps unknown target or has no authorized region')
        for start, end in visible:
            append(start, end, column=14, source=source)
        seen_events = set()
        for event in sorted(source.events, key=lambda e: (e.onset_us, e.event_id)):
            if event.event_id in seen_events:
                raise ValueError('Repeated event identity within a reference source')
            seen_events.add(event.event_id)
            if (event.type not in MODE_TYPES[mode]
                    or not isinstance(event.location, LaneLoc if mode == 'mania_4k' else LineLoc)
                    or not any(start <= event.onset_us < end for start, end in visible)):
                raise ValueError('Reference event outside its authorized mode/region')
            if (not 0 <= event.onset_us <= event.end_us <= duration_us
                    or event.is_hold and event.end_us <= event.onset_us
                    or not event.is_hold and event.end_us != event.onset_us):
                raise ValueError('Reference has an invalid event duration')
            append(event.onset_us, event.end_us, column=10, source=source, event=event)
    if not rows:
        raise ValueError('Empty reference must state an explicit unknown or known region')
    inputs = dict(reference=np.stack(rows), reference_times=np.asarray(times, np.float32),
                  reference_valid=np.ones(len(rows), bool))
    if not all(np.isfinite(value).all() for value in inputs.values()):
        raise ValueError('Nonfinite reference encoding')
    return inputs, provenance
