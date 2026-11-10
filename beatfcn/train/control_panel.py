"""Fixed user requests for the R03 short-window diagnostic.

The requested bands are chosen before selecting audio or reading model outputs.
Human charts select nonempty held-out excerpts and supply comparison targets,
never a numeric condition to be fed back to the model.
"""
from dataclasses import replace

import numpy as np

from ..events import controls as C
from ..model import features as F


REQUEST_BANDS = {
    'density_notes_per_second': [[1., 1.5], [3., 4.], [6., 8.]],
    'peak_notes_per_second': [[2., 3.], [6., 8.], [12., 16.]],
    'hold_head_ratio': [[0., .02], [.2, .3], [.5, .6]],
    'hold_occupancy_ratio': [[0., .02], [.3, .4], [.8, 1.]],
    'chord_ratio': [[0., .02], [.2, .3], [.5, .6]],
    'flick_ratio': [[0., .02], [.2, .3], [.5, .6]],
    'drag_ratio': [[0., .02], [.2, .3], [.5, .6]],
}


def requested_view(example, field=None, band=None):
    """A single explicit user condition; all other numeric fields are missing."""
    controls = np.full_like(example.controls, -1)
    if field is not None:
        mode = F.MODES[example.mode]
        if field not in C.MODE_FIELDS[mode] or band not in REQUEST_BANDS[field]:
            raise ValueError('unsupported diagnostic request')
        controls[F.CONTROL_FIELDS.index(field)] = C.BAND_EDGES[field].index(band[0])
    elif band is not None:
        raise ValueError('band without a field')
    return replace(example, controls=controls)


def response_gate(records, field):
    """Three complete, legal, strictly increasing outputs, all in their bands."""
    expected = REQUEST_BANDS[field]
    if len(records) != len(expected) or [r['requested'] for r in records] != expected:
        return {'passed': False, 'reason': 'missing or reordered requests'}
    values = [r['achieved'].get(field) for r in records]
    finite = all(v is not None and np.isfinite(v) for v in values)
    monotonic = finite and all(a < b for a, b in zip(values, values[1:]))
    in_band = finite and all(C.band_of(field, v) == band for v, band in zip(values, expected))
    valid = all(r['complete'] and r['legal'] for r in records)
    return dict(passed=bool(monotonic and in_band and valid), strictly_monotonic=bool(monotonic),
                all_in_band=bool(in_band), complete_legal=valid, values=values)


def field_decision(gates, *, total=4, required=3):
    """Early futility is mathematical: even all remaining successes cannot pass."""
    if len(gates) > total:
        raise ValueError('too many panel cases')
    successes = sum(g['passed'] for g in gates)
    remaining = total - len(gates)
    return dict(tested=len(gates), successes=successes, remaining=remaining,
                failed=successes + remaining < required,
                passed=len(gates) == total and successes >= required)


def feasible_witness(mode, field, level):
    """Construct a legal eight-second witness, solely to check request feasibility.

    These synthetic events never enter training or model quality scores. They
    establish that each individual user band is attainable on the blank canvas,
    not that this pattern would be musical or appropriate for a given song.
    """
    from ..events.canonical import Event, LaneLoc, LineLoc
    if field not in C.MODE_FIELDS[mode] or level not in range(3):
        raise ValueError('unsupported witness')
    notes = []
    def add(t, kind='tap', end=None, lane=0):
        loc = LaneLoc(lane) if mode == 'mania_4k' else LineLoc('0', 'above', float(lane))
        notes.append(Event(f'witness:{len(notes)}', t, t if end is None else end, kind, loc))
    if field == 'density_notes_per_second':
        count = (10, 28, 56)[level]
        for i in range(count):
            add(100000 + i * (7600000 // count), lane=i % 4)
    elif field == 'peak_notes_per_second':
        for i in range((2, 6, 12)[level]):
            add(100000 + i * 40000, lane=i % 4)
    elif field == 'hold_occupancy_ratio':
        duration = (0, 2800000, 7200000)[level]
        add(250000, 'hold' if duration else 'tap', 250000 + duration)
        add(7700000, lane=1)
    elif False:
        lane = 0
        for i in range(21):
            if i > (0, 7, 18)[level]:
                lane = (lane + 1) % 4
            add(100000 + i * 300000, lane=lane)
    else:
        count = (0, 5, 11)[level]
        for i in range(20):
            t = 100000 + i * 350000
            kind = ('hold' if field == 'hold_head_ratio' else 'flick' if field == 'flick_ratio' else
                    'drag' if field == 'drag_ratio' else 'tap') if i < count else 'tap'
            add(t, kind, t + 100000 if kind == 'hold' else t, lane=i % 4)
            if field == 'chord_ratio' and i < count:
                add(t, lane=(i + 1) % 4)
    return notes
