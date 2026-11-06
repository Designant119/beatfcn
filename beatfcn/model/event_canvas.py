"""Fixed-capacity rhythm targets for parallel, editable event generation.

Slots are allocated from the request interval, never its hidden event count.
EMPTY is a completed decision; MASK is unknown. This module is a representation
contract, not a trained generator or a repair policy. Phira spatial coordinates
are not model targets. Protected native objects remain in the request separately.
"""
from dataclasses import dataclass

import numpy as np

from beatfcn.events.canonical import TYPES


EMPTY = len(TYPES)
MASK = EMPTY + 1
KIND, HEAD_MS, COPY_INDEX, TAIL_MS, LANE = range(5)


@dataclass(frozen=True)
class ClockLayout:
    start_us: int
    end_us: int
    block_us: int = 400_000
    phase_us: int = 0

    def __post_init__(self):
        if any(type(x) is not int for x in (self.start_us, self.end_us, self.block_us, self.phase_us)):
            raise TypeError('clock coordinates must be integer microseconds')
        if self.start_us >= self.end_us or self.block_us <= 0:
            raise ValueError('empty interval or invalid block width')
        if self.block_us % 1000 or self.phase_us % 1000:
            raise ValueError('clock origin and width must be millisecond aligned')
        if not 0 <= self.phase_us < self.block_us:
            raise ValueError('phase must lie within one block')

    @property
    def origin_us(self):
        return ((self.start_us - self.phase_us) // self.block_us) * self.block_us + self.phase_us

    @property
    def blocks(self):
        return tuple((base, max(self.start_us, base), min(self.end_us, base + self.block_us))
                     for base in range(self.origin_us, self.end_us, self.block_us))

    def block_for(self, onset_us):
        if not self.start_us <= onset_us < self.end_us:
            raise ValueError('head outside the requested interval')
        return (onset_us - self.origin_us) // self.block_us


@dataclass(frozen=True, order=True)
class RhythmEvent:
    onset_us: int
    end_us: int
    type: str
    lane: int = -1

    @classmethod
    def from_event(cls, event, mode):
        if mode not in ('mania_4k', 'phira'):
            raise ValueError('unsupported mode')
        lane = event.location.lane if mode == 'mania_4k' else -1
        return cls(event.onset_us, event.end_us, event.type, lane)


class CanvasOverflow(ValueError):
    """A target cannot fit; callers must report it, never truncate it."""


@dataclass
class EventCanvas:
    layout: ClockLayout
    mode: str
    copy_times: tuple[int, ...]
    values: np.ndarray  # [blocks, configured slots, fields]; no target-length mask

    @property
    def slots(self):
        return self.values.shape[1]


def new_canvas(layout, mode, *, slots=32, copy_times=(), fill=MASK):
    """Allocate an input without inspecting any hidden events."""
    if mode not in ('mania_4k', 'phira') or type(slots) is not int or slots <= 0:
        raise ValueError('invalid mode or slot capacity')
    if fill not in (MASK, EMPTY):
        raise ValueError('blank canvas must be MASK or EMPTY')
    times = tuple(copy_times)
    if len(set(times)) != len(times) or any(type(t) is not int for t in times):
        raise ValueError('copy times must be unique integer microseconds')
    values = np.full((len(layout.blocks), slots, 5), -1, dtype=np.int64)
    values[..., KIND] = fill
    return EventCanvas(layout, mode, times, values)


def encode_targets(events, layout, mode, *, slots=32, copy_times=()):
    """Encode rhythm labels; never pass this completed canvas as an empty request."""
    result = new_canvas(layout, mode, slots=slots, copy_times=copy_times, fill=EMPTY)
    buckets = [[] for _ in layout.blocks]
    for event in events:
        buckets[layout.block_for(event.onset_us)].append(event)
    copy_lookup = {t: i for i, t in enumerate(result.copy_times)}
    for block, events_in_block in enumerate(buckets):
        if len(events_in_block) > slots:
            raise CanvasOverflow(f'block {block} needs {len(events_in_block)} slots, capacity {slots}')
        base = layout.blocks[block][0]
        # Phira targets are canonically ordered without hidden placement fields.
        events_in_block.sort(key=lambda e: (e.onset_us, e.lane if mode == 'mania_4k' else TYPES.index(e.type),
                                           e.end_us, TYPES.index(e.type)))
        for slot, event in enumerate(events_in_block):
            if event.type not in TYPES or (mode == 'mania_4k' and event.type not in ('tap', 'hold')):
                raise ValueError('unsupported event type')
            if (mode == 'mania_4k' and event.lane not in range(4)) or (mode == 'phira' and event.lane != -1):
                raise ValueError('invalid rhythm location')
            row = result.values[block, slot]
            row[KIND] = TYPES.index(event.type)
            if event.onset_us in copy_lookup:
                row[COPY_INDEX] = copy_lookup[event.onset_us]
            elif event.onset_us % 1000:
                raise ValueError('non-millisecond head requires an authorized copy time')
            else:
                row[HEAD_MS] = (event.onset_us - base) // 1000
            if event.type == 'hold':
                if event.end_us <= event.onset_us or event.end_us % 1000:
                    raise ValueError('target hold needs a positive millisecond-aligned tail')
                row[TAIL_MS] = event.end_us // 1000
            elif event.end_us != event.onset_us:
                raise ValueError('non-hold must end at its head')
            row[LANE] = event.lane
    return result


def decode_targets(canvas):
    """Decode a complete canvas; malformed predictions require explicit handling."""
    if canvas.mode not in ('mania_4k', 'phira'):
        raise ValueError('unsupported canvas mode')
    if canvas.values.ndim != 3 or canvas.values.shape != (len(canvas.layout.blocks), canvas.slots, 5):
        raise ValueError('invalid canvas shape')
    if canvas.values.dtype != np.int64:
        raise ValueError('canvas fields must be int64')
    events = []
    for block, (base, start, end) in enumerate(canvas.layout.blocks):
        for row in canvas.values[block]:
            kind, offset, copy_index, tail, lane = map(int, row)
            if min(offset, copy_index, tail, lane) < -1:
                raise ValueError('invalid field sentinel')
            if kind == MASK:
                raise ValueError('cannot finalize an unknown slot')
            if kind == EMPTY:
                if np.any(row[1:] != -1):
                    raise ValueError('empty slot has event payload')
                continue
            if not 0 <= kind < len(TYPES):
                raise ValueError('invalid event type')
            if (offset >= 0) == (copy_index >= 0):
                raise ValueError('head must use exactly one time representation')
            if copy_index >= 0:
                if copy_index >= len(canvas.copy_times):
                    raise ValueError('unknown copy time')
                onset = canvas.copy_times[copy_index]
            else:
                onset = base + offset * 1000
            if not start <= onset < end:
                raise ValueError('decoded head crosses its owning block')
            if canvas.mode == 'mania_4k' and (lane not in range(4) or kind > 1):
                raise ValueError('invalid mania event')
            if canvas.mode == 'phira' and lane != -1:
                raise ValueError('Phira rhythm output contains a lane')
            if kind == TYPES.index('hold'):
                finish = tail * 1000
                if tail < 0 or finish <= onset:
                    raise ValueError('hold tail must follow its head')
            else:
                if tail != -1:
                    raise ValueError('non-hold has a tail prediction')
                finish = onset
            events.append(RhythmEvent(onset, finish, TYPES[kind], lane))
    # Sorting does not remove multiplicity; legal Phira events may share fields.
    return tuple(sorted(events))


def masked_view(labels, visible):
    """Reveal explicitly selected slots; no target-derived padding/count mask."""
    visible = np.asarray(visible)
    if visible.dtype != np.bool_ or visible.shape != labels.values.shape[:2]:
        raise ValueError('visible mask must be bool with the fixed slot shape')
    view = new_canvas(labels.layout, labels.mode, slots=labels.slots, copy_times=labels.copy_times)
    view.values[visible] = labels.values[visible]
    return view


def active_holds(events, at_us):
    """Use known or already generated events only; never hidden future labels."""
    return tuple(sorted(e for e in events if e.type == 'hold' and e.onset_us < at_us < e.end_us))
