"""Exact complete-event sequence and causal state for the isolated R17 probe.

This is a serialization/input contract, not a neural model or a playability
repair. Grammar masks never prohibit occupied lanes or remove duplicate notes.
"""
from dataclasses import dataclass
from itertools import groupby

import numpy as np

from beatfcn.events.canonical import TYPES
from .event_canvas import ClockLayout, RhythmEvent, CanvasOverflow


FIELDS = ('group', 'head_block', 'head_cell', 'head_ms', 'member', 'kind',
          'tail_block', 'tail_cell', 'tail_ms', 'lane')
END, NEW = 0, 1


@dataclass(frozen=True)
class SequenceSpec:
    layout: ClockLayout
    mode: str
    duration_us: int
    copy_times: tuple[int, ...] = ()
    permitted_heads_us: tuple[int, ...] | None = None
    allowed_types: tuple[str, ...] | None = None
    max_decisions: int = 8192
    max_events: int = 2048

    def __post_init__(self):
        if self.mode not in ('mania_4k', 'phira'):
            raise ValueError('unsupported mode')
        if type(self.duration_us) is not int or self.duration_us < self.layout.end_us or self.layout.start_us < 0:
            raise ValueError('invalid declared time domain')
        if any(type(n) is not int or n <= 0 for n in (self.max_decisions, self.max_events)):
            raise ValueError('resource limits must be positive integers')
        if len(set(self.copy_times)) != len(self.copy_times) or any(type(t) is not int for t in self.copy_times):
            raise ValueError('COPY_TIME must contain distinct integer microseconds')
        if self.permitted_heads_us is not None:
            if any(type(t) is not int or t % 1000 for t in self.permitted_heads_us):
                raise ValueError('grid permissions must be integer milliseconds; use COPY_TIME otherwise')
            if len(set(self.permitted_heads_us)) != len(self.permitted_heads_us):
                raise ValueError('duplicate head permission')
        kinds = self.allowed_types if self.allowed_types is not None else self.mode_types
        if not kinds or len(set(kinds)) != len(kinds) or any(t not in self.mode_types for t in kinds):
            raise ValueError('unsupported type permission')

    @property
    def mode_types(self):
        return ('tap', 'hold') if self.mode == 'mania_4k' else TYPES

    @property
    def time_blocks(self):
        # Inclusive full tails at duration_us require a final bucket at an exact boundary.
        return self.duration_us // 2_000_000 + 1


@dataclass(frozen=True)
class Decision:
    field: str
    value: int


def validate_event(event, spec, *, carry=False):
    if any(type(t) is not int for t in (event.onset_us, event.end_us, event.lane)):
        raise ValueError('event coordinates must be integers')
    kinds = spec.mode_types if carry or spec.allowed_types is None else spec.allowed_types
    if event.type not in kinds:
        raise ValueError('event violates type permission')
    if spec.mode == 'mania_4k' and event.lane not in range(4) or spec.mode == 'phira' and event.lane != -1:
        raise ValueError('event violates mode location')
    if event.type == 'hold':
        if event.end_us <= event.onset_us or event.end_us % 1000 or event.end_us > spec.duration_us:
            raise ValueError('hold tail must be positive, exact and inside the full time domain')
    elif event.end_us != event.onset_us:
        raise ValueError('non-hold has duration')
    if carry and not (event.type == 'hold' and 0 <= event.onset_us < spec.layout.start_us < event.end_us):
        raise ValueError('carry requires an explicitly known incoming active hold')


class EventSequenceState:
    def __init__(self, spec, *, known_carry=()):
        self.spec = spec
        self.known_carry = tuple(known_carry)
        for event in self.known_carry:
            validate_event(event, spec, carry=True)
        low, high = spec.layout.start_us, spec.layout.end_us
        if spec.permitted_heads_us is None:
            self.grid_us = np.arange((low + 999) // 1000 * 1000, high, 1000, dtype=np.int64)
        else:
            self.grid_us = np.asarray(sorted(t for t in spec.permitted_heads_us if low <= t < high), dtype=np.int64)
        self.copy_ids = tuple(i for i, t in enumerate(spec.copy_times) if low <= t < high)
        kinds = spec.allowed_types if spec.allowed_types is not None else spec.mode_types
        if kinds == ('hold',):
            # A selected head must admit at least one positive integer-ms tail.
            # This depends only on declared time/type permission, never occupancy.
            last_tail_us = spec.duration_us // 1000 * 1000
            self.grid_us = self.grid_us[self.grid_us < last_tail_us]
            self.copy_ids = tuple(i for i in self.copy_ids if spec.copy_times[i] < last_tail_us)
        self.events = []
        self.phase = 'group'
        self.last_onset_us = -1
        self.onset_us = -1
        self.head_block = self.head_cell = -1
        self.tail_block = self.tail_cell = -1
        self.kind = self.end_us = -1
        self.members = 0
        self.decisions = 0
        self.done = False

    def _head_grid(self):
        result = self.grid_us[self.grid_us > self.last_onset_us] // 1000
        if self.phase in ('head_cell', 'head_ms'):
            result = result[result // 2000 == self.head_block]
        if self.phase == 'head_ms':
            result = result[result % 2000 // 40 == self.head_cell]
        return result

    def _tail_grid(self):
        # A syntactic positive hold, never a restriction based on another note's occupancy.
        low = self.onset_us // 1000 + 1
        high = self.spec.duration_us // 1000
        if self.phase in ('tail_cell', 'tail_ms'):
            low = max(low, self.tail_block * 2000)
            high = min(high, (self.tail_block + 1) * 2000 - 1)
        if self.phase == 'tail_ms':
            low = max(low, self.tail_block * 2000 + self.tail_cell * 40)
            high = min(high, self.tail_block * 2000 + (self.tail_cell + 1) * 40 - 1)
        return np.arange(low, high + 1, dtype=np.int64)

    def allowed_values(self):
        if self.done:
            return ()
        if self.phase == 'group':
            possible = bool(len(self._head_grid())) or any(self.spec.copy_times[i] > self.last_onset_us for i in self.copy_ids)
            return (END, NEW) if possible else (END,)
        if self.phase == 'head_block':
            grid = set(map(int, np.unique(self._head_grid() // 2000)))
            grid.update(self.spec.time_blocks + i for i in self.copy_ids
                        if self.spec.copy_times[i] > self.last_onset_us)
            return tuple(sorted(grid))
        if self.phase in ('head_cell', 'head_ms'):
            values = self._head_grid()
            values = values % 2000 // 40 if self.phase == 'head_cell' else values % 40
            return tuple(map(int, np.unique(values)))
        if self.phase == 'member':
            return (END, NEW) if self.members else (NEW,)
        if self.phase == 'kind':
            kinds = self.spec.allowed_types if self.spec.allowed_types is not None else self.spec.mode_types
            return tuple(TYPES.index(t) for t in kinds if t != 'hold' or self.onset_us // 1000 + 1 <= self.spec.duration_us // 1000)
        if self.phase.startswith('tail_'):
            values = self._tail_grid()
            values = values // 2000 if self.phase == 'tail_block' else values % 2000 // 40 if self.phase == 'tail_cell' else values % 40
            return tuple(map(int, np.unique(values)))
        if self.phase == 'lane':
            return (0, 1, 2, 3)
        raise AssertionError('unknown grammar phase')

    def _start_members(self, onset_us):
        self.onset_us = onset_us
        self.members = 0
        self.phase = 'member'

    def _commit(self, lane):
        if len(self.events) >= self.spec.max_events:
            raise CanvasOverflow('event sequence exceeds configured event bound')
        event = RhythmEvent(self.onset_us, self.end_us, TYPES[self.kind], lane)
        validate_event(event, self.spec)
        self.events.append(event)
        self.members += 1
        self.kind = self.end_us = self.tail_block = self.tail_cell = -1
        self.phase = 'member'

    def _finish_fields(self):
        if self.spec.mode == 'mania_4k':
            self.phase = 'lane'
        else:
            self._commit(-1)

    def apply(self, decision):
        if self.done or decision.field != self.phase:
            raise ValueError('unexpected decision phase or payload after END_CORE')
        if self.decisions >= self.spec.max_decisions:
            raise CanvasOverflow('event sequence exceeds configured decision bound; no implicit EOS')
        if type(decision.value) is not int or decision.value not in self.allowed_values():
            raise ValueError('decision violates grammar/time/type permission')
        value = decision.value
        self.decisions += 1
        if self.phase == 'group':
            if value == END:
                self.done = True
            else:
                self.phase = 'head_block'
        elif self.phase == 'head_block':
            self.head_block = value
            if value >= self.spec.time_blocks:
                self._start_members(self.spec.copy_times[value - self.spec.time_blocks])
            else:
                self.phase = 'head_cell'
        elif self.phase == 'head_cell':
            self.head_cell = value
            self.phase = 'head_ms'
        elif self.phase == 'head_ms':
            self._start_members((self.head_block * 2000 + self.head_cell * 40 + value) * 1000)
        elif self.phase == 'member':
            if value == END:
                self.last_onset_us = self.onset_us
                self.onset_us = self.head_block = self.head_cell = -1
                self.members = 0
                self.phase = 'group'
            else:
                self.phase = 'kind'
        elif self.phase == 'kind':
            self.kind = value
            if TYPES[value] == 'hold':
                self.phase = 'tail_block'
            else:
                self.end_us = self.onset_us
                self._finish_fields()
        elif self.phase == 'tail_block':
            self.tail_block = value
            self.phase = 'tail_cell'
        elif self.phase == 'tail_cell':
            self.tail_cell = value
            self.phase = 'tail_ms'
        elif self.phase == 'tail_ms':
            self.end_us = (self.tail_block * 2000 + self.tail_cell * 40 + value) * 1000
            self._finish_fields()
        elif self.phase == 'lane':
            self._commit(value)

    def model_view(self, *, history_enabled):
        """Only committed past events and current partial fields; no hidden target."""
        past = self.known_carry + tuple(self.events) if history_enabled else ()
        at = self.onset_us if self.onset_us >= 0 else max(self.spec.layout.start_us, self.last_onset_us)
        holds = [e for e in past if e.type == 'hold' and e.onset_us <= at < e.end_us]
        occupied = np.zeros((4, 3), np.int64)
        if self.spec.mode == 'mania_4k':
            for event in holds:
                occupied[event.lane, 0] += 1
                occupied[event.lane, 1] = max(occupied[event.lane, 1], event.end_us)
            for event in self.events if history_enabled else ():
                if self.onset_us >= 0 and event.onset_us == self.onset_us:
                    occupied[event.lane, 2] += 1
        # Head parts are cleared once the group ends. Completed event attributes
        # must never remain in A's partial-member path after a commit.
        syntax = (FIELDS.index(self.phase), self.last_onset_us, self.onset_us,
                  self.head_block, self.head_cell, self.kind, self.tail_block,
                  self.tail_cell, self.end_us, self.members, len(self.events))
        return dict(syntax=np.asarray(syntax, np.int64), history_known=bool(history_enabled),
                    completed_events=past, occupancy=occupied, active_hold_total=len(holds),
                    allowed_values=self.allowed_values(), done=self.done)


def encode_sequence(events, spec, *, known_carry=()):
    state = EventSequenceState(spec, known_carry=known_carry)
    ordered = sorted(events, key=lambda e: (e.onset_us, e.lane if spec.mode == 'mania_4k' else TYPES.index(e.type),
                                          e.end_us, TYPES.index(e.type)))
    for event in ordered:
        validate_event(event, spec)
    decisions = []

    def emit(field, value):
        decision = Decision(field, int(value))
        state.apply(decision)
        decisions.append(decision)

    for onset, members in groupby(ordered, key=lambda e: e.onset_us):
        emit('group', NEW)
        if onset in spec.copy_times:
            emit('head_block', spec.time_blocks + spec.copy_times.index(onset))
        else:
            if onset % 1000:
                raise ValueError('non-millisecond onset requires authorized COPY_TIME')
            onset_ms = onset // 1000
            emit('head_block', onset_ms // 2000)
            emit('head_cell', onset_ms % 2000 // 40)
            emit('head_ms', onset_ms % 40)
        for event in members:
            emit('member', NEW)
            emit('kind', TYPES.index(event.type))
            if event.type == 'hold':
                tail_ms = event.end_us // 1000
                emit('tail_block', tail_ms // 2000)
                emit('tail_cell', tail_ms % 2000 // 40)
                emit('tail_ms', tail_ms % 40)
            if spec.mode == 'mania_4k':
                emit('lane', event.lane)
        emit('member', END)
    emit('group', END)
    return tuple(decisions)


def decode_sequence(decisions, spec, *, known_carry=()):
    state = EventSequenceState(spec, known_carry=known_carry)
    for decision in decisions:
        state.apply(decision)
    if not state.done:
        raise ValueError('incomplete sequence: no END_CORE')
    return tuple(sorted(state.events))


def teacher_views(decisions, spec, *, history_enabled, known_carry=()):
    """Training-only strict-prefix views, captured before each supervised token."""
    state = EventSequenceState(spec, known_carry=known_carry)
    for decision in decisions:
        view = state.model_view(history_enabled=history_enabled)
        yield view, decision
        state.apply(decision)
    if not state.done:
        raise ValueError('incomplete teacher sequence')
