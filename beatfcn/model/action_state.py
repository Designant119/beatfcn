"""Lossless rhythm action boundaries, isolated from production model generation.

Only an explicit action prefix enters State. Future hold ends are actions to be
predicted, never fields stored in active holds. Encoding builds supervised labels;
it must not be used to create the input of a blank generation request.
"""
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import product

from beatfcn.events.canonical import MODE_TYPES, TYPES
from beatfcn.model.event_canvas import RhythmEvent


# Per-key digits in an atomic four-key action, least significant key first.
KEEP, TAP, START, RELEASE, RELEASE_TAP, RELEASE_START = range(6)


@dataclass(frozen=True)
class State:
    mode: str
    last_time_us: int | None = None
    lane_starts: tuple[int | None, ...] = (None, None, None, None)
    hold_groups: tuple[tuple[int, int], ...] = ()  # Phira (observed start, active count)

    def __post_init__(self):
        if self.mode not in MODE_TYPES or (self.last_time_us is not None and type(self.last_time_us) is not int):
            raise ValueError('invalid mode or state clock')
        if type(self.lane_starts) is not tuple or len(self.lane_starts) != 4 or type(self.hold_groups) is not tuple:
            raise ValueError('state containers must be immutable canonical tuples')
        for start in self.lane_starts:
            if start is not None and (type(start) is not int or self.last_time_us is None or start > self.last_time_us):
                raise ValueError('hold start is not in the observed prefix')
        previous = None
        for pair in self.hold_groups:
            if (type(pair) is not tuple or len(pair) != 2 or any(type(x) is not int for x in pair)
                    or pair[1] <= 0 or self.last_time_us is None or pair[0] > self.last_time_us
                    or previous is not None and pair[0] <= previous):
                raise ValueError('invalid or noncanonical active hold group')
            previous = pair[0]
        if self.mode == 'mania_4k' and self.hold_groups or self.mode == 'phira' and any(x is not None for x in self.lane_starts):
            raise ValueError('cross-mode state fields')

    @property
    def active_holds(self):
        return sum(x is not None for x in self.lane_starts) if self.mode == 'mania_4k' else sum(n for _, n in self.hold_groups)


@dataclass(frozen=True)
class ManiaAction:
    time_us: int
    code: int


@dataclass(frozen=True)
class PhiraAction:
    time_us: int
    starts: tuple[int, int, int, int]  # tap, hold, flick, drag
    releases: tuple[tuple[int, int], ...] = ()  # visible start-group pointer, count


@dataclass(frozen=True)
class Replay:
    events: tuple[RhythmEvent, ...]
    state: State


def mania_code(digits):
    if len(digits) != 4 or any(type(x) is not int or x not in range(6) for x in digits):
        raise ValueError('four per-key action digits required')
    return sum(value * 6 ** lane for lane, value in enumerate(digits))


def mania_digits(code):
    if type(code) is not int or not 0 <= code < 6 ** 4:
        raise ValueError('invalid joint action code')
    return tuple(code // 6 ** lane % 6 for lane in range(4))


@lru_cache(maxsize=16)
def _mania_support(occupied_mask):
    choices = [(KEEP, RELEASE, RELEASE_TAP, RELEASE_START) if occupied_mask & (1 << lane)
               else (KEEP, TAP, START) for lane in range(4)]
    return tuple(sorted(mania_code(x) for x in product(*choices) if any(x)))


def legal_mania_actions(state):
    if state.mode != 'mania_4k':
        raise ValueError('mania support requested for another mode')
    occupied = sum(1 << lane for lane, start in enumerate(state.lane_starts) if start is not None)
    return _mania_support(occupied)


def apply_action(state, action):
    """Atomically advance a visible/generated prefix or reject the whole action."""
    if not isinstance(action, (ManiaAction, PhiraAction)) or type(action.time_us) is not int:
        raise ValueError('invalid action or noninteger clock')
    t = action.time_us
    if state.last_time_us is not None and t <= state.last_time_us:
        raise ValueError('boundaries must be strictly increasing; merge simultaneous actions')
    completed = []
    if state.mode == 'mania_4k':
        if not isinstance(action, ManiaAction) or type(action.code) is not int or action.code not in legal_mania_actions(state):
            raise ValueError('joint action violates the visible occupancy state')
        starts = list(state.lane_starts)
        for lane, digit in enumerate(mania_digits(action.code)):
            if digit in (RELEASE, RELEASE_TAP, RELEASE_START):
                completed.append(RhythmEvent(starts[lane], t, 'hold', lane))
                starts[lane] = None
            if digit in (TAP, RELEASE_TAP):
                completed.append(RhythmEvent(t, t, 'tap', lane))
            elif digit in (START, RELEASE_START):
                starts[lane] = t
        updated = State(state.mode, t, tuple(starts))
    else:
        if (not isinstance(action, PhiraAction) or type(action.starts) is not tuple or len(action.starts) != 4
                or any(type(n) is not int or n < 0 for n in action.starts) or type(action.releases) is not tuple):
            raise ValueError('invalid Phira action counts')
        groups = dict(state.hold_groups)
        previous = None
        for pair in action.releases:
            if (type(pair) is not tuple or len(pair) != 2 or any(type(x) is not int for x in pair)
                    or pair[1] <= 0 or previous is not None and pair[0] <= previous):
                raise ValueError('invalid or repeated release pointer')
            start, count = pair
            previous = start
            if start not in groups or count > groups[start] or start >= t:
                raise ValueError('release pointer/count is absent from the visible state')
            completed.extend(RhythmEvent(start, t, 'hold') for _ in range(count))
            groups[start] -= count
            if not groups[start]:
                del groups[start]
        if not any(action.starts) and not action.releases:
            raise ValueError('an empty action is not a boundary')
        for kind, count in zip(TYPES, action.starts):
            if kind == 'hold':
                if count:
                    groups[t] = count
            else:
                completed.extend(RhythmEvent(t, t, kind) for _ in range(count))
        updated = State(state.mode, t, hold_groups=tuple(sorted(groups.items())))
    return Replay(tuple(sorted(completed)), updated)


def replay(actions, mode, *, initial=None, require_closed=True):
    state = State(mode) if initial is None else initial
    if state.mode != mode:
        raise ValueError('incoming state mode mismatch')
    completed = []
    for action in actions:
        step = apply_action(state, action)
        completed.extend(step.events)
        state = step.state
    if require_closed and state.active_holds:
        raise ValueError('cannot finalize a chart with unreleased holds')
    return Replay(tuple(sorted(completed)), state)


def encode(events, mode):
    """Compile complete supervised labels; no rounding, truncation or spatial input."""
    if mode not in MODE_TYPES:
        raise ValueError('unsupported mode')
    events = tuple(events)
    starts, releases = defaultdict(list), defaultdict(list)
    for event in events:
        if any(type(x) is not int for x in (event.onset_us, event.end_us, event.lane)):
            raise ValueError('coordinates and lane must be integer')
        if event.type not in MODE_TYPES[mode] or (mode == 'mania_4k' and event.lane not in range(4)) or (mode == 'phira' and event.lane != -1):
            raise ValueError('invalid mode-specific note')
        if event.type == 'hold':
            if event.end_us <= event.onset_us:
                raise ValueError('hold requires a positive interval')
            releases[event.end_us].append(event)
        elif event.end_us != event.onset_us:
            raise ValueError('non-hold has a duration')
        starts[event.onset_us].append(event)
    actions = []
    for t in sorted(starts.keys() | releases.keys()):
        if mode == 'mania_4k':
            digits = [KEEP] * 4
            for event in releases[t]:
                if digits[event.lane] != KEEP:
                    raise ValueError('multiple releases of one key at one time')
                digits[event.lane] = RELEASE
            attacked = set()
            for event in starts[t]:
                if event.lane in attacked:
                    raise ValueError('multiple heads on one key at one time')
                attacked.add(event.lane)
                digit = TAP if event.type == 'tap' else START
                digits[event.lane] = digit + (3 if digits[event.lane] == RELEASE else 0)
            actions.append(ManiaAction(t, mania_code(digits)))
        else:
            counts = tuple(sum(e.type == kind for e in starts[t]) for kind in TYPES)
            by_start = defaultdict(int)
            for event in releases[t]:
                by_start[event.onset_us] += 1
            actions.append(PhiraAction(t, counts, tuple(sorted(by_start.items()))))
    actions = tuple(actions)
    # This is a label validator, not a repair. A physically inconsistent source
    # fails instead of receiving a convenient alternative lane or shorter hold.
    if replay(actions, mode).events != tuple(sorted(events)):
        raise ValueError('action stream does not reconstruct the source multiset')
    return actions
