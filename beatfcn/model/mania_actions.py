"""Lossless four-key action codes; an isolated representation, not a decoder.

Each base-three digit belongs to a physical key: 0 no NEW note at this onset,
1 tap start, 2 hold start. Zero does not release an earlier hold or mean idle;
sustained occupancy is represented by complete note intervals separately.
An empty completed group is code 0. Unknown is -1 and cannot be decoded.
Phira's rhythm multiplicity is deliberately outside this four-key contract.
"""
from dataclasses import dataclass

from .event_canvas import RhythmEvent

UNKNOWN_ACTION = -1
ACTION_COUNT = 81


def action_states(code: int) -> tuple[int, int, int, int]:
    if type(code) is not int or not 0 <= code < ACTION_COUNT:
        raise ValueError('requires a completed four-key action code, not unknown')
    return tuple((code // (3 ** lane)) % 3 for lane in range(4))


def action_size(code: int) -> int:
    return sum(state != 0 for state in action_states(code))


@dataclass(frozen=True)
class ManiaAction:
    code: int
    # Absolute integer microseconds. Only hold keys have a tail payload.
    tails_us: tuple[int | None, int | None, int | None, int | None]

    def __post_init__(self):
        states = action_states(self.code)
        if type(self.tails_us) is not tuple or len(self.tails_us) != 4:
            raise ValueError('four immutable physical-key tail slots required')
        for state, tail in zip(states, self.tails_us):
            if state == 2:
                if type(tail) is not int:
                    raise ValueError('hold requires an integer tail')
            elif tail is not None:
                raise ValueError('no-new-note/tap action must not carry a hold tail')


def encode_action(events: tuple[RhythmEvent, ...] | list[RhythmEvent], *, onset_us: int) -> ManiaAction:
    if type(onset_us) is not int:
        raise TypeError('onset must be integer microseconds')
    states, tails = [0] * 4, [None] * 4
    for event in events:
        if event.onset_us != onset_us or type(event.onset_us) is not int or type(event.end_us) is not int:
            raise ValueError('one exact integer onset per action group')
        if type(event.lane) is not int or event.lane not in range(4) or event.type not in ('tap', 'hold'):
            raise ValueError('four-key mania actions only')
        if states[event.lane]:
            raise ValueError('duplicate same-onset key; cannot silently merge or drop')
        if event.type == 'hold':
            if event.end_us <= onset_us:
                raise ValueError('hold must have positive duration')
            states[event.lane], tails[event.lane] = 2, event.end_us
        else:
            if event.end_us != onset_us:
                raise ValueError('tap must end at onset')
            states[event.lane] = 1
    return ManiaAction(sum(state * 3 ** lane for lane, state in enumerate(states)), tuple(tails))


def decode_action(action: ManiaAction, *, onset_us: int) -> tuple[RhythmEvent, ...]:
    if type(onset_us) is not int:
        raise TypeError('onset must be integer microseconds')
    events = []
    for lane, state in enumerate(action_states(action.code)):
        if state == 0:
            continue
        tail = action.tails_us[lane] if state == 2 else onset_us
        if state == 2 and tail <= onset_us:
            raise ValueError('hold tail must follow the supplied head; no clipping')
        events.append(RhythmEvent(onset_us, tail, 'hold' if state == 2 else 'tap', lane))
    return tuple(sorted(events))
