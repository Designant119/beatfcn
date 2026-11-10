"""Complete onset-group grammar for the full-scale event Transformer.

Input events are the existing millisecond model view, not rounded here. Exact
authorized anchor times use COPY_TIME. Hold ends never move the onset cursor.
All decoder query metadata and legal supports depend only on the emitted prefix.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import groupby

import numpy as np

from beatfcn.events.canonical import TYPES
from .event_canvas import RhythmEvent

FIELDS = ('group', 'head_block', 'head_cell', 'head_ms', 'member',
          'kind', 'tail_block', 'tail_cell', 'tail_ms')
END, NEW = 0, 1
PAD, BOS = 0, 1


@dataclass(frozen=True)
class EventVocabulary:
    """Token layout plus opt-in grammar coordinate fixes.

    ``tail_query``: hold-tail cell/ms decisions are queried at the already
    emitted tail block/cell time instead of the onset (heads always refined
    their query time; tails did not, so the decoder looked up audio at the
    onset while choosing a tail that may lie seconds later).

    ``relative_blocks``: head/tail block token VALUES count 2s blocks from the
    request's own block ``start_us // 2_000_000`` instead of absolute song
    blocks. Every model input time is request-relative, so absolute block ids
    forced the decoder to memorise song position. Decoded events stay absolute
    microseconds; copy-time tokens and vocabulary size are unchanged.

    The model is a timing extractor: mania uses the same member/kind/tail
    grammar as Phira with ``lane == -1`` -- no lane is ever predicted,
    supervised or gated (incoming ``known_carry`` holds stay accepted as timing
    context with their lane ignored). Lane placement is the chart editor's
    job, not the model's. The historical 81-token 4-lane pattern field and
    every checkpoint trained with it are retired; the vocabulary shrank
    1985 -> 1904, so old checkpoints fail the strict load by construction.

    Both flags default False so historical token streams of THIS grammar
    reproduce.
    The same vocabulary must be passed to encode_events/teacher_sequence/
    EventState and to the model that decodes them.
    """
    max_duration_ms: int = 1_200_000
    max_copy_times: int = 512
    tail_query: bool = False
    relative_blocks: bool = False

    def block_base(self, request):
        """Absolute 2s block that block token value 0 denotes for ``request``."""
        return request.start_us // 2_000_000 if self.relative_blocks else 0

    @property
    def blocks(self):
        return self.max_duration_ms // 2000 + 1

    @property
    def sizes(self):
        return (2, self.blocks + self.max_copy_times, 50, 40, 2, 4,
                self.blocks, 50, 40)

    @property
    def size(self):
        return 2 + sum(self.sizes)

    def token(self, field, value):
        i = FIELDS.index(field)
        if type(value) is not int or not 0 <= value < self.sizes[i]:
            raise ValueError(f'Out-of-range {field} value: {value}')
        return 2 + sum(self.sizes[:i]) + value

    def decision(self, token):
        if type(token) is not int:
            raise ValueError('Token must be an integer')
        offset = 2
        for field, size in zip(FIELDS, self.sizes):
            if offset <= token < offset + size:
                return field, token - offset
            offset += size
        raise ValueError('Not an event decision token')


@dataclass(frozen=True)
class EventRequest:
    mode: str
    start_us: int
    end_us: int
    duration_us: int
    copy_times: tuple = ()
    known_carry: tuple = ()
    allowed_types: tuple | None = None
    max_events: int = 4096
    max_tokens: int = 32768

    def __post_init__(self):
        if self.mode not in ('mania_4k', 'phira'):
            raise ValueError('Unsupported mode')
        if any(type(t) is not int for t in (self.start_us, self.end_us, self.duration_us)):
            raise ValueError('Time coordinates must be integer microseconds')
        if not 0 <= self.start_us < self.end_us <= self.duration_us:
            raise ValueError('Invalid request interval')
        if self.max_events <= 0 or self.max_tokens <= 0:
            raise ValueError('Invalid resource capacity')
        legal_types = ('tap', 'hold') if self.mode == 'mania_4k' else TYPES
        if self.allowed_types is not None and (not self.allowed_types or not set(self.allowed_types) <= set(legal_types)):
            raise ValueError('Invalid type permission')
        if len(set(self.copy_times)) != len(self.copy_times) or any(type(t) is not int for t in self.copy_times):
            raise ValueError('Copy times must be unique integer microseconds')
        for e in self.known_carry:
            if (e.type != 'hold' or not 0 <= e.onset_us < self.start_us < e.end_us <= self.duration_us
                    or (self.mode == 'mania_4k' and e.lane not in (*range(4), -1))):
                raise ValueError('Invalid authorized incoming hold')


class EventState:
    def __init__(self, request, vocab=None):
        self.request, self.vocab = request, vocab or EventVocabulary()
        if request.duration_us > self.vocab.max_duration_ms * 1000 or len(request.copy_times) > self.vocab.max_copy_times:
            raise ValueError('Request exceeds the declared vocabulary; do not truncate it')
        self.phase, self.done = 'group', False
        # head_block/tail_block attributes stay ABSOLUTE song blocks; only the
        # emitted token values are shifted by block_base.
        self.block_base = self.vocab.block_base(request)
        self.last_onset = self.onset = -1
        self.head_block = self.head_cell = self.tail_block = self.tail_cell = -1
        self.kind = None
        self.group_members = 0
        self.events, self.tokens = [], []
        # known_carry holds are timing context only; their lane is never inspected
        # (lane placement is the chart editor's job, not the grammar's).

    def _heads_ms(self):
        lo = (max(self.request.start_us, self.last_onset + 1) + 999) // 1000
        hi = (self.request.end_us - 1) // 1000
        if self.request.allowed_types == ('hold',):
            hi = min(hi, self.request.duration_us // 1000 - 1)
        if self.phase in ('head_cell', 'head_ms'):
            lo, hi = max(lo, self.head_block * 2000), min(hi, self.head_block * 2000 + 1999)
        if self.phase == 'head_ms':
            base = self.head_block * 2000 + self.head_cell * 40
            lo, hi = max(lo, base), min(hi, base + 39)
        return lo, hi

    def _tails_ms(self):
        lo, hi = self.onset // 1000 + 1, self.request.duration_us // 1000
        if self.phase in ('tail_cell', 'tail_ms'):
            lo, hi = max(lo, self.tail_block * 2000), min(hi, self.tail_block * 2000 + 1999)
        if self.phase == 'tail_ms':
            base = self.tail_block * 2000 + self.tail_cell * 40
            lo, hi = max(lo, base), min(hi, base + 39)
        return lo, hi

    def allowed_values(self):
        if self.done:
            return ()
        lo, hi = self._heads_ms()
        copies = tuple(self.vocab.blocks + i for i, t in enumerate(self.request.copy_times)
                       if max(self.request.start_us, self.last_onset + 1) <= t < self.request.end_us
                       and (self.request.allowed_types != ('hold',) or t // 1000 < self.request.duration_us // 1000))
        if self.phase == 'group':
            return (END, NEW) if lo <= hi or copies else (END,)
        if self.phase == 'head_block':
            base = self.block_base
            return tuple(range(lo // 2000 - base, hi // 2000 - base + 1)) + copies if lo <= hi else copies
        if self.phase == 'head_cell':
            return tuple(range(lo % 2000 // 40, hi % 2000 // 40 + 1)) if lo <= hi else ()
        if self.phase == 'head_ms':
            return tuple(range(lo % 40, hi % 40 + 1)) if lo <= hi else ()
        hold_possible = self.onset // 1000 + 1 <= self.request.duration_us // 1000
        types = self.request.allowed_types or (('tap', 'hold') if self.request.mode == 'mania_4k' else TYPES)
        if self.phase == 'member':
            return (END, NEW) if self.group_members else (NEW,)
        if self.phase == 'kind':
            return tuple(i for i, kind in enumerate(TYPES) if kind in types and (hold_possible or kind != 'hold'))
        lo, hi = self._tails_ms()
        if self.phase == 'tail_block':
            return tuple(range(lo // 2000 - self.block_base, hi // 2000 - self.block_base + 1))
        if self.phase == 'tail_cell':
            return tuple(range(lo % 2000 // 40, hi % 2000 // 40 + 1))
        if self.phase == 'tail_ms':
            return tuple(range(lo % 40, hi % 40 + 1))
        raise ValueError('Unknown grammar phase')

    def allowed_tokens(self):
        return tuple(self.vocab.token(self.phase, int(v)) for v in self.allowed_values())

    def query(self):
        """Prefix-only coordinates before predicting the next token."""
        t = self.onset if self.onset >= 0 else max(self.request.start_us, self.last_onset)
        if self.phase == 'head_cell':
            t = self.head_block * 2_000_000
        elif self.phase == 'head_ms':
            t = self.head_block * 2_000_000 + self.head_cell * 40_000
        elif self.vocab.tail_query and self.phase == 'tail_cell':
            t = self.tail_block * 2_000_000
        elif self.vocab.tail_query and self.phase == 'tail_ms':
            t = self.tail_block * 2_000_000 + self.tail_cell * 40_000
        return FIELDS.index(self.phase), (t - self.request.start_us) / 1e6

    def _add(self, kind, end):
        if len(self.events) >= self.request.max_events:
            raise ValueError('Event capacity exceeded; no truncation allowed')
        self.events.append(RhythmEvent(self.onset, end, kind, -1))
        self.group_members += 1

    def _start(self, onset):
        self.onset, self.group_members = onset, 0
        self.phase = 'member'

    def _finish_group(self):
        self.last_onset, self.onset, self.phase = self.onset, -1, 'group'
        self.head_block = self.head_cell = -1

    def apply(self, token):
        if self.done or len(self.tokens) >= self.request.max_tokens:
            raise ValueError('Complete stream or token capacity exceeded')
        field, value = self.vocab.decision(token)
        if field != self.phase or value not in self.allowed_values():
            raise ValueError(f'Illegal {field}:{value} in {self.phase}')
        self.tokens.append(token)
        if field == 'group':
            if value == END:
                self.done = True
            else:
                self.phase = 'head_block'
        elif field == 'head_block':
            if value >= self.vocab.blocks:
                self._start(self.request.copy_times[value - self.vocab.blocks])
            else:
                self.head_block, self.phase = value + self.block_base, 'head_cell'
        elif field == 'head_cell':
            self.head_cell, self.phase = value, 'head_ms'
        elif field == 'head_ms':
            self._start((self.head_block * 2000 + self.head_cell * 40 + value) * 1000)
        elif field == 'member':
            if value == END:
                self._finish_group()
            else:
                self.phase = 'kind'
        elif field == 'kind':
            self.kind = TYPES[value]
            if self.kind == 'hold':
                self.phase = 'tail_block'
            else:
                self._add(self.kind, self.onset)
                self.phase = 'member'
        elif field == 'tail_block':
            self.tail_block, self.phase = value + self.block_base, 'tail_cell'
        elif field == 'tail_cell':
            self.tail_cell, self.phase = value, 'tail_ms'
        elif field == 'tail_ms':
            end = (self.tail_block * 2000 + self.tail_cell * 40 + value) * 1000
            self._add('hold', end)
            self.tail_block = self.tail_cell = -1
            self.phase = 'member'


@lru_cache(maxsize=64)
def _field_offsets(vocab):
    """First token id of every grammar field (``vocab.token(field, 0)``)."""
    return {field: vocab.token(field, 0) for field in FIELDS}


class FastEventState(EventState):
    """The same grammar as EventState, with O(1) legal-support descriptors.

    Inference-only helper for batched decoding. ``allowed_values`` returns the
    identical value sequence (same order) as ``EventState.allowed_values`` but
    as ``range`` objects / cached tuples, so ``apply``'s legality check and the
    support construction no longer materialise hundreds of tokens per step.
    All transitions are inherited unchanged from EventState. Equivalence with
    the reference state is covered by randomized differential tests.
    """

    def _copies(self):
        request = self.request
        if not request.copy_times:
            return ()
        return tuple(self.vocab.blocks + i for i, t in enumerate(request.copy_times)
                     if max(request.start_us, self.last_onset + 1) <= t < request.end_us
                     and (request.allowed_types != ('hold',) or t // 1000 < request.duration_us // 1000))

    def _types(self):
        return self.request.allowed_types or (('tap', 'hold') if self.request.mode == 'mania_4k' else TYPES)

    def allowed_values(self):
        if self.done:
            return ()
        phase = self.phase
        if phase in ('group', 'head_block', 'head_cell', 'head_ms'):
            lo, hi = self._heads_ms()
            if phase == 'head_cell':
                return range(lo % 2000 // 40, hi % 2000 // 40 + 1) if lo <= hi else ()
            if phase == 'head_ms':
                return range(lo % 40, hi % 40 + 1) if lo <= hi else ()
            copies = self._copies()
            if phase == 'group':
                return range(END, NEW + 1) if lo <= hi or copies else range(END, END + 1)
            if lo > hi:
                return copies
            blocks = range(lo // 2000 - self.block_base, hi // 2000 - self.block_base + 1)
            return tuple(blocks) + copies if copies else blocks
        if phase == 'member':
            return range(END, NEW + 1) if self.group_members else range(NEW, NEW + 1)
        if phase == 'kind':
            hold_possible = self.onset // 1000 + 1 <= self.request.duration_us // 1000
            types = self._types()
            return tuple(i for i, kind in enumerate(TYPES) if kind in types and (hold_possible or kind != 'hold'))
        if phase in ('tail_block', 'tail_cell', 'tail_ms'):
            lo, hi = self._tails_ms()
            if phase == 'tail_block':
                return range(lo // 2000 - self.block_base, hi // 2000 - self.block_base + 1)
            if phase == 'tail_cell':
                return range(lo % 2000 // 40, hi % 2000 // 40 + 1)
            return range(lo % 40, hi % 40 + 1)
        raise ValueError('Unknown grammar phase')

    def allowed_spec(self):
        """``(first, count, tokens)`` describing ``allowed_tokens()`` exactly.

        ``tokens is None`` means the support is the ascending token range
        ``first .. first+count-1``; otherwise ``tokens`` is the explicit tuple
        (same order as ``allowed_tokens()``). ``count == 0`` is a dead end.
        """
        offset = _field_offsets(self.vocab)[self.phase]
        values = self.allowed_values()
        if isinstance(values, range):
            return offset + values.start, len(values), None
        if not values:
            return offset, 0, ()
        return offset + values[0], len(values), tuple(offset + v for v in values)


def encode_events(events, request, vocab=None):
    state = EventState(request, vocab)

    def emit(field, value):
        state.apply(state.vocab.token(field, int(value)))

    def tail(event):
        if event.end_us % 1000 or not event.onset_us < event.end_us <= request.duration_us:
            raise ValueError('Hold end must be an exact positive model-view millisecond')
        ms = event.end_us // 1000
        for field, value in [('tail_block', ms // 2000 - state.block_base), ('tail_cell', ms % 2000 // 40), ('tail_ms', ms % 40)]:
            emit(field, value)

    ordered = sorted(events)
    for event in ordered:
        if (type(event.onset_us) is not int or type(event.end_us) is not int
                or not request.start_us <= event.onset_us < request.end_us
                or event.type not in (('tap', 'hold') if request.mode == 'mania_4k' else TYPES)
                or (event.type != 'hold' and event.end_us != event.onset_us)
                # mania carries no lane decision; human-chart lanes are accepted and ignored
                or (event.lane not in (*range(4), -1) if request.mode == 'mania_4k' else event.lane != -1)):
            raise ValueError('Invalid model-view event')
    for onset, group in groupby(ordered, key=lambda e: e.onset_us):
        members = list(group)
        emit('group', NEW)
        if onset in request.copy_times:
            emit('head_block', state.vocab.blocks + request.copy_times.index(onset))
        else:
            if onset % 1000:
                raise ValueError('Sub-ms heads require an authorized copy anchor, not a hidden target')
            ms = onset // 1000
            for field, value in [('head_block', ms // 2000 - state.block_base), ('head_cell', ms % 2000 // 40), ('head_ms', ms % 40)]:
                emit(field, value)
        for event in sorted(members, key=lambda e: (TYPES.index(e.type), e.end_us)):
            emit('member', NEW)
            emit('kind', TYPES.index(event.type))
            if event.type == 'hold':
                tail(event)
        emit('member', END)
    emit('group', END)
    check = [RhythmEvent(e.onset_us, e.end_us, e.type, -1) for e in ordered] if request.mode == 'mania_4k' else ordered
    if tuple(sorted(state.events)) != tuple(check):
        raise AssertionError('Grammar did not preserve the target multiset')
    return tuple(state.tokens)


def teacher_sequence(tokens, request, vocab=None):
    state = EventState(request, vocab)
    previous, fields, times, masks = [], [], [], []
    for token in tokens:
        field, time = state.query()
        previous.append(state.tokens[-1] if state.tokens else BOS)
        fields.append(field)
        times.append(time)
        mask = np.zeros(state.vocab.size, dtype=bool)
        mask[list(state.allowed_tokens())] = True
        masks.append(mask)
        state.apply(token)
    if not state.done:
        raise ValueError('Missing END_CORE')
    return dict(tokens=np.asarray(previous, np.int64), fields=np.asarray(fields, np.int64),
                times=np.asarray(times, np.float32), targets=np.asarray(tokens, np.int64),
                allowed=np.asarray(masks), valid=np.ones(len(tokens), bool))
