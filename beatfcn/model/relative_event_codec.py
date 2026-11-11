"""Opt-in relative head coordinates; original event, tail and COPY semantics.

No training or generation entry point selects this module by default. The
absolute codec remains the validation authority for the input event multiset.
"""
import numpy as np

from .full_event_codec import BOS, EventState, encode_events as absolute_encode


class RelativeHeadState(EventState):
    def head_base_ms(self):
        return (self.last_onset if self.last_onset >= 0 else self.request.start_us) // 1000

    def _heads_ms(self):
        lo = (max(self.request.start_us, self.last_onset + 1) + 999) // 1000
        hi = (self.request.end_us - 1) // 1000
        if self.request.allowed_types == ('hold',):
            hi = min(hi, self.request.duration_us // 1000 - 1)
        base = self.head_base_ms()
        lo, hi = lo - base, hi - base
        if self.phase in ('head_cell', 'head_ms'):
            lo, hi = max(lo, self.head_block * 2000), min(hi, self.head_block * 2000 + 1999)
        if self.phase == 'head_ms':
            coarse = self.head_block * 2000 + self.head_cell * 40
            lo, hi = max(lo, coarse), min(hi, coarse + 39)
        return lo, hi

    def query(self):
        field, time = super().query()
        if self.phase in ('head_cell', 'head_ms'):
            time += self.head_base_ms() / 1000
        return field, time

    def _start(self, onset):
        # Numeric ms is relative, while head_block COPY supplies exact physical us.
        if self.phase == 'head_ms':
            onset += self.head_base_ms() * 1000
        super()._start(onset)


def encode_events(events, request, vocab=None):
    events = tuple(events)
    absolute = absolute_encode(events, request, vocab)
    state = RelativeHeadState(request, vocab)
    i = 0
    while i < len(absolute):
        field, value = state.vocab.decision(absolute[i])
        if field == 'head_block' and value < state.vocab.blocks:
            cell_field, cell = state.vocab.decision(absolute[i+1])
            ms_field, ms = state.vocab.decision(absolute[i+2])
            if (cell_field, ms_field) != ('head_cell', 'head_ms'):
                raise AssertionError('Absolute encoder changed its three-field head grammar')
            delta = value * 2000 + cell * 40 + ms - state.head_base_ms()
            for name, part in [('head_block', delta // 2000),
                               ('head_cell', delta % 2000 // 40), ('head_ms', delta % 40)]:
                state.apply(state.vocab.token(name, int(part)))
            i += 3
        else:
            state.apply(absolute[i])
            i += 1
    shape = lambda e: (e.onset_us, e.end_us, e.type)
    if (not state.done or sorted(shape(e) for e in state.events) != sorted(shape(e) for e in events)
            or len(state.tokens) != len(absolute)):
        raise AssertionError('Relative coordinate conversion changed the event multiset or token count')
    return tuple(state.tokens)


def teacher_sequence(tokens, request, vocab=None):
    state = RelativeHeadState(request, vocab)
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
