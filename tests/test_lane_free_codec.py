"""Timing-only grammar: mania carries no lane decision (lane placement is the editor's).

The 81-token 4-lane pattern field is gone from the vocabulary (1985 -> 1904); mania
encodes through the same member/kind/tail grammar as Phira with ``lane == -1``.
These tests pin that contract.
"""
import pytest

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import (FIELDS, EventRequest, EventState, EventVocabulary, FastEventState,
                                            encode_events, teacher_sequence)

REQ = EventRequest(mode='mania_4k', start_us=1_000_000, end_us=10_000_000, duration_us=10_000_000,
                   copy_times=(1_500_500,), known_carry=(RhythmEvent(500_000, 3_000_000, 'hold', 2),))
CHART = [
    RhythmEvent(1_000_000, 1_000_000, 'tap', 0),
    RhythmEvent(1_000_000, 1_000_000, 'tap', 3),      # chord: two simultaneous keys
    RhythmEvent(1_500_500, 2_500_000, 'hold', 1),     # copy-time head + hold tail
    RhythmEvent(3_000_000, 3_000_000, 'tap', 2),      # inside the carry hold's span
    RhythmEvent(4_000_000, 5_200_000, 'hold', 0),
    RhythmEvent(4_000_000, 4_000_000, 'tap', 1),
]


def test_vocabulary_has_no_lane_pattern_field():
    assert 'mania' not in FIELDS and len(FIELDS) == 9
    assert EventVocabulary().size == 1904
    with pytest.raises(ValueError):
        EventVocabulary().token('mania', 0)


def test_mania_round_trip_strips_lanes():
    tokens = encode_events(CHART, REQ)
    state = EventState(REQ)
    for token in tokens:
        state.apply(token)
    assert state.done and tuple(state.tokens) == tuple(tokens)
    expected = sorted(RhythmEvent(e.onset_us, e.end_us, e.type, -1) for e in CHART)
    assert sorted(state.events) == expected and all(e.lane == -1 for e in state.events)


def test_lane_minus_one_events_encode():
    laneless = [RhythmEvent(e.onset_us, e.end_us, e.type, -1) for e in CHART]
    assert encode_events(laneless, REQ)
    phira = EventRequest('phira', 0, 8_000_000, 10_000_000)
    with pytest.raises(ValueError):  # phira events must stay lane -1
        encode_events([RhythmEvent(1_000_000, 1_000_000, 'tap', 2)], phira)


def test_head_inside_carry_hold_is_representable():
    # No busy-key gating: a tap during an incoming hold's span is plain timing
    # (placement is resolved downstream by the editor).
    events = [RhythmEvent(2_000_000, 2_000_000, 'tap', 2)]  # lane 2 == the carry hold's lane
    assert encode_events(events, REQ)


def test_fast_state_matches_reference():
    tokens = encode_events(CHART, REQ)
    fast, ref = FastEventState(REQ), EventState(REQ)
    for token in tokens:
        assert fast.allowed_tokens() == ref.allowed_tokens()
        fast.apply(token)
        ref.apply(token)
    assert sorted(fast.events) == sorted(ref.events)


def test_teacher_sequence_masks_fit_vocabulary():
    tokens = encode_events(CHART, REQ)
    sequence = teacher_sequence(tokens, REQ)
    assert sequence['allowed'].shape[1] == EventVocabulary().size
    assert len(sequence['targets']) == len(tokens)
    assert all(sequence['allowed'][i, t] for i, t in enumerate(sequence['targets']))


def test_multi_hold_chord_encodes_like_phira():
    events = [RhythmEvent(2_000_000, 3_000_000, 'hold', 0), RhythmEvent(2_000_000, 4_000_000, 'hold', 3),
              RhythmEvent(2_000_000, 2_000_000, 'tap', 1)]
    tokens = encode_events(events, REQ)
    state = EventState(REQ)
    for token in tokens:
        state.apply(token)
    assert sorted((e.onset_us, e.end_us, e.type) for e in state.events) == \
           sorted((e.onset_us, e.end_us, e.type) for e in events)


def test_consensus_reward_matches_lane_free_predictions():
    torch = pytest.importorskip('torch')
    from beatfcn.train.consensus_rl import event_f1
    target = [(1_000_000, 1_000_000, 'tap', 3), (2_000_000, 2_000_000, 'tap', 1)]
    predicted = [(1_000_000, 1_000_000, 'tap', -1), (2_000_000, 2_000_000, 'tap', -1)]
    assert event_f1(predicted, target) == 1.
    wrong_time = [(1_500_000, 1_500_000, 'tap', -1), (2_500_000, 2_500_000, 'tap', -1)]
    assert event_f1(wrong_time, target) == 0.
    assert event_f1([], []) == 1.
