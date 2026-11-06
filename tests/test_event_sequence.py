"""R17 exact targets, causal state, and absence of occupancy repair."""
from dataclasses import replace
import unittest

import numpy as np

from beatfcn.model.event_canvas import ClockLayout, RhythmEvent, CanvasOverflow
from beatfcn.model.event_sequence import (
    END, NEW, Decision, SequenceSpec, EventSequenceState, encode_sequence,
    decode_sequence, teacher_views,
)


class EventSequenceTest(unittest.TestCase):
    def spec(self, mode='mania_4k', **kwargs):
        return SequenceSpec(ClockLayout(8_000_000, 16_000_000), mode, 32_000_000, **kwargs)

    def test_exact_long_tail_and_group_multiplicity(self):
        spec = self.spec('phira')
        truth = (RhythmEvent(8_001_000, 32_000_000, 'hold'),) + (
            RhythmEvent(8_001_000, 8_001_000, 'drag'),) * 9 + (
            RhythmEvent(9_999_000, 9_999_000, 'flick'),)
        tokens = encode_sequence(truth, spec)
        self.assertEqual(decode_sequence(tokens, spec), tuple(sorted(truth)))
        self.assertEqual(sum(d.field == 'member' and d.value == NEW for d in tokens), 11)

    def test_exact_authorized_submillisecond_copy(self):
        spec = self.spec('phira', copy_times=(8_000_333,))
        truth = (RhythmEvent(8_000_333, 8_001_000, 'hold'),)
        self.assertEqual(decode_sequence(encode_sequence(truth, spec), spec), truth)
        with self.assertRaisesRegex(ValueError, 'COPY_TIME'):
            encode_sequence(truth, self.spec('phira'))

    def test_known_carry_survives_end_core_and_releases_at_exact_end(self):
        hold = RhythmEvent(7_000_000, 10_000_000, 'hold', 2)
        spec = self.spec(allowed_types=('tap',))
        truth = (RhythmEvent(9_000_000, 9_000_000, 'tap', 1),
                 RhythmEvent(10_000_000, 10_000_000, 'tap', 1))
        state = EventSequenceState(spec, known_carry=(hold,))
        observed = {}
        for token in encode_sequence(truth, spec, known_carry=(hold,)):
            if token.field == 'kind':
                observed[state.onset_us] = state.model_view(history_enabled=True)['occupancy'][2, 0]
            state.apply(token)
        self.assertEqual(observed, {9_000_000: 1, 10_000_000: 0})
        self.assertEqual(state.known_carry, (hold,))
        self.assertEqual(decode_sequence(encode_sequence((), spec, known_carry=(hold,)), spec, known_carry=(hold,)), ())

    def test_conflicting_lane_remains_available_and_conflict_is_preserved(self):
        spec = self.spec()
        truth = (RhythmEvent(8_000_000, 18_000_000, 'hold', 0),
                 RhythmEvent(9_000_000, 9_000_000, 'tap', 0))
        state = EventSequenceState(spec)
        for token in encode_sequence(truth, spec):
            if token.field == 'lane' and state.onset_us == 9_000_000:
                self.assertEqual(state.allowed_values(), (0, 1, 2, 3))
                self.assertEqual(state.model_view(history_enabled=True)['occupancy'][0, 0], 1)
            state.apply(token)
        self.assertEqual(tuple(sorted(state.events)), truth)

    def test_completed_attributes_absent_in_A_but_present_in_B(self):
        spec = self.spec()
        variants = [(RhythmEvent(8_000_000, tail, 'hold', lane),
                     RhythmEvent(9_000_000, 9_000_000, 'tap', 2))
                    for tail, lane in ((15_000_000, 0), (20_000_000, 3))]
        states = []
        for events in variants:
            state = EventSequenceState(spec)
            for token in encode_sequence(events, spec):
                if token.field == 'group' and len(state.events) == 1:
                    states.append(state)
                    break
                state.apply(token)
        a, b = [s.model_view(history_enabled=False) for s in states]
        for key in ('syntax', 'occupancy'):
            np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(a['completed_events'], ())
        self.assertEqual(a['active_hold_total'], 0)
        self.assertEqual(a['allowed_values'], b['allowed_values'])
        self.assertNotEqual(states[0].model_view(history_enabled=True)['completed_events'],
                            states[1].model_view(history_enabled=True)['completed_events'])

    def test_teacher_views_do_not_contain_the_current_or_future_member(self):
        spec = self.spec()
        truth = (RhythmEvent(8_005_000, 8_005_000, 'tap', 1),
                 RhythmEvent(12_000_000, 20_000_000, 'hold', 3))
        views = list(teacher_views(encode_sequence(truth, spec), spec, history_enabled=True))
        first_lane = next(view for view, token in views if token.field == 'lane')
        self.assertEqual(first_lane['completed_events'], ())
        self.assertEqual(first_lane['active_hold_total'], 0)
        changed = (truth[0], replace(truth[1], end_us=22_000_000, lane=0))
        other = list(teacher_views(encode_sequence(changed, spec), spec, history_enabled=True))
        for (a, ta), (b, tb) in zip(views, other):
            if ta != tb:
                np.testing.assert_array_equal(a['syntax'], b['syntax'])
                self.assertEqual(a['completed_events'], b['completed_events'])
                break
            np.testing.assert_array_equal(a['syntax'], b['syntax'])
            self.assertEqual(a['completed_events'], b['completed_events'])

    def test_empty_and_early_end_are_real_predictions_not_target_truncation(self):
        spec = self.spec()
        self.assertEqual(decode_sequence((Decision('group', END),), spec), ())
        state = EventSequenceState(spec)
        self.assertEqual(state.allowed_values(), (END, NEW))
        state.apply(Decision('group', NEW))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            decode_sequence((Decision('group', NEW),), spec)
        with self.assertRaisesRegex(ValueError, 'after END_CORE'):
            decode_sequence((Decision('group', END), Decision('group', END)), spec)

    def test_bounds_and_permissions_fail_without_truncation_or_tail_clipping(self):
        truth = (RhythmEvent(8_000_000, 32_001_000, 'hold', 0),)
        with self.assertRaisesRegex(ValueError, 'full time domain'):
            encode_sequence(truth, self.spec())
        truth = (RhythmEvent(8_001_000, 8_001_000, 'tap', 0),)
        with self.assertRaisesRegex(ValueError, 'permission'):
            encode_sequence(truth, self.spec(permitted_heads_us=(8_000_000,)))
        with self.assertRaises(CanvasOverflow):
            encode_sequence(truth, self.spec(max_decisions=2))
        with self.assertRaises(CanvasOverflow):
            encode_sequence(truth * 2, self.spec(max_events=1))

    def test_translation_preserves_full_intervals_and_types(self):
        truth = tuple(RhythmEvent(t, 13_608_000, 'hold', lane)
                      for lane, t in enumerate((13_308_000, 13_408_000, 13_508_000)))
        shifted = tuple(replace(e, onset_us=e.onset_us-2_000_000, end_us=e.end_us-2_000_000) for e in truth)
        self.assertEqual(decode_sequence(encode_sequence(shifted, self.spec()), self.spec()), shifted)

    def test_hold_only_head_cannot_enter_a_tail_domain_dead_end(self):
        spec = SequenceSpec(ClockLayout(16_000_000, 16_000_500), 'phira', 16_000_500,
                            copy_times=(16_000_100,), allowed_types=('hold',))
        self.assertEqual(EventSequenceState(spec).allowed_values(), (END,))
        with self.assertRaisesRegex(ValueError, 'permission'):
            EventSequenceState(spec).apply(Decision('group', NEW))
        valid = replace(spec, layout=ClockLayout(15_999_000, 16_000_500))
        truth = (RhythmEvent(15_999_000, 16_000_000, 'hold'),)
        self.assertEqual(decode_sequence(encode_sequence(truth, valid), valid), truth)


if __name__ == '__main__':
    unittest.main()
