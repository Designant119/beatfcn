from itertools import permutations
import unittest

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.mania_actions import (ACTION_COUNT, UNKNOWN_ACTION, ManiaAction,
                                        action_states, action_size, encode_action, decode_action)


class ManiaActionTest(unittest.TestCase):
    def test_all_codes_and_member_permutations(self):
        for code in range(ACTION_COUNT):
            states = action_states(code)
            action = ManiaAction(code, tuple(30_000_001 + lane if state == 2 else None
                                             for lane, state in enumerate(states)))
            events = decode_action(action, onset_us=123_457)
            self.assertEqual(len(events), action_size(code))
            for ordered in permutations(events):
                self.assertEqual(encode_action(ordered, onset_us=123_457), action)
                self.assertEqual(decode_action(encode_action(ordered, onset_us=123_457), onset_us=123_457), events)

    def test_binding_is_not_a_member_permutation(self):
        a = [RhythmEvent(5, 20_000_007, 'hold', 0), RhythmEvent(5, 5, 'tap', 1)]
        b = [RhythmEvent(5, 5, 'tap', 0), RhythmEvent(5, 20_000_007, 'hold', 1)]
        self.assertNotEqual(encode_action(a, onset_us=5), encode_action(b, onset_us=5))
        self.assertEqual(encode_action(a[::-1], onset_us=5), encode_action(a, onset_us=5))

    def test_malformed_actions_fail_without_repair(self):
        with self.assertRaises(ValueError):
            action_states(UNKNOWN_ACTION)
        self.assertEqual(decode_action(ManiaAction(0, (None,) * 4), onset_us=1), ())
        with self.assertRaises(ValueError):
            encode_action([RhythmEvent(1, 1, 'tap', 0)] * 2, onset_us=1)
        with self.assertRaises(ValueError):
            encode_action([RhythmEvent(1, 1, 'drag', -1)], onset_us=1)
        with self.assertRaises(ValueError):
            encode_action([RhythmEvent(1, 1, 'hold', 0)], onset_us=1)
        with self.assertRaises(ValueError):
            ManiaAction(1, (10, None, None, None))
        with self.assertRaises(ValueError):
            decode_action(ManiaAction(2, (10, None, None, None)), onset_us=10)

    def test_time_shift_preserves_binding_and_complete_tail(self):
        events = [RhythmEvent(-123, 20_000_007, 'hold', 3), RhythmEvent(-123, -123, 'tap', 0)]
        action = encode_action(events, onset_us=-123)
        shifted = [RhythmEvent(e.onset_us + 7, e.end_us + 7, e.type, e.lane) for e in events]
        moved = encode_action(shifted, onset_us=-116)
        self.assertEqual(action.code, moved.code)
        self.assertEqual(moved.tails_us[3], action.tails_us[3] + 7)
        self.assertEqual(decode_action(moved, onset_us=-116), tuple(sorted(shifted)))

    def test_later_zero_action_does_not_release_prior_hold(self):
        prior = encode_action([RhythmEvent(0, 20_000_000, 'hold', 0)], onset_us=0)
        later = encode_action([RhythmEvent(10_000_000, 10_000_000, 'tap', 1)], onset_us=10_000_000)
        self.assertEqual(action_states(later.code)[0], 0)
        self.assertEqual(decode_action(prior, onset_us=0)[0].end_us, 20_000_000)

    def test_physical_lane_swap_changes_code_not_tail_identity(self):
        original = encode_action([RhythmEvent(10, 1001, 'hold', 0)], onset_us=10)
        swapped = encode_action([RhythmEvent(10, 1001, 'hold', 2)], onset_us=10)
        self.assertNotEqual(original.code, swapped.code)
        self.assertEqual(original.tails_us[0], swapped.tails_us[2])


if __name__ == '__main__':
    unittest.main()
