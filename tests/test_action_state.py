import unittest
from dataclasses import asdict

from beatfcn.model.action_state import (
    State, ManiaAction, PhiraAction, apply_action, encode, replay,
    mania_code, mania_digits, legal_mania_actions,
    KEEP, TAP, START, RELEASE, RELEASE_TAP, RELEASE_START,
)
from beatfcn.model.event_canvas import RhythmEvent as E


class ActionStateTests(unittest.TestCase):
    def test_exact_microseconds_and_simultaneous_release_restart(self):
        events = (E(13, 1001, 'hold', 0), E(1001, 2003, 'hold', 0),
                  E(1001, 1001, 'tap', 3), E(2003, 2003, 'tap', 0))
        actions = encode(events, 'mania_4k')
        self.assertEqual(actions[1], ManiaAction(1001, mania_code((RELEASE_START, KEEP, KEEP, TAP))))
        self.assertEqual(mania_digits(actions[2].code), (RELEASE_TAP, KEEP, KEEP, KEEP))
        self.assertEqual(replay(actions, 'mania_4k').events, tuple(sorted(events)))

    def test_same_start_phira_holds_need_no_future_duration_id(self):
        events = (E(7, 103, 'hold'), E(7, 107, 'hold'), E(7, 107, 'hold'),
                  E(103, 103, 'drag'), E(103, 103, 'drag'), E(104, 104, 'flick'))
        actions = encode(events, 'phira')
        step = apply_action(State('phira'), actions[0])
        self.assertEqual(step.state.hold_groups, ((7, 3),))
        self.assertEqual(actions[1].releases, ((7, 1),))
        self.assertEqual(actions[-1].releases, ((7, 2),))
        self.assertEqual(replay(actions, 'phira').events, tuple(sorted(events)))

    def test_active_state_is_identical_when_only_unobserved_future_changes(self):
        for mode, lane in [('mania_4k', 1), ('phira', -1)]:
            a = encode([E(13, 1001, 'hold', lane)], mode)
            b = encode([E(13, 2003, 'hold', lane), E(9000, 9000, 'tap', lane)], mode)
            left = apply_action(State(mode), a[0]).state
            right = apply_action(State(mode), b[0]).state
            self.assertEqual(left, right)
            self.assertNotIn('end_us', asdict(left))
            if mode == 'mania_4k':
                self.assertEqual(legal_mania_actions(left), legal_mania_actions(right))

    def test_window_carry_preserves_unfinished_holds_without_clipping(self):
        events = (E(0, 9001, 'hold', 0), E(100, 100, 'tap', 1), E(9001, 10000, 'hold', 0))
        actions = encode(events, 'mania_4k')
        prefix = replay(actions[:2], 'mania_4k', require_closed=False)
        self.assertEqual(prefix.events, (E(100, 100, 'tap', 1),))
        self.assertEqual(prefix.state.lane_starts, (0, None, None, None))
        suffix = replay(actions[2:], 'mania_4k', initial=prefix.state)
        self.assertEqual(tuple(sorted(prefix.events + suffix.events)), tuple(sorted(events)))
        with self.assertRaisesRegex(ValueError, 'unreleased'):
            replay(actions[:2], 'mania_4k')

    def test_release_new_start_at_same_time_is_atomic_for_phira(self):
        events = [E(0, 100, 'hold'), E(100, 101, 'hold')]
        actions = encode(events, 'phira')
        old = apply_action(State('phira'), actions[0]).state
        step = apply_action(old, actions[1])
        self.assertEqual(step.state.hold_groups, ((100, 1),))
        self.assertEqual(step.events, (E(0, 100, 'hold'),))

    def test_joint_support_depends_only_on_current_key_occupancy(self):
        empty = State('mania_4k')
        self.assertEqual(len(legal_mania_actions(empty)), 3 ** 4 - 1)
        full = State('mania_4k', 10, (0, 1, 2, 3))
        self.assertEqual(len(legal_mania_actions(full)), 4 ** 4 - 1)
        self.assertNotIn(mania_code((TAP, KEEP, KEEP, KEEP)), legal_mania_actions(full))
        self.assertIn(mania_code((RELEASE_TAP, KEEP, KEEP, KEEP)), legal_mania_actions(full))

    def test_illegal_source_is_rejected_not_repaired(self):
        bad = [[E(0, 100, 'hold', 0), E(99, 99, 'tap', 0)],
               [E(1, 1, 'tap', 0), E(1, 1, 'tap', 0)], [E(1, 1, 'hold', 0)]]
        for events in bad:
            with self.subTest(events=events), self.assertRaises(ValueError):
                encode(events, 'mania_4k')

    def test_invalid_release_rejects_entire_action_without_mutating_state(self):
        state = State('phira', 10, hold_groups=((1, 1),))
        before = asdict(state)
        for action in [PhiraAction(11, (5, 0, 0, 0), ((1, 2),)),
                       PhiraAction(11, (0, 0, 0, 0), ((1, 1), (2, 1))),
                       PhiraAction(11, (0, 0, 0, 0), ((1, 1), (1, 1)))]:
            with self.assertRaises(ValueError):
                apply_action(state, action)
            self.assertEqual(asdict(state), before)

    def test_empty_repeat_and_noninteger_actions_are_not_valid_boundaries(self):
        state = State('mania_4k', 10)
        for action in [ManiaAction(11, 0), ManiaAction(10, 1), ManiaAction(9, 1),
                       ManiaAction(10.5, 1), ManiaAction(11, True), ManiaAction(11, 1296)]:
            with self.assertRaises(ValueError):
                apply_action(state, action)
        with self.assertRaises(ValueError):
            apply_action(State('phira'), PhiraAction(0, (0, 0, 0, 0)))

    def test_empty_chart_is_distinct_from_unfinished_chart(self):
        self.assertEqual(encode([], 'phira'), ())
        self.assertEqual(replay([], 'phira').events, ())
        with self.assertRaises(ValueError):
            State('phira', None, hold_groups=((0, 1),))


if __name__ == '__main__':
    unittest.main()
