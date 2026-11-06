"""The assistance proxy must reject flooding, silence, and incorrect hold tails."""
import unittest
from dataclasses import replace

from beatfcn.events.canonical import Event, LaneLoc
from beatfcn.model.edit_utility import edit_utility


def event(t, lane=0, duration=0):
    return Event(f'{t}:{lane}', t, t + duration, 'hold' if duration else 'tap', LaneLoc(lane))


class EditUtilityTests(unittest.TestCase):
    def test_perfect_empty_and_spam(self):
        truth = [event(100_000), event(300_000)]
        perfect = edit_utility(truth, truth, 'mania_4k')
        self.assertEqual(perfect['full_note']['saved_fraction'], 1)
        empty = edit_utility([], truth, 'mania_4k')
        self.assertEqual(empty['full_note']['saved_fraction'], 0)
        spam = edit_utility(truth + [event(t) for t in (500_000, 600_000, 700_000)], truth, 'mania_4k')
        self.assertLess(spam['full_note']['saved_fraction'], 0)

    def test_hold_head_is_not_a_finished_note(self):
        truth = [event(100_000, duration=500_000)]
        pred = [replace(truth[0], end_us=900_000)]
        result = edit_utility(pred, truth, 'mania_4k')
        self.assertEqual(result['typed_head']['matched'], 1)
        self.assertEqual(result['full_note']['matched'], 0)

    def test_unique_time_points_do_not_multiply_chords(self):
        truth = [event(100_000, lane) for lane in range(4)]
        result = edit_utility(truth[:1], truth, 'mania_4k')
        self.assertEqual(result['timing_point']['true'], 1)
        self.assertEqual(result['timing_point']['saved_fraction'], 1)
        self.assertEqual(result['full_note']['saved_fraction'], .25)

    def test_type_and_lane_corrections_are_distinguished(self):
        result = edit_utility([event(100_000, 1)], [event(100_000, 0)], 'mania_4k')
        self.assertEqual(result['rhythm_head']['matched'], 1)
        self.assertEqual(result['typed_head']['matched'], 0)

    def test_empty_truth_is_not_a_quality_pass(self):
        result = edit_utility([], [], 'mania_4k')
        self.assertIsNone(result['full_note']['saved_fraction'])
        self.assertEqual(result['full_note']['edit_cost'], 0)
        result = edit_utility([event(100_000)], [], 'mania_4k')
        self.assertEqual(result['full_note']['delete'], 1)
        self.assertIsNone(result['full_note']['saved_fraction'])

    def test_review_cost_and_delete_cost_sensitivity(self):
        truth = [event(100_000)]
        result = edit_utility(truth + [event(500_000)], truth, 'mania_4k', delete_cost=.5, review_cost=.1)
        self.assertAlmostEqual(result['full_note']['saved_fraction'], .3)

    def test_invalid_cost_and_tolerance_rejected(self):
        for kwargs in ({'delete_cost': -1}, {'review_cost': float('nan')}, {'tolerance_us': -1}):
            with self.assertRaises(ValueError):
                edit_utility([], [], 'mania_4k', **kwargs)


if __name__ == '__main__':
    unittest.main()
