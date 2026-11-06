import unittest
from collections import Counter

import numpy as np

from beatfcn.model.event_canvas import (
    EMPTY, MASK, KIND, HEAD_MS, COPY_INDEX, TAIL_MS, CanvasOverflow,
    ClockLayout, RhythmEvent, active_holds, decode_targets, encode_targets,
    masked_view, new_canvas,
)


class EventCanvasTest(unittest.TestCase):
    def test_both_phases_preserve_multiplicity_boundaries_and_long_tails(self):
        events = [RhythmEvent(0, 0, 'tap'), RhythmEvent(399000, 1200000, 'hold'),
                  RhythmEvent(400000, 400000, 'drag'), RhythmEvent(400000, 400000, 'drag'),
                  RhythmEvent(401000, 401000, 'flick'), RhythmEvent(799000, 799000, 'tap')]
        for phase in (0, 200000):
            layout = ClockLayout(0, 800000, phase_us=phase)
            canvas = encode_targets(events, layout, 'phira')
            self.assertEqual(Counter(events), Counter(decode_targets(canvas)))
            self.assertEqual(active_holds(decode_targets(canvas), 800000), (events[1],))
        self.assertEqual(len(ClockLayout(0, 800000, phase_us=200000).blocks), 3)

    def test_mania_chord_keeps_lanes_and_tail_beyond_interval(self):
        events = [RhythmEvent(20000, 20000, 'tap', 0), RhythmEvent(20000, 900000, 'hold', 3)]
        canvas = encode_targets(events, ClockLayout(0, 400000), 'mania_4k')
        self.assertEqual(Counter(events), Counter(decode_targets(canvas)))

    def test_exact_anchor_time_is_not_rounded(self):
        events = [RhythmEvent(200123, 650000, 'hold')]
        layout = ClockLayout(0, 400000)
        canvas = encode_targets(events, layout, 'phira', copy_times=(200123,))
        self.assertEqual(canvas.values[0, 0, COPY_INDEX], 0)
        self.assertEqual(decode_targets(canvas), tuple(events))
        with self.assertRaisesRegex(ValueError, 'authorized copy'):
            encode_targets(events, layout, 'phira')

    def test_blank_request_does_not_leak_number_or_presence_of_targets(self):
        layout = ClockLayout(0, 800000)
        targets = [[RhythmEvent(1000, 1000, 'tap')], [],
                   [RhythmEvent(i * 1000, i * 1000, 'drag') for i in range(20)]]
        blank = new_canvas(layout, 'phira')
        for events in targets:
            labels = encode_targets(events, layout, 'phira')
            full_mask = masked_view(labels, np.zeros(labels.values.shape[:2], dtype=bool))
            np.testing.assert_array_equal(blank.values, full_mask.values)
        self.assertTrue(np.all(blank.values[..., KIND] == MASK))
        self.assertTrue(np.all(encode_targets([], layout, 'phira').values[..., KIND] == EMPTY))
        with self.assertRaisesRegex(ValueError, 'unknown slot'):
            decode_targets(blank)

    def test_overflow_and_out_of_interval_are_rejected_without_truncation(self):
        layout = ClockLayout(0, 400000)
        events = [RhythmEvent(i * 1000, i * 1000, 'drag') for i in range(33)]
        with self.assertRaises(CanvasOverflow):
            encode_targets(events, layout, 'phira')
        with self.assertRaisesRegex(ValueError, 'outside'):
            encode_targets([RhythmEvent(400000, 400000, 'tap')], layout, 'phira')

    def test_bad_predictions_fail_explicitly(self):
        layout = ClockLayout(0, 800000)
        canvas = encode_targets([RhythmEvent(10000, 900000, 'hold')], layout, 'phira')
        for field, value, reason in ((HEAD_MS, 400, 'owning block'), (TAIL_MS, 1, 'follow'),
                                     (COPY_INDEX, 0, 'exactly one'), (COPY_INDEX, -2, 'sentinel')):
            bad = encode_targets([RhythmEvent(10000, 900000, 'hold')], layout, 'phira')
            bad.values[0, 0, field] = value
            with self.assertRaisesRegex(ValueError, reason):
                decode_targets(bad)
        canvas.values[0, 1, HEAD_MS] = 20
        with self.assertRaisesRegex(ValueError, 'empty slot'):
            decode_targets(canvas)


if __name__ == '__main__':
    unittest.main()
