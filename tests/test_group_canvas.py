import unittest
from collections import Counter
from dataclasses import replace

import numpy as np

from beatfcn.model.event_canvas import ClockLayout, RhythmEvent, CanvasOverflow, EMPTY, MASK
from beatfcn.model.group_canvas import (
    new_group_canvas, encode_group_targets, decode_group_targets, masked_group_view,
)


class GroupCanvasTest(unittest.TestCase):
    def test_groups_share_head_but_keep_distinct_notes_and_long_tails(self):
        events = [RhythmEvent(199_000, 199_000, 'drag')] * 3 + [
            RhythmEvent(200_000, 9_999_000, 'hold'), RhythmEvent(200_000, 200_000, 'flick'),
            RhythmEvent(400_000, 400_000, 'tap')]
        for phase in (0, 200_000):
            encoded = encode_group_targets(events, ClockLayout(0, 800_000, phase_us=phase), 'phira')
            self.assertEqual(Counter(decode_group_targets(encoded)), Counter(events))
            self.assertEqual(int(encoded.counts.sum()), 3)
            self.assertEqual(int(encoded.multiplicity.sum()), 6)

    def test_mania_preserves_four_keys_with_two_hold_endpoints(self):
        events = [RhythmEvent(100_000, end, kind, lane) for lane, kind, end in
                  ((0, 'hold', 9_000_000), (1, 'tap', 100_000), (2, 'hold', 10_000_000), (3, 'tap', 100_000))]
        labels = encode_group_targets(events, ClockLayout(0, 800_000), 'mania_4k')
        self.assertEqual(Counter(decode_group_targets(labels)), Counter(events))
        self.assertEqual(labels.counts.tolist(), [1, 0])
        self.assertEqual(labels.multiplicity[0, 0], 4)

    def test_blank_and_partial_views_do_not_leak_hidden_counts(self):
        layout = ClockLayout(0, 800_000)
        blanks = []
        for count in (0, 1, 12):
            labels = encode_group_targets([RhythmEvent(i * 10_000, i * 10_000, 'tap')
                                           for i in range(count)], layout, 'phira')
            view = masked_group_view(labels, np.zeros_like(labels.multiplicity, dtype=bool))
            blanks.append(view)
            self.assertTrue((view.counts == -1).all())
            self.assertTrue((view.multiplicity == -1).all())
            self.assertTrue((view.notes[..., 0] == MASK).all())
            with self.assertRaises(ValueError):
                decode_group_targets(view)
        for view in blanks[1:]:
            for key in ('counts', 'heads', 'multiplicity', 'notes'):
                np.testing.assert_array_equal(getattr(view, key), getattr(blanks[0], key))
        visible = np.zeros_like(labels.multiplicity, dtype=bool)
        visible[0, 0] = True
        partial = masked_group_view(labels, visible)
        self.assertTrue((partial.counts == -1).all())
        self.assertEqual(partial.multiplicity[0, 0], 1)

    def test_exact_copy_is_shared_without_rounding(self):
        events = [RhythmEvent(100_123, 100_123, 'drag')] * 2
        layout = ClockLayout(0, 400_000)
        labels = encode_group_targets(events, layout, 'phira', copy_times=(100_123,))
        self.assertEqual(Counter(decode_group_targets(labels)), Counter(events))
        with self.assertRaises(ValueError):
            encode_group_targets(events, layout, 'phira')

    def test_both_capacity_overflows_are_explicit(self):
        layout = ClockLayout(0, 400_000)
        with self.assertRaises(CanvasOverflow):
            encode_group_targets([RhythmEvent(i * 1000, i * 1000, 'drag') for i in range(33)], layout, 'phira')
        with self.assertRaises(CanvasOverflow):
            encode_group_targets([RhythmEvent(0, 0, 'drag')] * 9, layout, 'phira')

    def test_malformed_count_and_hidden_payload_cannot_be_finalized(self):
        empty = new_group_canvas(ClockLayout(0, 400_000), 'phira', unknown=False)
        bad = replace(empty, counts=np.array([33], np.int64))
        with self.assertRaises(ValueError):
            decode_group_targets(bad)
        empty.notes[0, 0, 0, 0] = 0
        with self.assertRaises(ValueError):
            decode_group_targets(empty)


if __name__ == '__main__':
    unittest.main()
