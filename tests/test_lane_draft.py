import unittest

import torch

from beatfcn.model.lane_draft import (LanePrefix, assign_lanes, make_lane_draft,
    maximum_overlap, ordered_lane_labels, prefix_inputs, rhythm_signature)


def note(head, lane, tail=None):
    return dict(onset_us=head, end_us=head if tail is None else tail,
                type='tap' if tail is None else 'hold', lane=lane)


class LaneDraftTests(unittest.TestCase):
    def test_lane_values_and_source_order_do_not_enter_draft(self):
        events = [note(8_100_000, 3), note(8_100_000, 1), note(8_400_000, 0, 9_100_000)]
        first = make_lane_draft(events)
        other = make_lane_draft([dict(e, lane=3 - e['lane']) for e in reversed(events)])
        self.assertEqual(first.events, other.events)
        self.assertTrue(torch.equal(first.features, other.features))
        self.assertEqual(first.features.shape, (3, 64))
        self.assertEqual(rhythm_signature(assign_lanes(first, [0, 2, 3])), rhythm_signature(events))

    def test_future_and_current_labels_do_not_enter_prefix(self):
        events = [note(8_000_000, 0, 8_900_000), note(8_200_000, 1), note(8_400_000, 2)]
        draft, labels = make_lane_draft(events), ordered_lane_labels(events)
        before = prefix_inputs(draft, labels, conditioned=True)
        changed = labels.clone(); changed[1:] = (changed[1:] + 1) % 4
        after = prefix_inputs(draft, changed, conditioned=True)
        for a, b in zip(before, after):
            self.assertTrue(torch.equal(a[:2], b[:2]))
        for a, b in zip(prefix_inputs(draft, labels, conditioned=False),
                        prefix_inputs(draft, changed, conditioned=False)):
            self.assertTrue(torch.equal(a, b))

    def test_hold_endpoint_release_and_same_onset_exclusion(self):
        state = LanePrefix()
        state.advance((8_000_000, 9_000_000, 'hold'), 0)
        self.assertFalse(state.allowed(8_999_999)[0])
        self.assertTrue(state.allowed(9_000_000)[0])
        state.advance((9_000_000, 9_000_000, 'tap'), 0)
        self.assertFalse(state.allowed(9_000_000)[0])
        self.assertTrue(state.allowed(9_000_001)[0])

    def test_overfull_draft_is_detected_without_removing_members(self):
        events = [note(8_000_000, i, 9_000_000) for i in range(4)] + [note(8_100_000, 0)]
        draft = make_lane_draft(events)
        self.assertEqual(maximum_overlap(draft), 5)
        self.assertEqual(len(draft.events), 5)
        with self.assertRaises(ValueError):
            assign_lanes(draft, [0, 1, 2, 3])


if __name__ == '__main__':
    unittest.main()
