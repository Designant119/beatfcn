import unittest

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.train.counterfactual_density import (edit_events_to_density,
    edit_events_to_multiplier, real_density_curve, sanitize_mania)


class CounterfactualDensityTest(unittest.TestCase):
    def test_silence_and_thickening(self):
        events = [RhythmEvent(0, 0, 'tap', -1), RhythmEvent(500_000, 500_000, 'tap', -1)]
        empty, actual = edit_events_to_density(events, 0, 1_000_000, 0., 'phira', 8_000_000)
        self.assertEqual(empty, [])
        self.assertEqual(actual, [0.0, 0.0])
        thick, actual = edit_events_to_density(events, 0, 1_000_000, 4., 'phira', 8_000_000)
        self.assertEqual(len(thick), 4)
        self.assertEqual(actual, [4.0, 4.0])

    def test_mania_added_taps_do_not_share_lane_and_onset(self):
        events = [RhythmEvent(0, 0, 'tap', 0)]
        thick, _ = edit_events_to_density(events, 0, 1_000_000, 8., 'mania_4k', 8_000_000)
        seen = set()
        for event in thick:
            self.assertIn(event.lane, range(4))
            key = (event.onset_us, event.lane)
            self.assertNotIn(key, seen)
            seen.add(key)

    def test_holds_are_removed_whole(self):
        events = [RhythmEvent(0, 300_000, 'hold', -1), RhythmEvent(100_000, 100_000, 'tap', -1)]
        thin, _ = edit_events_to_density(events, 0, 1_000_000, 0.5, 'phira', 8_000_000)
        self.assertTrue(all(event.type != 'hold' for event in thin))

    def test_multiplier_upsampling_keeps_human_types(self):
        events = [RhythmEvent(0, 0, 'tap', 0), RhythmEvent(150_000, 150_000, 'flick', 1),
                  RhythmEvent(250_000, 650_000, 'hold', 2)]
        thick, actual = edit_events_to_multiplier(events, 0, 500_000, 2., 'mania_4k',
                                                  8_000_000, seed=3)
        self.assertGreater(len(thick), len(events))
        self.assertEqual({event.type for event in events} - {event.type for event in thick}, set())
        self.assertTrue(any(event.type == 'hold' for event in thick))
        self.assertTrue(any(event.type == 'flick' for event in thick))
        self.assertGreater(actual[0], 0)

    def test_multiplier_leaves_empty_cells_empty(self):
        thick, _ = edit_events_to_multiplier([], 0, 500_000, 4., 'mania_4k', 8_000_000)
        self.assertEqual(thick, [])

    def test_multiplier_thinning_is_a_subset(self):
        events = [RhythmEvent(i*50_000, i*50_000, 'tap', i % 4) for i in range(8)]
        thin, _ = edit_events_to_multiplier(events, 0, 500_000, 0.5, 'mania_4k',
                                            8_000_000, seed=1)
        self.assertLess(len(thin), len(events))
        self.assertTrue({(event.onset_us, event.lane) for event in thin}
                        <= {(event.onset_us, event.lane) for event in events})

    def test_real_density_curve_reports_human_counts(self):
        events = [RhythmEvent(0, 0, 'tap', 0), RhythmEvent(100_000, 100_000, 'tap', 1)]
        curve = real_density_curve(events, 0, 1_000_000)
        self.assertEqual([round(cell['density'], 1) for cell in curve], [4.0, 0.0])

    def test_sanitizer_clips_hold_behind_a_later_note(self):
        events = [RhythmEvent(100_000, 400_000, 'hold', 1),
                  RhythmEvent(120_000, 120_000, 'tap', 1)]
        cleaned = sanitize_mania(events, 'mania_4k')
        hold = next(event for event in cleaned if event.type == 'hold')
        tap = next(event for event in cleaned if event.type == 'tap')
        self.assertLess(hold.end_us, tap.onset_us)

    def test_sanitizer_drops_two_notes_on_one_lane_and_onset(self):
        events = [RhythmEvent(0, 0, 'tap', 2), RhythmEvent(0, 0, 'flick', 2)]
        self.assertEqual(len(sanitize_mania(events, 'mania_4k')), 1)

    def test_relative_hold_copy_preserves_original_and_is_encodable(self):
        from beatfcn.model.full_event_codec import EventRequest, encode_events, teacher_sequence
        originals = [RhythmEvent(0, 0, 'tap', 1), RhythmEvent(200000, 300000, 'hold', 0),
                     RhythmEvent(450000, 450000, 'tap', 2)]
        thick, _ = edit_events_to_multiplier(originals, 0, 500000, 2.,
                                              'mania_4k', 1000000, seed=0)
        self.assertTrue(set(originals) <= set(thick))
        self.assertGreater(len(thick), len(originals))
        holds = [e for e in thick if e.type == 'hold']
        self.assertTrue(all(e.end_us-e.onset_us == 100000 for e in holds))
        request = EventRequest('mania_4k', 0, 500000, 1000000)
        teacher_sequence(encode_events(thick, request), request)

    def test_lone_head_is_not_invented_and_density_is_honest(self):
        original = RhythmEvent(200000, 400000, 'hold', 0)
        thick, actual = edit_events_to_multiplier([original], 0, 500000, 2.,
                                                   'mania_4k', 1000000, seed=0)
        self.assertEqual(thick, [original])
        self.assertAlmostEqual(actual[0], 2.0)

    def test_upsampling_subdivides_without_short_gaps(self):
        from beatfcn.train.counterfactual_density import MIN_GAP_US
        events = [RhythmEvent(i*200_000, i*200_000, 'drag', -1) for i in range(10)]
        for rate in (1.5, 2.0):
            thick, _ = edit_events_to_multiplier(events, 0, 2_000_000, rate, 'phira',
                                                  8_000_000, seed=4)
            onsets = sorted({e.onset_us for e in thick})
            self.assertGreater(len(thick), len(events))
            self.assertTrue(set(events) <= set(thick))
            self.assertGreaterEqual(min(b-a for a, b in zip(onsets, onsets[1:])), MIN_GAP_US)
            added = [e for e in thick if e not in events]
            self.assertTrue(all(e.onset_us % 50_000 == 0 and e.type == 'drag' for e in added))

    def test_relative_hold_copy_cannot_cover_future_cell_original(self):
        from beatfcn.model.full_event_codec import EventRequest, encode_events
        originals = [RhythmEvent(200000, 700000, 'hold', 0)]
        originals += [RhythmEvent(600000, 600000, 'tap', lane) for lane in (1, 2, 3)]
        thick, _ = edit_events_to_multiplier(originals, 0, 1000000, 2.,
                                              'mania_4k', 1000000, seed=0)
        self.assertTrue(set(originals) <= set(thick))
        encode_events(thick, EventRequest('mania_4k', 0, 1000000, 1000000))


if __name__ == '__main__':
    unittest.main()
