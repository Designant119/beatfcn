import unittest
from types import SimpleNamespace

from beatfcn.events.canonical import Event
from beatfcn.model.group_canvas import decode_group_targets
from scripts.matched_acoustic_data import target_events, common_physical_targets, make_example
from scripts.prepare_matched_acoustic_panel import select_covered_pairs


class MatchedClockTests(unittest.TestCase):
    def setUp(self):
        self.plan = dict(core_us=8_000_000, core_origin_in_clip_us=8_000_000,
                         clip_us=32_000_000, untrained_translation_us=2_000_000)

    def event(self, head, tail, kind='hold'):
        return Event.from_json(dict(event_id=f'{head}:{tail}', onset_us=head, end_us=tail,
            type=kind, location=dict(kind='lane', lane=0), native_ref='fixture', fake=False))

    def test_hold_crossing_requested_end_keeps_complete_tail(self):
        chart = SimpleNamespace(mode='mania_4k', events=[self.event(15_000_000, 19_000_000)])
        normal, _ = target_events(chart, 8_000_000, self.plan)
        shifted, _ = target_events(chart, 10_000_000, self.plan)
        self.assertEqual(decode_group_targets(normal)[0].end_us, 19_000_000)
        self.assertEqual(decode_group_targets(shifted)[0].end_us, 17_000_000)
        identical, interval = common_physical_targets(normal, shifted,
            dict(source_core_us=8_000_000, source_clip_start_us=0),
            dict(source_core_us=10_000_000, source_clip_start_us=2_000_000), self.plan)
        self.assertTrue(identical)
        self.assertEqual(interval, [10_000_000, 16_000_000])

    def test_incoming_hold_rejected_without_revealing_context(self):
        chart = SimpleNamespace(mode='mania_4k', events=[self.event(7_000_000, 9_000_000)])
        with self.assertRaisesRegex(ValueError, 'incoming_hold'):
            target_events(chart, 8_000_000, self.plan)

    def test_tail_outside_supported_clip_is_not_truncated(self):
        chart = SimpleNamespace(mode='mania_4k', events=[self.event(15_000_000, 33_000_000)])
        with self.assertRaisesRegex(ValueError, 'outside'):
            target_events(chart, 8_000_000, self.plan)

    def test_sub_summary_shift_cannot_fake_aligned_cache(self):
        with self.assertRaisesRegex(ValueError, 'feature clocks not aligned'):
            make_example(dict(source_core_us=8_000_000),
                         dict(self.plan, untrained_translation_us=400_000), translated=True)

    def test_coverage_does_not_substitute_duplicate_song_for_support(self):
        def pair(audio, types):
            return [dict(row=dict(mode='mania_4k', audio_sha256=audio), types=types, source_core_us=t)
                    for t in (8_000_000, 32_000_000)]
        candidates = [pair('a', {'tap': 10}), pair('a', {'hold': 3}), pair('b', {'hold': 1})]
        picked = select_covered_pairs(candidates, ('mania_4k',))
        self.assertEqual([x['row']['audio_sha256'] for x in picked], ['a', 'a', 'b', 'b'])
        self.assertEqual(picked[2]['types'], {'hold': 1})
        with self.assertRaisesRegex(ValueError, 'no distinct-song'):
            select_covered_pairs(candidates[:2], ('mania_4k',))


if __name__ == '__main__':
    unittest.main()
