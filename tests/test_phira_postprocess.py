import unittest

from beatfcn.events.phira_layout import density_layout
from beatfcn.events.phira_postprocess import clean_chords, clean_types, limit_fingers, limit_holds, merge_fragments, playable


def tap(t):
    return dict(onset_us=t, end_us=t, type='tap')


class FingerLimitTest(unittest.TestCase):
    def fingers(self, events, t):
        return sum(e['onset_us'] <= t <= e['end_us'] for e in events)

    def test_chord_is_cut_to_two(self):
        out, report = limit_fingers([tap(0), tap(0), tap(0), tap(0)])
        self.assertEqual(len(out), 2)
        self.assertEqual(report['dropped'], 2)

    def test_old_hold_ends_before_a_third_finger(self):
        h1, h2 = dict(onset_us=0, end_us=2_000_000, type='hold'), dict(onset_us=500_000, end_us=2_000_000, type='hold')
        out, report = limit_fingers([h1, h2, tap(1_000_000)])
        self.assertEqual(report['shortened'], 1)
        self.assertLessEqual(self.fingers(out, 1_000_000), 2)

    def test_short_hold_becomes_tap(self):
        h1, h2 = dict(onset_us=0, end_us=900_000, type='hold'), dict(onset_us=10_000, end_us=900_000, type='hold')
        out, report = limit_fingers([h1, h2, tap(60_000)])
        self.assertEqual(report['to_tap'], 1)

    def test_within_limit_is_untouched(self):
        events = [tap(0), tap(0), tap(100_000)]
        out, report = limit_fingers(events)
        self.assertEqual(len(out), 3)
        self.assertEqual(sum(report.values()), 0)


class CleanTypesTest(unittest.TestCase):
    def test_lone_flick_in_fast_run_becomes_tap(self):
        out, report = clean_types([tap(0), dict(onset_us=100_000, end_us=100_000, type='flick'), tap(200_000)])
        self.assertEqual([e['type'] for e in out], ['tap']*3)
        self.assertEqual(report['types_cleaned'], 1)

    def test_slow_accent_is_kept(self):
        out, report = clean_types([tap(0), dict(onset_us=500_000, end_us=500_000, type='flick'), tap(1_000_000)])
        self.assertEqual(out[1]['type'], 'flick')
        self.assertEqual(report['types_cleaned'], 0)

    def test_chord_flick_is_kept(self):
        out, _ = clean_types([tap(0), dict(onset_us=100_000, end_us=100_000, type='flick'), tap(100_000), tap(200_000)])
        self.assertIn('flick', [e['type'] for e in out])


class DensityLayoutTest(unittest.TestCase):
    def test_every_event_is_placed_once_in_bounds(self):
        events = [tap(t) for t in range(0, 8_000_000, 100_000)]+[dict(onset_us=500_000, end_us=900_000, type='hold')]
        out = density_layout(events)
        self.assertEqual(len(out), len(events))
        self.assertTrue(all(-675 <= p['x'] <= 675 for p in out))

    def test_fast_runs_stay_narrower_than_slow_sections(self):
        fast = density_layout([tap(t) for t in range(0, 5_000_000, 100_000)])
        slow = density_layout([tap(t) for t in range(0, 50_000_000, 1_000_000)])
        self.assertLess(max(abs(p['x']) for p in fast), max(abs(p['x']) for p in slow))


class ProfileRulesTest(unittest.TestCase):
    def test_close_onsets_merge(self):
        out, r = merge_fragments([tap(0), tap(70_000), tap(200_000)], 90_000)
        self.assertEqual(len(out), 2)
        self.assertEqual(r['fragments_dropped'], 1)

    def test_merge_keeps_the_stronger_note(self):
        out, _ = merge_fragments([tap(0), dict(onset_us=70_000, end_us=500_000, type='hold')], 90_000)
        self.assertEqual([e['type'] for e in out], ['hold'])

    def test_only_one_hold_down(self):
        a, b = dict(onset_us=0, end_us=2_000_000, type='hold'), dict(onset_us=1_000_000, end_us=2_000_000, type='hold')
        out, r = limit_holds([a, b], 1, 150_000)
        self.assertEqual(r['holds_shortened'], 1)
        self.assertLess(out[0]['end_us'], out[1]['onset_us'])

    def test_hold_right_after_a_hold_becomes_tap(self):
        a, b = dict(onset_us=0, end_us=90_000, type='hold'), dict(onset_us=100_000, end_us=500_000, type='hold')
        out, r = limit_holds([a, b], 1, 150_000)
        self.assertEqual(out[1]['type'], 'tap')

    def test_chord_keeps_one_special_note(self):
        out, r = clean_chords([dict(onset_us=0, end_us=100_000, type='hold'), dict(onset_us=0, end_us=0, type='flick'), tap(0)])
        self.assertEqual(sorted(e['type'] for e in out), ['hold', 'tap', 'tap'])
        self.assertEqual(r['chords_cleaned'], 1)

    def test_profile_runs_and_expert_keeps_more(self):
        events = [tap(t) for t in range(0, 2_000_000, 50_000)]
        relaxed, _ = playable(events, 'relaxed')
        expert, _ = playable(events, 'expert')
        self.assertLess(len(relaxed), len(expert))


if __name__ == '__main__':
    unittest.main()
