import unittest

import numpy as np

from scripts.audit_phigros_clock import calibration, chart_clock


PLAN = dict(seed=20261025, min_confidence=3, max_cross_chart_span_ms=20,
    minimum_independent_training_audio=1, maximum_source_abs_deviation_p90_ms=20,
    maximum_half_difference_ms=10, minimum_audio_per_tier=1,
    maximum_tier_difference_ms=10, max_thirds_span_ms=20,
    max_whole_thirds_difference_ms=10, max_view_difference_ms=20)


def record(chart, audio, delta, split='train'):
    return dict(chart_id=chart, audio_sha256=audio, split=split, status='ok', tier='low',
        calibration_eligible=True, whole={'tap_hold': {'delta_ms': delta}})


class ClockCalibrationTest(unittest.TestCase):
    def test_split_is_checked_even_when_a_cached_flag_is_wrong(self):
        result = calibration([record('v', 'validation-song', 100, 'val'),
                              record('t', 'training-song', 0)], PLAN)
        self.assertEqual([r['audio_sha256'] for r in result['qualified_songs']], ['training-song'])

    def test_each_song_has_equal_weight_despite_many_chart_versions(self):
        rows = [record(str(i), 'one-song', 20) for i in range(100)]
        rows.extend([record('b', 'second-song', 0), record('c', 'third-song', 0)])
        result = calibration(rows, PLAN)
        self.assertEqual(result['raw_estimator_source_constant_ms'], 0)
        self.assertEqual(len(result['qualified_songs']), 3)

    def test_artistic_drag_and_weak_audio_are_retained_without_calibration(self):
        events = [dict(event_id=str(i), onset_us=1_000_000+i*20_000,
                       end_us=1_000_000+i*20_000, type='drag') for i in range(100)]
        row = dict(chart_id='gesture', audio_sha256='song', name='gesture',
            source_chart_id='native', split='train', difficulty_tier='low',
            structural_audit_pass=True)
        result = chart_clock(row, {'events': events}, np.zeros(1000), 240000, PLAN)
        self.assertEqual(result['notes'], 100)
        self.assertEqual(result['types'], {'drag': 100})
        self.assertEqual(result['decoded_bounds_violations'], [])
        self.assertFalse(result['calibration_eligible'])
        self.assertEqual(len(events), 100)


if __name__ == '__main__':
    unittest.main()
