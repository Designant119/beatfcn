import unittest
import json
import numpy as np
from scripts.audio_grid_geometry import grid_points, residuals, error_summary, drag_burst_indices
from scripts.audit_audio_grid import measured, quantization_damage, screen_passed


class AudioGridGeometryTests(unittest.TestCase):
    def test_local_phase_resets_at_tempo_boundary_and_extends_before_first_anchor(self):
        # The first beat anchor is in the middle of the file. The second tempo
        # begins off the first grid; its own anchor must nevertheless be exact.
        grid = grid_points([[125, 500], [1600, 300]], 2200000, [1, 2])
        np.testing.assert_array_equal(grid, [125000, 375000, 625000, 875000, 1125000,
                                           1375000, 1600000, 1750000, 1900000, 2050000, 2200000])
        early = grid_points([[1125, 500]], 1300000, [1])
        np.testing.assert_array_equal(early, [125000, 625000, 1125000])

    def test_phase_control_preserves_period_but_breaks_known_alignment(self):
        times = np.array([500000, 1000000, 1500000])
        good = residuals(times, grid_points([[0, 500]], 2000000, [1]))[0]
        shifted = residuals(times, grid_points([[0, 500]], 2000000, [1], phase_shift_us=137000))[0]
        self.assertEqual(error_summary(good)['within_20ms'], 1)
        self.assertEqual(error_summary(shifted)['within_20ms'], 0)
        np.testing.assert_array_equal(shifted, [-137000] * 3)

    def test_residual_retains_off_grid_time_and_empty_is_not_success(self):
        times = np.array([13007, 498331, 1000067])
        error, closest = residuals(times, grid_points([[0, 500]], 1500000, [1, 2, 3, 4]))
        np.testing.assert_array_equal(error + closest, times)
        self.assertIsNone(error_summary([])['within_20ms'])
        with self.assertRaises(ValueError):
            grid_points([[10, 500], [5, 300]], 1000000, [1])

    def test_drag_burst_requires_distinct_times_and_preserves_all_objects(self):
        times = [0] * 5 + [1000000, 1050000, 1100000, 1150000, 1150000, 1500000]
        events = [dict(type='drag', onset_us=t) for t in times]
        np.testing.assert_array_equal(drag_burst_indices(events), [5, 6, 7, 8, 9])

    def test_dense_grid_covers_wrong_phase_too(self):
        times = np.array([13000, 58000, 84000])
        grid = grid_points([[0, 160]], 200000, [8])
        shifted = grid_points([[0, 160]], 200000, [8], phase_shift_us=7000)
        score, _, _ = measured(times, grid, [shifted])
        self.assertEqual(score['within_20ms'], 1)
        self.assertEqual(score['phase_excess_20ms'], 0)

    def test_snap_damage_counts_distinct_groups_and_destroyed_holds(self):
        heads = np.array([100000, 100000, 110000])
        ends = np.array([110000, 100000, 110000])
        damage = quantization_damage(heads, ends, np.array([True, False, False]),
                                     np.array([0, 100000, 200000]))
        self.assertEqual(damage['distinct_onsets'], 2)
        self.assertEqual(damage['collapsed_distinct_onsets'], 1)
        self.assertEqual(damage['nonpositive_holds'], 1)

    def test_coordinate_screen_is_json_serializable_and_empty_does_not_pass(self):
        gate = dict(minimum_coverage_20ms=.85, minimum_phase_excess_20ms=.20)
        score = dict(within_20ms=.9, phase_excess_20ms=np.float64(.3))
        self.assertEqual(json.loads(json.dumps(dict(passed=screen_passed(score, gate)))), {'passed': True})
        self.assertFalse(screen_passed(dict(within_20ms=None, phase_excess_20ms=None), gate))


if __name__ == '__main__':
    unittest.main()
