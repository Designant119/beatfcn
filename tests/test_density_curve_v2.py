import unittest
from types import SimpleNamespace

import numpy as np

from beatfcn.train.intensity_panel import IntensityPanel


class Panel:
    def __init__(self, count, split='train', end=500_000):
        self.records = [dict(split=split, request=dict(start_us=0, end_us=end))]
        self.count = count

    def targets(self, index):
        return [SimpleNamespace(onset_us=0) for _ in range(self.count)]


class DensityV2Tests(unittest.TestCase):
    def test_high_density_is_not_saturated(self):
        low = IntensityPanel(Panel(8)).curve_for(0)
        high = IntensityPanel(Panel(16)).curve_for(0)
        self.assertGreater(high['intensity'][0, 0], low['intensity'][0, 0])

    def test_unspecified_is_distinct_from_explicit_zero(self):
        absent = IntensityPanel(Panel(8), hide_probability=1).curve_for(0)
        zero = IntensityPanel(Panel(0)).curve_for(0)
        self.assertFalse(absent['intensity_valid'].any())
        self.assertTrue(zero['intensity_valid'].all())

    def test_validation_does_not_derive_hidden_curve(self):
        class Forbidden(Panel):
            def targets(self, index):
                raise AssertionError('Hidden validation target read')
        curve = IntensityPanel(Forbidden(8, split='val')).curve_for(0)
        self.assertFalse(curve['intensity_valid'].any())

    def test_partial_cell_has_true_center_and_duration(self):
        curve = IntensityPanel(Panel(1, end=100_000)).curve_for(0)
        self.assertAlmostEqual(float(curve['intensity_times'][0]), .05)
        np.testing.assert_allclose(curve['intensity'][0, 1:], [0, .1])
        expected = np.log1p(10)/np.log1p(16)
        self.assertAlmostEqual(float(curve['intensity'][0, 0]), expected, places=6)

    def test_training_and_user_segments_have_identical_representation(self):
        from beatfcn.train.density_curve import requested_density_curve
        training = IntensityPanel(Panel(8)).curve_for(0)
        requested = requested_density_curve([dict(start_us=0, end_us=500_000, density=16)],
            0, 500_000)
        for key in training:
            np.testing.assert_array_equal(training[key], requested[key])


if __name__ == '__main__':
    unittest.main()
