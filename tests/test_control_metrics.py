import unittest

from beatfcn.train.control_metrics import (curve_metrics, curve_shuffle_gap,
    readout_metrics, type_metrics)


def event(onset_us, kind):
    return dict(onset_us=onset_us, type=kind)


class ControlMetricsTest(unittest.TestCase):
    def test_curve_correlation_and_shuffle(self):
        predicted = [event(0, 'tap'), event(1, 'tap'), event(500_000, 'tap')]
        requested = [1.0, 0.5, 0.0]
        shuffled = [0.0, 0.5, 1.0]
        real = curve_metrics(predicted, requested, 0)
        fake = curve_metrics(predicted, shuffled, 0)
        self.assertGreater(real['pearson'], fake['pearson'])
        gap = curve_shuffle_gap(predicted, requested, shuffled, 0)
        self.assertGreater(gap['pearson_gap'], 0)

    def test_type_recall_and_js(self):
        predicted = [event(0, 'tap'), event(1, 'drag'), event(2, 'drag')]
        truth = [event(0, 'tap'), event(1, 'drag'), event(2, 'flick')]
        metrics = type_metrics(predicted, truth)
        self.assertEqual(metrics['per_type']['tap']['count_ratio'], 1.0)
        self.assertEqual(metrics['per_type']['drag']['count_ratio'], 2.0)
        self.assertEqual(metrics['per_type']['flick']['count_ratio'], 0.0)
        self.assertGreater(metrics['js_divergence'], 0)
        self.assertLess(metrics['js_divergence'], 1)

    def test_readout_nonempty(self):
        metrics = readout_metrics([[event(0, 'tap')], []])
        self.assertEqual(metrics['songs'], 2)
        self.assertEqual(metrics['nonempty'], 1)
        self.assertEqual(metrics['nonempty_rate'], 0.5)


if __name__ == '__main__':
    unittest.main()
