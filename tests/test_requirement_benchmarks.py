import unittest

from beatfcn.train.requirement_benchmarks import requirement_metrics, summarize


def event(onset, kind='tap', lane=-1):
    return dict(onset_us=onset, end_us=onset, type=kind, lane=lane)


class RequirementBenchmarkTest(unittest.TestCase):
    def test_free_and_curve_metrics(self):
        predicted = [event(0), event(500_000), event(1_000_000)]
        truth = [event(0), event(500_000), event(1_000_000)]
        metrics = requirement_metrics(predicted, truth, 'phira', 0, 2_000_000,
            requested_curve=[dict(start_us=0, end_us=2_000_000, density=2.0)],
            task='curve')
        self.assertEqual(metrics['task'], 'curve')
        self.assertEqual(metrics['note_f1_50ms'], 1.0)
        self.assertGreater(metrics['curve']['mae_heads_per_s'], 0)
        summary = summarize([metrics])
        self.assertEqual(summary['tasks'], 1)
        self.assertEqual(summary['mean_note_f1_50ms'], 1.0)

    def test_continuation_task_label(self):
        metrics = requirement_metrics([event(2_000_000, lane=0)], [event(2_000_000, lane=0)],
            'mania_4k', 2_000_000, 3_000_000, task='continuation')
        self.assertEqual(metrics['task'], 'continuation')
        self.assertEqual(metrics['onset_f1_50ms'], 1.0)


if __name__ == '__main__':
    unittest.main()
