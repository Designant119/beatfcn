import copy
import unittest

from beatfcn.train.comparison import compare_compute_benchmarks


def measurement(seconds):
    return {'rows_sha256': 'corrected-data', 'sample_sequence': [[{'chart_id': 'a', 'seed': 7}]],
            'config': {'dropout': 0.}, 'effective_batch': 32, 'steps': 100,
            'seconds_per_step': seconds}


class ComputeComparisonTests(unittest.TestCase):
    def test_faster_but_dearer_is_tradeoff_not_dominance(self):
        report = compare_compute_benchmarks({'L40S': measurement(.45), 'H100': measurement(.30)},
                                            {'L40S': .001, 'H100': .002})
        self.assertEqual(set(report['resource_frontier']), {'L40S', 'H100'})
        self.assertEqual(report['provisional_gpu'], 'H100')
        self.assertFalse(report['quality_evaluated'])
        self.assertFalse(report['pareto_improvement_established'])

    def test_faster_and_cheaper_dominates_only_resource_axes(self):
        report = compare_compute_benchmarks({'L40S': measurement(.8), 'H100': measurement(.3)},
                                            {'L40S': .001, 'H100': .002})
        self.assertEqual(report['resource_frontier'], ['H100'])
        self.assertFalse(report['pareto_improvement_established'])

    def test_mismatched_samples_or_training_configuration_cannot_be_compared(self):
        original = measurement(.3)
        changes = {'rows_sha256': 'old-data', 'sample_sequence': [[{'chart_id': 'easy-only'}]],
                   'config': {'dropout': .1}, 'effective_batch': 64, 'steps': 50}
        for key, value in changes.items():
            with self.subTest(key=key):
                candidate = copy.deepcopy(original)
                candidate[key] = value
                with self.assertRaisesRegex(ValueError, key):
                    compare_compute_benchmarks({'a': original, 'b': candidate}, {'a': .001, 'b': .001})

    def test_missing_provenance_or_invalid_timing_fails_closed(self):
        for key, value in [('sample_sequence', [[]]), ('rows_sha256', ''),
                           ('seconds_per_step', float('nan')), ('seconds_per_step', 0),
                           ('effective_batch', 0)]:
            with self.subTest(key=key, value=value):
                candidate = {**measurement(.3), key: value}
                with self.assertRaises(ValueError):
                    compare_compute_benchmarks({'a': candidate}, {'a': .001})


if __name__ == '__main__':
    unittest.main()
