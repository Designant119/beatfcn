import importlib.util
import json
from pathlib import Path
import unittest

import torch


spec = importlib.util.spec_from_file_location('initial_loss_app',
    Path(__file__).resolve().parents[1] / 'scripts/initial_loss_app.py')
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


class GradientComparisonTests(unittest.TestCase):
    def test_records_every_mismatch_without_losing_first_failure_evidence(self):
        left = {'band': torch.tensor([1., 2.]), 'later': torch.tensor([0.]), 'unused': None}
        right = {'band': torch.tensor([1., 2.01]), 'later': torch.tensor([.1]), 'unused': None}
        result = app.compare_gradients(left, right)
        self.assertFalse(result['passed'])
        self.assertEqual(result['failed_parameters'], ['band', 'later'])
        self.assertEqual(result['parameters']['band']['elements_outside_tolerance'], 1)
        self.assertGreater(result['parameters']['later']['maximum_absolute_difference'], .09)

    def test_nonfinite_and_missing_gradients_fail_even_when_both_nan(self):
        result = app.compare_gradients({'nan': torch.tensor([float('nan')]), 'missing': None},
                                       {'nan': torch.tensor([float('nan')]), 'missing': torch.ones(1)})
        self.assertFalse(result['passed'])
        self.assertEqual(result['failed_parameters'], ['nan', 'missing'])
        json.dumps(result, allow_nan=False)

    def test_rng_and_loss_are_gated_separately_from_gradient_equality(self):
        left = ({'loss': 1., 'rng_sha256': 'same'}, {'w': torch.ones(1)})
        self.assertTrue(app.compare_snapshots(left, left)['passed'])
        for summary in ({'loss': 1.01, 'rng_sha256': 'same'},
                        {'loss': 1., 'rng_sha256': 'changed'}):
            self.assertFalse(app.compare_snapshots(left, (summary, left[1]))['passed'])


if __name__ == '__main__':
    unittest.main()
