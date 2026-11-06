from copy import deepcopy
import unittest
from scripts.condition_cohort_probe import assess_probe
from scripts.resume_condition_light import assess_existing_pipeline


class CohortContinuationTests(unittest.TestCase):
    def fixture(self):
        return {'complete': True, 'phases': [
            {'panel': panel, 'cohort': cohort, 'wall_s': wall, 'microbatches': 128,
             'finite_gradients': True, 'input_provenance_sha256': panel, 'losses': [2.] * 256}
            for panel, cohort, wall in [('A', False, 100.), ('A', True, 70.), ('B', True, 100.), ('B', False, 110.)]]}

    def test_total_time_in_both_orders_must_improve_before_paid_resume(self):
        report = self.fixture()
        self.assertAlmostEqual(assess_probe(report)['speedup'], 210 / 170)
        report['phases'][2]['wall_s'] = 140.
        with self.assertRaisesRegex(ValueError, 'total pipeline time'):
            assess_probe(report)

    def test_missing_or_numerically_changed_data_cannot_open_the_gate(self):
        original = self.fixture()
        for change in ('coverage', 'input', 'loss', 'nan'):
            report = deepcopy(original)
            if change == 'coverage':
                report['phases'].pop()
            elif change == 'input':
                report['phases'][1]['input_provenance_sha256'] = 'different'
            elif change == 'loss':
                report['phases'][1]['losses'][0] += .01
            else:
                report['phases'][1]['losses'][0] = float('nan')
            with self.subTest(change=change), self.assertRaises(ValueError):
                assess_probe(report)

    def test_original_pipeline_needs_both_finite_utilization_windows(self):
        report = self.fixture()
        for row in report['phases']:
            row['gpu'] = {'util_mean': 80.}
        self.assertEqual(len(assess_existing_pipeline(report)['diagnostic_windows']), 2)
        report['phases'][0]['gpu']['util_mean'] = 69.
        with self.assertRaisesRegex(ValueError, 'short finite/utilization'):
            assess_existing_pipeline(report)
        report['phases'][0]['gpu']['util_mean'] = None
        with self.assertRaisesRegex(ValueError, 'short finite/utilization'):
            assess_existing_pipeline(report)
