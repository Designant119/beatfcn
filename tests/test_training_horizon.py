import unittest

from scripts.acoustic_probe_metrics import event_metrics
from scripts.probe_training_horizon import compare, distribution_metrics


class TrainingHorizonTests(unittest.TestCase):
    def test_tail_diagnostic_includes_hold_that_full_event_matching_rejects(self):
        truth = [dict(onset_us=12_513_333, end_us=13_346_667, type='hold', lane=-1)]
        predicted = [dict(truth[0], end_us=12_930_000)]
        self.assertEqual(event_metrics(predicted, truth, 'phira')['matched'], 0)
        result = distribution_metrics(predicted, truth)
        self.assertEqual(result['hold_head_matched_tail_errors_ms'], [-416.667])
        self.assertEqual(result['timing_point_f1'], 1.)

    def test_near_ceiling_improvement_is_attainable_but_missing_type_still_fails(self):
        def summary(score):
            result = {mode: dict(full_note_f1=score, per_type={'hold': dict(true=9, pred=9)})
                      for mode in ['mania_4k', 'phira']}
            result['all'] = dict(full_note_f1=score, timing_point_f1=score, cores=5, complete_legal=5)
            return result
        base, final = summary(.99), summary(1.)
        self.assertTrue(all(compare(base, final).values()))
        final['phira']['per_type']['hold']['pred'] = 0
        self.assertFalse(compare(base, final)['no_applicable_type_disappeared'])
        self.assertFalse(compare(summary(1.), summary(1.))['gain_has_headroom'])


if __name__ == '__main__':
    unittest.main()
