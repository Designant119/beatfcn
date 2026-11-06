import unittest

from scripts.run_local_portfolio_data_scale import arm_passed, cite_saved_health, window_passed


def item(full, onset, count, target, complete=True):
    return dict(prediction=dict(complete=complete, events=[{}] * count),
        target=[{}] * target,
        scores=dict(full_note_f1=full, onset_10ms=dict(f1=onset)))


class DataScaleGate(unittest.TestCase):
    def test_saved_small_arm_rule(self):
        good = [item(.97, .97, 92, 87), item(.98, .98, 65, 63), item(0, None, 0, 0)]
        self.assertTrue(window_passed(good[-1]))
        self.assertTrue(arm_passed(good))
        weak = [item(.82, .68, 59, 63), item(0, None, 0, 0)]
        self.assertFalse(window_passed(weak[0]))
        self.assertFalse(arm_passed(weak))
        borderline = [item(.76, .81, 10, 10), item(.76, .81, 10, 10)]
        self.assertTrue(all(window_passed(row) for row in borderline))
        self.assertFalse(arm_passed(borderline))

    def test_follow_up_requires_capacity_failure_and_matching_small_arms(self):
        report = dict(id='local-portfolio-scale-v1', status='health_failed', health_updates=100,
            health_passed=False, updates=dict(small24=0, small64=0, capacity64=0),
            plan_sha256='parent', initial_weight_sha256=dict(small24='same', small64='same', capacity64='other'))
        predictions = dict(small24=[item(.97, .97, 10, 10)], small64=[item(.97, .97, 10, 10)],
            capacity64=[item(.64, .60, 8, 10)])
        citation = cite_saved_health(report, predictions, dict(small24='ckpt', small64='ckpt'))
        self.assertEqual(citation['initial_weight_sha256'], 'same')
        self.assertTrue(citation['capacity64_blocked_quality'])
        with self.assertRaises(ValueError):
            cite_saved_health(report, predictions, dict(small24='ckpt', small64='other'))
        passed_capacity = dict(predictions, capacity64=[item(.97, .97, 10, 10)])
        with self.assertRaises(ValueError):
            cite_saved_health(report, passed_capacity, dict(small24='ckpt', small64='ckpt'))


if __name__ == '__main__':
    unittest.main()
