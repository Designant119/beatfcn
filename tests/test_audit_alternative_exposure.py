import unittest
from collections import Counter

import numpy as np

from beatfcn.train.alternative_context import normalize_alternative_context
from beatfcn.train.counterfactual_exposure import build_exposure
from beatfcn.train.exposure_loader import CounterfactualExposureDataset
from scripts.audit_alternative_exposure import audit, nearest_abs, overlap, summarize
from tests.test_alternative_context import SiblingPanel


class AuditAlternativeExposureTest(unittest.TestCase):
    def test_nearest_and_overlap(self):
        a = np.asarray([0, 100_000, 200_000], np.int64)
        self.assertEqual(nearest_abs(a, np.asarray([5_000, 300_000])).tolist(), [5_000, 95_000, 100_000])
        self.assertEqual(nearest_abs(a, np.asarray([150_000])).tolist(), [150_000, 50_000, 50_000])
        self.assertTrue(np.isinf(nearest_abs(a, np.asarray([], np.int64))).all())
        o = overlap(a, np.asarray([5_000, 300_000], np.int64))
        self.assertAlmostEqual(o['recall_10ms'], 1 / 3)
        self.assertAlmostEqual(o['recall_50ms'], 1 / 3)
        self.assertAlmostEqual(o['precision_10ms'], 1 / 2)
        self.assertEqual(overlap(np.asarray([], np.int64), a), {})

    def test_replay_matches_the_training_loader(self):
        settings = normalize_alternative_context(dict(p=.7, nearest=.6))
        panel = SiblingPanel()
        exposure = build_exposure(list(range(len(panel.records))), steps=80, seed=5)
        dataset = CounterfactualExposureDataset(panel, exposure, microbatch=4, alternative_context=settings)
        loader = Counter()
        for step in exposure['steps']:
            keys = step.get('occurrences', step['epochs'])
            for slot, index in enumerate(step['indices']):
                _, record = dataset.sample(index, step['rates'][slot], edit_seed=step['edit_seeds'][slot],
                                           curve_specified=step['curve_specified'][slot],
                                           specified_fields=step['specified_condition_fields'][slot],
                                           alternative_key=keys[slot])
                info = record['alternative']
                loader[(panel.records[index]['mode'], info['reason'] or 'applied')] += 1
        counts, shown = audit(panel, exposure, settings, step_indices=range(len(exposure['steps'])))
        replay = Counter({(mode, k): v for mode, c in counts.items() for k, v in c.items()
                          if k not in ('slots', 'native', 'edited_target')})
        replay[('mania_4k', 'edited_target_native_only')] = counts['mania_4k']['edited_target']
        replay[('phira', 'edited_target_native_only')] = counts['phira']['edited_target']
        self.assertEqual(+replay, +loader)
        self.assertGreater(counts['mania_4k']['applied'], 0)
        self.assertEqual(counts['phira']['applied'], 0)  # lone Phira chart: no sibling
        summary = summarize(counts, shown)
        self.assertGreater(summary['mania_4k']['shown']['aligned_recall_10ms'], 0)
        self.assertEqual(summary['mania_4k']['applied_share_of_slots'],
                         counts['mania_4k']['applied'] / counts['mania_4k']['slots'])


if __name__ == '__main__':
    unittest.main()
