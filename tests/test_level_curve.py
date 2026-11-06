import unittest

import numpy as np

from beatfcn.train.counterfactual_exposure import build_exposure
from beatfcn.train.exposure_loader import CounterfactualExposureDataset, normalize_level_curve
from tests.test_alternative_context import SiblingPanel


def slots(exposure):
    for step in exposure['steps']:
        keys = step.get('occurrences', step['epochs'])
        for slot, index in enumerate(step['indices']):
            yield step, slot, index, keys[slot]


class LevelCurveTest(unittest.TestCase):
    def sample(self, dataset, step, slot, index):
        return dataset.sample(index, step['rates'][slot], edit_seed=step['edit_seeds'][slot],
                              curve_specified=step['curve_specified'][slot],
                              specified_fields=step['specified_condition_fields'][slot])

    def test_settings(self):
        self.assertIsNone(normalize_level_curve(None))
        self.assertEqual(normalize_level_curve({'p': .35}), {'p': .35})
        with self.assertRaises(ValueError):
            normalize_level_curve({'p': 2})
        with self.assertRaises(ValueError):
            normalize_level_curve({'q': .1})

    def test_off_and_p0_identical_and_on_gives_constant_song_level(self):
        panel = SiblingPanel()
        exposure = build_exposure(list(range(len(panel.records))), steps=60, seed=4)
        off = CounterfactualExposureDataset(panel, exposure, microbatch=4)
        zero = CounterfactualExposureDataset(panel, exposure, microbatch=4, level_curve={'p': 0.})
        on = CounterfactualExposureDataset(panel, exposure, microbatch=4, level_curve={'p': 1.})
        applied = 0
        for step, slot, index, _ in slots(exposure):
            a, _ = self.sample(off, step, slot, index)
            b, _ = self.sample(zero, step, slot, index)
            for k in a['inputs']:
                np.testing.assert_array_equal(a['inputs'][k], b['inputs'][k])
            c, audit = self.sample(on, step, slot, index)
            if step['curve_specified'][slot]:
                applied += 1
                self.assertTrue(audit.get('level_curve'))
                rows = np.asarray(c['inputs']['intensity'])[np.asarray(c['inputs']['intensity_valid'], bool)]
                self.assertGreater(len(rows), 0)
                self.assertTrue(np.allclose(rows[:, 0], rows[0, 0]))   # one constant density level (cols 1-2 = cell times)
                np.testing.assert_array_equal(c['inputs']['controls'], a['inputs']['controls'])
                np.testing.assert_array_equal(c['sequence']['tokens'] if 'tokens' in c['sequence'] else 0,
                                              a['sequence']['tokens'] if 'tokens' in a['sequence'] else 0)
            else:
                self.assertNotIn('level_curve', audit)
        self.assertGreater(applied, 0)


if __name__ == '__main__':
    unittest.main()
