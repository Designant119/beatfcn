import unittest
import numpy as np

from beatfcn.train.local_features import feature_window


class LocalFeatureWindowTests(unittest.TestCase):
    def data(self, samples):
        cells = samples//960
        layers = [np.zeros((cells, 8, 1024), np.int8) for _ in range(3)]
        scales = {f'scale_{l}': np.ones(1024, np.float32) for l in (9, 16, 24)}
        mel = np.zeros(((samples+959)//960*4, 128), np.float16)
        summary = dict(values=np.zeros((1, 1024), np.float32), times=np.array([.25], np.float32))
        return layers, scales, mel, summary

    def test_partial_tail_uses_real_mel_interval_and_no_fabricated_pupu(self):
        samples = 240000+250
        r = feature_window(*self.data(samples), samples, 8_000_000, samples*1_000_000//24000)
        self.assertEqual(r['local'].shape[1], 150)
        self.assertEqual(len(r['mel']), 604)
        self.assertEqual(r['mel_valid'][-4:].tolist(), [True, True, False, False])
        self.assertAlmostEqual(float(r['mel_times'][-3]), (240240+samples)/48000-8, places=6)
        self.assertEqual(r['local_valid'].sum(), 150)

    def test_exact_cell_tail_and_outside_scope(self):
        samples = 240000
        r = feature_window(*self.data(samples), samples, 8_000_000, 10_000_000)
        self.assertTrue(r['mel_valid'].all())
        self.assertEqual(len(r['mel']), r['local'].shape[1]*4)
        with self.assertRaisesRegex(ValueError, 'outside'):
            feature_window(*self.data(samples), samples, 8_000_000, 10_000_001)

    def test_full_pupu_tail_or_unpadded_mel_is_not_silently_accepted(self):
        samples = 240250
        layers, scales, mel, summary = self.data(samples)
        with self.assertRaisesRegex(ValueError, 'Pupu'):
            feature_window([np.pad(a, ((0, 1), (0, 0), (0, 0))) for a in layers],
                           scales, mel, summary, samples, 0, 8_000_000)
        with self.assertRaisesRegex(ValueError, 'Complete mel'):
            feature_window(layers, scales, mel[:-4], summary, samples, 0, 8_000_000)
