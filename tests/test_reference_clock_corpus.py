import json
import unittest

import numpy as np

from scripts.audit_reference_clock_corpus import tensor_health


class TensorHealthTests(unittest.TestCase):
    def test_large_finite_fp16_does_not_overflow_reduction(self):
        mel = np.full((2000, 128), -4, dtype=np.float16)
        mel[1000:] = 4
        result = tensor_health(mel)
        self.assertEqual(result['mean'], 0.)
        self.assertEqual(result['std'], 4.)
        json.dumps(result, allow_nan=False)

    def test_invalid_data_is_rejected(self):
        for data in (np.empty((0, 128)), np.zeros((10, 127)), np.full((10, 128), np.nan),
                     np.full((10, 128), np.inf)):
            with self.subTest(shape=data.shape), self.assertRaises(ValueError):
                tensor_health(data)
