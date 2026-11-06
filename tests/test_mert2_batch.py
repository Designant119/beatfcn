from types import SimpleNamespace
import unittest

import numpy as np

from beatfcn.audio.mert2 import FrozenMert2, Mert2Spec
from beatfcn.audio.mert2_batch import encode_windows_batch, feature_parity


class FakeModule:
    def __call__(self, input_values, output_hidden_states):
        base = input_values.reshape(input_values.shape[0], -1, 960).float().mean(dim=-1)
        base = base[:, :, None].expand(-1, -1, 1024)
        return SimpleNamespace(hidden_states=tuple(base * (index + 1) for index in range(24)))


def fake_model():
    model = FrozenMert2.__new__(FrozenMert2)
    model.spec = Mert2Spec()
    model.device = 'cpu'
    model.revision = 'test'
    model.model = FakeModule()
    return model


class Mert2BatchTests(unittest.TestCase):
    def test_equal_length_batch_and_long_window_stitch_match_single(self):
        model = fake_model()
        waves = [np.linspace(-1, 1, 8_640_001, dtype=np.float32),
                 np.linspace(1, -1, 8_640_001, dtype=np.float32),
                 np.linspace(-.5, .5, 1000, dtype=np.float32)]
        batched = encode_windows_batch(model, waves, window_batch=4)
        for index, wave in enumerate(waves):
            single = model.encode(wave)
            parity = feature_parity(single['layers'], batched[index]['layers'])
            self.assertEqual(parity['max_abs'], 0.)
            self.assertEqual(parity['float16_exact_fraction'], 1.)
            self.assertTrue(np.array_equal(single['times_s'], batched[index]['times_s']))
            self.assertTrue(np.array_equal(single['intervals_s'], batched[index]['intervals_s']))
            self.assertEqual(single['metadata']['windows'], batched[index]['metadata']['windows'])


if __name__ == '__main__':
    unittest.main()
