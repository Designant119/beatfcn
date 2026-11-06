import unittest

import torch

from beatfcn.train.local_shapes import pad_input_shapes


class LocalShapesTests(unittest.TestCase):
    def test_pad_keeps_valid_values_and_rejects_truncation(self):
        inputs = dict(local=torch.arange(48).reshape(1, 2, 3, 2, 4),
                      local_times=torch.tensor([[.1, .2, .3]]),
                      local_valid=torch.tensor([[True, True, False]]))
        padded = pad_input_shapes(inputs, {'local': 5})
        self.assertTrue(torch.equal(padded['local'][:, :, :3], inputs['local']))
        self.assertEqual(padded['local_valid'].tolist(), [[True, True, False, False, False]])
        self.assertEqual(padded['local_times'][0, 3:].tolist(), [0., 0.])
        self.assertEqual(inputs['local'].shape[2], 3)
        with self.assertRaises(ValueError):
            pad_input_shapes(inputs, {'local': 2})
