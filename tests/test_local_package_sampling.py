import unittest

import numpy as np
import torch

from beatfcn.model.local_event_package import EventPackageCanvas
from beatfcn.model.local_package_sampling import sample_package
from tests.test_local_mamba_canvas import small_config, synthetic_inputs


class PackageSamplingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(619)
        self.model = EventPackageCanvas(small_config(), density_reference=4).eval()
        self.inputs, self.request = synthetic_inputs()

    def output(self, kind, count):
        with torch.no_grad():
            output = self.model(self.inputs)
            output['count'].zero_()
            output['count'][..., 0] = 1
            output['count'][0, 0, kind].zero_()
            output['count'][0, 0, kind, count] = 1
        return output

    def test_sample_preserves_native_time_duplicate_multiset(self):
        with torch.no_grad():
            self.model.onset_read.weight.zero_()
            self.model.onset_read.bias.fill_(-1e4)
            self.model.onset_read.bias[5] = 1e4
        events = sample_package(self.model, self.inputs, self.request, seed=5, output=self.output(0, 3))
        self.assertEqual(len(events), 3)
        self.assertEqual([e.onset_us for e in events], [5_000] * 3)
        self.assertTrue(all(e.end_us == e.onset_us for e in events))

    def test_hold_is_sampled_from_legal_joint_domain_and_reproducible(self):
        output = self.output(1, 2)
        first = sample_package(self.model, self.inputs, self.request, seed=12, output=output)
        second = sample_package(self.model, self.inputs, self.request, seed=12, output=output)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertTrue(all(e.onset_us < e.end_us <= self.request.duration_us for e in first))
        self.assertTrue(all(e.end_us % 1000 == 0 for e in first))

    def test_invalid_permutation_cannot_move_events_into_padding(self):
        permutation = np.arange(200)
        permutation[[0, 100]] = permutation[[100, 0]]
        with self.assertRaisesRegex(ValueError, 'valid address support'):
            sample_package(self.model, self.inputs, self.request, seed=0,
                           output=self.output(0, 1), frame_permutation=permutation)


if __name__ == '__main__':
    unittest.main()
