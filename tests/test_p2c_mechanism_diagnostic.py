"""Counterfactual interventions must not change request metadata or padding."""
from dataclasses import dataclass
import unittest

import torch

from scripts.diagnose_p2c_mechanism import perturb_inputs


@dataclass
class Inputs:
    local: torch.Tensor
    song: torch.Tensor
    local_valid: torch.Tensor
    local_times: torch.Tensor
    conditions: torch.Tensor
    duration_ms: int


class InterventionTests(unittest.TestCase):
    def setUp(self):
        self.inputs = Inputs(torch.tensor([[0.], [1.], [2.], [3.], [0.]]),
            torch.tensor([[4.], [5.], [6.]]), torch.tensor([False, True, True, True, False]),
            torch.arange(5.), torch.tensor([[0., 1., 2.]]), 9000)

    def test_local_reverse_changes_only_valid_feature_order(self):
        x = perturb_inputs(self.inputs, 'reverse_local')
        self.assertEqual(x.local.flatten().tolist(), [0., 3., 2., 1., 0.])
        self.assertTrue(torch.equal(x.song, self.inputs.song))
        self.assertEqual(self.inputs.local.flatten().tolist(), [0., 1., 2., 3., 0.])
        for key in ('local_times', 'local_valid', 'conditions'):
            self.assertIs(getattr(x, key), getattr(self.inputs, key))
        self.assertEqual(x.duration_ms, self.inputs.duration_ms)

    def test_joint_reverse_preserves_feature_multisets(self):
        x = perturb_inputs(self.inputs, 'reverse_all')
        self.assertEqual(x.song.flatten().tolist(), [6., 5., 4.])
        self.assertTrue(torch.equal(x.local.sort(0).values, self.inputs.local.sort(0).values))
        self.assertTrue(torch.equal(x.song.sort(0).values, self.inputs.song.sort(0).values))
        self.assertTrue(torch.equal(perturb_inputs(x, 'reverse_all').local, self.inputs.local))

    def test_zero_audio_preserves_clock_and_conditions(self):
        x = perturb_inputs(self.inputs, 'zero_audio')
        self.assertEqual(int(x.local.count_nonzero() + x.song.count_nonzero()), 0)
        self.assertIs(x.local_times, self.inputs.local_times)
        self.assertIs(x.conditions, self.inputs.conditions)
        self.assertIs(x.local_valid, self.inputs.local_valid)

    def test_unknown_intervention_rejected(self):
        with self.assertRaises(ValueError):
            perturb_inputs(self.inputs, 'shuffle_labels')


if __name__ == '__main__':
    unittest.main()
