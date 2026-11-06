import unittest

import torch

from beatfcn.model.full_event_model import event_loss
from beatfcn.train.sequence_likelihood import ModeSequenceNLL


class SequenceLikelihoodTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(203)
        self.logits = torch.randn(4, 7, 5, requires_grad=True)
        self.sequence = dict(allowed=torch.ones(4, 7, 5, dtype=torch.bool),
            valid=torch.arange(7)[None, :] < torch.tensor([1, 7, 3, 5])[:, None],
            targets=torch.randint(0, 5, (4, 7)))
        self.modes = torch.tensor([1, 1, 1, 0])
        self.loss = ModeSequenceNLL(mania_mean_tokens=5, phira_mean_tokens=11/3)

    def test_exact_request_scaling_and_gradients(self):
        direct = self.loss(self.logits, self.sequence, self.modes)
        expected = sum(event_loss(self.logits[i:i+1], {k:v[i:i+1] for k,v in self.sequence.items()})
            * self.sequence['valid'][i].sum() / self.loss.mean_tokens[self.modes[i]] for i in range(4))/4
        self.assertTrue(torch.allclose(direct, expected, atol=1e-7))
        g1 = torch.autograd.grad(direct, self.logits, retain_graph=True)[0]
        g2 = torch.autograd.grad(expected, self.logits)[0]
        self.assertTrue(torch.allclose(g1, g2, atol=1e-7))

    def test_microbatch_partition_does_not_change_objective(self):
        full = self.loss(self.logits, self.sequence, self.modes)
        micro = sum(self.loss(self.logits[i:i+1], {k:v[i:i+1] for k,v in self.sequence.items()}, self.modes[i:i+1]) for i in range(4))/4
        self.assertTrue(torch.allclose(full, micro, atol=1e-7))
        self.assertTrue(torch.allclose(torch.autograd.grad(full, self.logits, retain_graph=True)[0],
            torch.autograd.grad(micro, self.logits)[0], atol=1e-7))

    def test_padding_does_not_contribute(self):
        gradient = torch.autograd.grad(self.loss(self.logits, self.sequence, self.modes), self.logits)[0]
        self.assertEqual(float(gradient[~self.sequence['valid']].abs().sum()), 0)
        self.assertGreater(float(gradient[0, 0].abs().sum()), 0)

    def test_total_token_coefficient_mass_retains_mode_share(self):
        weights = self.sequence['valid'].sum(-1)/self.loss.mean_tokens[self.modes]
        self.assertAlmostEqual(float(weights[self.modes == 1].sum()), 3, places=6)
        self.assertAlmostEqual(float(weights[self.modes == 0].sum()), 1, places=6)
        self.assertNotEqual(float(weights[0]), float(weights[1]))

    def test_batch_token_mean_is_not_equivalent(self):
        numerator = sum(event_loss(self.logits[i:i+1], {k:v[i:i+1] for k,v in self.sequence.items()})
                        * self.sequence['valid'][i].sum() for i in range(4))
        token_mean = numerator / self.sequence['valid'].sum()
        self.assertGreater(float((token_mean-self.loss(self.logits, self.sequence, self.modes)).detach().abs()), 1e-4)

    def test_invalid_constants_rejected(self):
        for value in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                ModeSequenceNLL(mania_mean_tokens=value, phira_mean_tokens=3)

    def test_invalid_modes_rejected(self):
        for value in (-1, 2):
            modes = self.modes.clone()
            modes[0] = value
            with self.assertRaises((IndexError, RuntimeError)):
                self.loss(self.logits, self.sequence, modes)

    def test_forced_legal_choice_has_zero_gradient(self):
        self.sequence['allowed'][0, 0] = False
        self.sequence['allowed'][0, 0, self.sequence['targets'][0, 0]] = True
        gradient = torch.autograd.grad(self.loss(self.logits, self.sequence, self.modes), self.logits)[0]
        self.assertEqual(float(gradient[0, 0].abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
