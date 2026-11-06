import unittest

import torch

from scripts.run_noisy_proposal_refiner_v2 import zero_proposal_loss


class EmptyProposalTrainingTests(unittest.TestCase):
    def test_zero_request_keeps_batch_weight_and_frozen_parameters(self):
        model = torch.nn.Linear(2, 1)
        model.bias.requires_grad_(False)
        positive = model(torch.ones(1, 2)).sum()
        loss = (positive + zero_proposal_loss(model)) / 2
        loss.backward()
        self.assertTrue(torch.equal(model.weight.grad, torch.full_like(model.weight, .5)))
        self.assertIsNone(model.bias.grad)

    def test_all_empty_batch_has_finite_zero_gradients(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
        loss = (zero_proposal_loss(model) + zero_proposal_loss(model)) / 2
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        for parameter in model.parameters():
            self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter)))


if __name__ == '__main__':
    unittest.main()
