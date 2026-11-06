import unittest

import torch

from beatfcn.model.full_event_model import event_loss
from beatfcn.train.condition_fork import condition_fork_loss


class ConditionForkTests(unittest.TestCase):
    def test_zero_weight_identity_and_only_verified_position_receives_extra_gradient(self):
        torch.manual_seed(41)
        logits = torch.randn(2, 4, 3, requires_grad=True)
        sequence = dict(allowed=torch.ones(2, 4, 3, dtype=torch.bool),
                        valid=torch.ones(2, 4, dtype=torch.bool), targets=torch.zeros(2, 4, dtype=torch.long))
        zero, _ = condition_fork_loss(logits, sequence, weight=0.)
        torch.testing.assert_close(zero, event_loss(logits, sequence), atol=0., rtol=0.)
        base = torch.autograd.grad(zero, logits)[0]
        active, _ = condition_fork_loss(logits, sequence, torch.tensor([1, -1]), weight=1.)
        extra = torch.autograd.grad(active, logits)[0] - base
        self.assertGreater(float(extra[0, 1].abs().max()), 0.)
        extra[0, 1] = 0
        self.assertEqual(float(extra.abs().max()), 0.)

    def test_absent_pair_is_zero_and_illegal_classes_remain_masked(self):
        logits = torch.tensor([[[1., 5., 1000.]]], requires_grad=True)
        sequence = dict(allowed=torch.tensor([[[True, True, False]]]),
                        valid=torch.tensor([[True]]), targets=torch.tensor([[0]]))
        absent, _ = condition_fork_loss(logits, sequence, torch.tensor([-1]), weight=1.)
        torch.testing.assert_close(absent, event_loss(logits, sequence), atol=0., rtol=0.)
        active, _ = condition_fork_loss(logits, sequence, torch.tensor([0]), weight=1.)
        active.backward()
        self.assertEqual(float(logits.grad[0, 0, 2]), 0.)
