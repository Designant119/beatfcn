import unittest

import torch

from beatfcn.train.condition_interaction import condition_interaction_loss


class ConditionInteractionTests(unittest.TestCase):
    def test_row_and_common_token_bias_invariance_including_gradients(self):
        a = torch.tensor([[.2, -.3], [.1, .5]], dtype=torch.float64, requires_grad=True)
        row = torch.tensor([[11.], [-7.]], dtype=torch.float64, requires_grad=True)
        column = torch.tensor([[4., -3.]], dtype=torch.float64, requires_grad=True)
        base, gap = condition_interaction_loss(a)
        other, other_gap = condition_interaction_loss(a + row + column)
        torch.testing.assert_close(base, other)
        torch.testing.assert_close(gap, other_gap)
        ga, gr, gc = torch.autograd.grad(other, (a, row, column), retain_graph=True)
        torch.testing.assert_close(ga, torch.autograd.grad(base, a)[0])
        self.assertEqual(float(gr.abs().max()), 0.)
        self.assertEqual(float(gc.abs().max()), 0.)

    def test_identical_responses_still_supply_opposite_condition_gradients(self):
        scores = torch.tensor([[2., 1.], [2., 1.]], requires_grad=True)
        loss, gap = condition_interaction_loss(scores)
        loss.backward()
        self.assertEqual(float(gap), 0.)
        self.assertLess(float(scores.grad[0, 0]), 0.)
        self.assertGreater(float(scores.grad[1, 0]), 0.)
        torch.testing.assert_close(scores.grad[0], -scores.grad[1])

    def test_high_gap_does_not_mean_absolute_decision_is_correct(self):
        loss, gap = condition_interaction_loss(torch.tensor([[10., 0.], [1., 0.]]))
        self.assertEqual(float(gap), 9.)
        self.assertLess(float(loss), .001)
        self.assertEqual(torch.tensor([1., 0.]).argmax().item(), 0)  # B is still wrong.

    def test_pair_equal_weighting_and_log_probability_equivalence(self):
        x = torch.tensor([[[4., 2.], [2., 5.]], [[2., 4.], [3., 2.]]], dtype=torch.float64)
        loss, gap = condition_interaction_loss(x)
        torch.testing.assert_close(loss, torch.stack([condition_interaction_loss(z)[0] for z in x]).mean())
        ploss, pgap = condition_interaction_loss(x.log_softmax(-1))
        torch.testing.assert_close(loss, ploss)
        torch.testing.assert_close(gap, pgap)

    def test_reject_wrong_geometry_empty_and_integral_scores(self):
        for x in (torch.zeros(2), torch.zeros(2, 3), torch.zeros(0, 2, 2), torch.zeros(2, 2, dtype=torch.long)):
            with self.assertRaises(ValueError):
                condition_interaction_loss(x)

    def test_both_condition_and_target_swap_preserves_objective(self):
        x = torch.tensor([[1., 2.], [4., 8.]], dtype=torch.float64)
        loss, gap = condition_interaction_loss(x)
        other, other_gap = condition_interaction_loss(x.flip((0, 1)))
        torch.testing.assert_close(loss, other)
        torch.testing.assert_close(gap, other_gap)

    def test_zero_weight_and_streamed_pair_backward_match_combined_objective(self):
        x = torch.arange(20, dtype=torch.float64).reshape(4, 5).div(10).requires_grad_(True)
        targets = torch.tensor([0, 1, 2, 3])
        ce = torch.nn.functional.cross_entropy(x, targets)
        scores = torch.stack([x[:2, :2], x[2:, 2:4]])
        aux, _ = condition_interaction_loss(scores)
        base_grad = torch.autograd.grad(ce, x, retain_graph=True)[0]
        zero_grad = torch.autograd.grad(ce + 0 * aux, x, retain_graph=True)[0]
        torch.testing.assert_close(zero_grad, base_grad, atol=0, rtol=0)
        combined = torch.autograd.grad(ce + aux, x)[0]
        y = x.detach().clone().requires_grad_(True)
        torch.nn.functional.cross_entropy(y, targets).backward()
        for z in (y[:2, :2], y[2:, 2:4]):
            (condition_interaction_loss(z)[0] / 2).backward()
        torch.testing.assert_close(y.grad, combined, atol=1e-15, rtol=1e-14)
