import unittest

import torch

from beatfcn.model.full_event_model import event_loss
from beatfcn.train.sequence_likelihood import ModeSequenceNLL
from scripts.run_local_sequence_quality import training_objective


class SequenceQualityTest(unittest.TestCase):
    def test_explicit_control_matches_original_loss_and_gradient(self):
        torch.manual_seed(17)
        logits = torch.randn(1, 9, 4, requires_grad=True)
        sequence = dict(allowed=torch.ones_like(logits, dtype=torch.bool),
            valid=torch.tensor([[True]*5+[False]*4]), targets=torch.randint(0, 4, (1, 9)))
        objective = ModeSequenceNLL(mania_mean_tokens=268.106, phira_mean_tokens=249.457)
        control = training_objective('request_mean', objective, logits, sequence, torch.tensor([1]))
        old = event_loss(logits, sequence)
        self.assertEqual(float(control.detach()), float(old.detach()))
        self.assertTrue(torch.equal(torch.autograd.grad(control, logits, retain_graph=True)[0],
                                   torch.autograd.grad(old, logits, retain_graph=True)[0]))
        candidate = training_objective('fixed_mode_sequence', objective, logits, sequence, torch.tensor([1]))
        expected = old*5/249.457
        self.assertTrue(torch.allclose(candidate, expected, atol=1e-7))
        self.assertTrue(torch.allclose(torch.autograd.grad(candidate, logits, retain_graph=True)[0],
                                      torch.autograd.grad(expected, logits)[0], atol=1e-7))

    def test_unknown_mode_cannot_silently_select_control(self):
        with self.assertRaisesRegex(ValueError, 'Unknown objective'):
            training_objective('typo', None, None, None, None)


if __name__ == '__main__':
    unittest.main()
