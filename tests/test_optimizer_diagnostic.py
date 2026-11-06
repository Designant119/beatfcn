"""Check diagnostic proposals cannot corrupt the baseline optimizer."""
from copy import deepcopy
import unittest

import torch

from scripts.diagnose_optimizer_updates import hypothetical_step, tree_sha


class OptimizerDiagnosticTest(unittest.TestCase):
    def test_proposal_matches_independent_step_without_mutating_source(self):
        with torch.random.fork_rng():
            torch.manual_seed(19)
            model = torch.nn.Linear(3, 2)
            inputs = torch.randn(4, 3)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=0.)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                model(inputs).square().mean().backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            model(inputs).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            before = tree_sha((model.state_dict(), optimizer.state_dict(), [p.grad for p in model.parameters()]))
            state = torch.random.get_rng_state().clone()
            candidate = hypothetical_step(model, optimizer, 3.)
            reference = deepcopy(model)
            reference_opt = torch.optim.AdamW(reference.parameters(), lr=.0003, weight_decay=0.)
            reference_opt.load_state_dict(deepcopy(optimizer.state_dict()))
            for group in reference_opt.param_groups:
                group['lr'] *= 3.
            for a, b in zip(model.parameters(), reference.parameters()):
                b.grad = a.grad.clone()
            reference_opt.step()
            self.assertEqual(tree_sha(candidate.state_dict()), tree_sha(reference.state_dict()))
            self.assertEqual(before, tree_sha((model.state_dict(), optimizer.state_dict(), [p.grad for p in model.parameters()])))
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
            self.assertNotEqual(tree_sha(model.state_dict()), tree_sha(candidate.state_dict()))


if __name__ == '__main__':
    unittest.main()
