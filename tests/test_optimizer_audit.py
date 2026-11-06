from copy import deepcopy
import unittest
import torch

from beatfcn.train.optimizer_audit import adamw_proposal


class OptimizerAuditTests(unittest.TestCase):
    def test_proposal_matches_real_step_without_mutating_source(self):
        for decay in (0., .1):
            p = torch.nn.Parameter(torch.tensor([[.2,-.4],[.8,1.]]))
            optimizer = torch.optim.AdamW([p], lr=.001, weight_decay=decay, foreach=False)
            p.grad = torch.tensor([[.1, .2], [-.3, .0]])
            optimizer.step()
            p.grad = torch.tensor([[-.01, .1], [.2, .0]])
            before = p.detach().clone(); state = deepcopy(optimizer.state[p]); original = deepcopy(state)
            proposed = adamw_proposal(p, p.grad, state, optimizer.param_groups[0])
            torch.testing.assert_close(p, before, atol=0., rtol=0.)
            for key in state:
                torch.testing.assert_close(state[key], original[key], atol=0., rtol=0.)
            optimizer.step()
            torch.testing.assert_close(before+proposed['delta'], p, atol=1e-7, rtol=0.)
