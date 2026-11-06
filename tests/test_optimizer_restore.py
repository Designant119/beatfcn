"""Optimizer continuation preserves identities when input parameters are added."""
from copy import deepcopy
import unittest

import torch

from beatfcn.train.optimizer_restore import restore_named_optimizer_state


class OptimizerRestoreTests(unittest.TestCase):
    def test_added_parameter_does_not_shift_old_moments_or_mutate_parent(self):
        original = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(original.parameters(), lr=.001)
        for i, param in enumerate(original.parameters()):
            param.grad = torch.full_like(param, i + 1.)
        optimizer.step()
        saved = deepcopy(optimizer.state_dict())
        # A root parameter precedes the submodule's parameters in traversal.
        candidate = torch.nn.Module()
        candidate.extra = torch.nn.Parameter(torch.zeros(2))
        candidate.base = deepcopy(original)
        fresh = torch.optim.AdamW(candidate.parameters(), lr=123.)
        proof = restore_named_optimizer_state(
            fresh, saved, ['base.' + n for n, _ in original.named_parameters()],
            list(dict(candidate.named_parameters())))
        self.assertEqual(proof, {'restored_parameters': 2, 'new_parameters': ['extra']})
        self.assertEqual(fresh.param_groups[0]['lr'], .001)
        self.assertNotIn(candidate.extra, fresh.state)
        for (_, param), old_id in zip(candidate.base.named_parameters(), saved['param_groups'][0]['params']):
            for key, expected in saved['state'][old_id].items():
                actual = fresh.state[param][key]
                torch.testing.assert_close(actual, expected)
                self.assertIsNot(actual, expected)
        for param in candidate.parameters():
            param.grad = torch.ones_like(param)
        fresh.step()
        self.assertTrue(all(float(v['step']) == 1 for v in saved['state'].values()))

    def test_missing_or_duplicate_names_are_rejected(self):
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        saved = optimizer.state_dict()
        for before, after in [(['a', 'b'], ['a', 'a']), (['a', 'b'], ['a', 'c'])]:
            with self.assertRaises(ValueError):
                restore_named_optimizer_state(optimizer, saved, before, after)
