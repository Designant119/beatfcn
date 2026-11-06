import unittest

import torch

from beatfcn.train.hn2_optimizer_audit import optimizer_inventory, parameter_adam_step
from beatfcn.train.hn2_safe_runtime import assess_stratified_health
from tests.test_hn2_safe_protocol import SafeHealthTests


class AuditTests(unittest.TestCase):
    def test_actual_adam_lazy_unused_parameter_is_zero_and_not_created(self):
        active = torch.nn.Parameter(torch.tensor([2.]))
        unused = torch.nn.Parameter(torch.tensor([3.]))
        optimizer = torch.optim.AdamW([active, unused], lr=.01)
        active.square().sum().backward();optimizer.step()
        self.assertEqual(parameter_adam_step(optimizer, active), (1, True))
        before = len(optimizer.state)
        self.assertEqual(parameter_adam_step(optimizer, unused), (0, False))
        self.assertEqual(len(optimizer.state), before)
        self.assertNotIn(unused, optimizer.state)

    def test_empty_state_also_stays_explicitly_uninitialized(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter])
        optimizer.state[parameter] = {}
        self.assertEqual(parameter_adam_step(optimizer, parameter), (0, False))

    def test_missing_parameter_step_does_not_relax_existing_health_gate(self):
        fixture = SafeHealthTests();fixture.setUp()
        verdict = assess_stratified_health(fixture.before, fixture.after, fixture.free,
                                          [100, 0], fixture.acceptance)
        self.assertFalse(verdict['checks']['actual_optimizer100'])
        self.assertFalse(verdict['passed'])


if __name__ == '__main__':
    unittest.main()
