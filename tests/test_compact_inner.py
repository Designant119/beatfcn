import copy
from dataclasses import replace
import unittest
import torch
from beatfcn.model.p1 import P1Model
from beatfcn.model.batch import prepare_inner_indices
from test_p1_model import CFG, batch


class CompactInnerTests(unittest.TestCase):
    def compare(self, bt, placement):
        torch.manual_seed(23)
        reference = P1Model(replace(CFG, phira_location=placement))
        compact = P1Model(replace(CFG, phira_location=placement, compact_inner=True))
        compact.load_state_dict(reference.state_dict())
        la, ma = reference(copy.deepcopy(bt))
        lb, mb = compact(copy.deepcopy(bt))
        torch.testing.assert_close(la, lb, atol=1e-6, rtol=1e-5)
        for key in ma:
            if key == '_illegal':
                self.assertEqual(float(ma[key]), float(mb[key]))
            else:
                for a, b in zip(ma[key], mb[key]):
                    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5, msg=key)
        la.backward()
        lb.backward()
        for (name, a), b in zip(reference.named_parameters(), compact.parameters()):
            if a.grad is None:
                self.assertIsNone(b.grad, name)
            else:
                torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-4, msg=name)

    def test_mixed_modes_holds_and_padding_preserve_losses_and_gradients(self):
        bt = batch(4, seed=3)
        self.assertLess(len(bt['inner_index']), bt['grp_valid'].numel())
        for placement in (False, True):
            self.compare(bt, placement)

    def test_empty_groups_keep_zero_inner_loss_and_finite_gradients(self):
        bt = batch(2, seed=4)
        bt['grp_valid'].zero_()
        bt['hold_valid'].zero_()
        prepare_inner_indices(bt)
        self.assertEqual(len(bt['inner_index']), 1)
        self.compare(bt, False)


if __name__ == '__main__':
    unittest.main()
