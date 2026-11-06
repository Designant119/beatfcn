import unittest

import torch

from beatfcn.model.anchor_mark_chain import AnchorMarkChain, mark_loss, sampled_marks
from beatfcn.model.no_mark_history import NoMarkHistory
from beatfcn.train.anchor_marks import teacher_example
from scripts.run_local_portfolio_scale import weight_digest
import tests.test_anchor_mark_chain as fixture


class NoMarkHistoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_initialization_and_gate_one_are_exactly_original(self):
        torch.manual_seed(20261032); old = AnchorMarkChain()
        torch.manual_seed(20261032); new = NoMarkHistory(gate=1)
        self.assertEqual(weight_digest(old), weight_digest(new))
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        x, y, heads = teacher_example(fixture.AnchorMarkTests.row())
        for key, value in old.teacher(x, y['kind']).items():
            torch.testing.assert_close(value, new.teacher(x, y['kind'])[key], rtol=0, atol=0)
        args = dict(index=7, view_enum=0, seeds=[2026102901, 2026102902])
        self.assertEqual(sampled_marks(old, x, heads, 100000, **args),
                         sampled_marks(new, x, heads, 100000, **args))

    def test_gate_zero_is_invariant_to_labels_and_preserves_acoustic_gradients(self):
        torch.manual_seed(5); model = NoMarkHistory()
        x, y, _ = teacher_example(fixture.AnchorMarkTests.row())
        original = model.teacher(x, y['kind'])
        changed = model.teacher(x, (y['kind'] + 1) % 4)
        for key in original:
            torch.testing.assert_close(original[key], changed[key], rtol=0, atol=0)
        loss, _ = mark_loss(model, x, y); loss.backward()
        self.assertEqual(float(model.types.weight.grad[:4].abs().max()), 0)
        self.assertGreater(float(model.types.weight.grad[4].abs().max()), 0)
        for module in (model.projection, model.gru, model.kind, model.tail_block):
            self.assertTrue(any(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                and bool(p.grad.abs().sum() > 0) for p in module.parameters()))

    def test_bos_and_reference_are_retained(self):
        torch.manual_seed(5); model = NoMarkHistory()
        x, y, _ = teacher_example(fixture.AnchorMarkTests.row())
        ids = torch.tensor([[4, 0, 1, 2, 3]])
        value = model.types(ids)
        torch.testing.assert_close(value[0, 0], model.types.weight[4])
        self.assertEqual(float(value[0, 1:].abs().max()), 0)
        x['reference_present'][:] = True; x['reference'][:] = 1
        first = model.teacher(x, y['kind'])['kind']
        x['reference_present'][:] = False
        self.assertGreater(float((first - model.teacher(x, y['kind'])['kind']).abs().max()), 0)
