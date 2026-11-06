import unittest

import torch

from beatfcn.train.training_objective import training_loss
from tests.test_full_event_model import EventModelTests


class CompiledLossTests(unittest.TestCase):
    def test_full_model_loss_and_all_gradients_match_without_renaming_state(self):
        fixture = EventModelTests()
        fixture.setUp()
        model, inputs, sequence = fixture.model, fixture.inputs, fixture.seq
        original_keys = set(model.state_dict())
        settings = dict(type_weight=.1, z_weight=.0001, first_token_weight=5.)
        def loss_fn(inp, seq):
            return training_loss(model(inp, seq), seq, settings)
        expected = loss_fn(inputs, sequence)
        expected_grad = torch.autograd.grad(expected, tuple(model.parameters()))
        compiled = torch.compile(loss_fn, backend='aot_eager', dynamic=True, fullgraph=True)
        actual = compiled(inputs, sequence)
        actual_grad = torch.autograd.grad(actual, tuple(model.parameters()))
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        for a, b in zip(actual_grad, expected_grad):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
        self.assertEqual(set(model.state_dict()), original_keys)


class RegionalCompileTests(unittest.TestCase):
    def test_repeated_regions_preserve_all_gradients_and_state_names(self):
        from beatfcn.train.compiled_training import enable_regional_compile
        fixture = EventModelTests()
        fixture.setUp()
        model, inputs, sequence = fixture.model, fixture.inputs, fixture.seq
        names = set(model.state_dict())
        expected = training_loss(model(inputs, sequence), sequence)
        expected_grad = torch.autograd.grad(expected, tuple(model.parameters()))
        count = enable_regional_compile(model, backend='aot_eager')
        actual = training_loss(model(inputs, sequence), sequence)
        actual_grad = torch.autograd.grad(actual, tuple(model.parameters()))
        self.assertEqual(count, 7)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        for a, b in zip(actual_grad, expected_grad):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
        self.assertEqual(set(model.state_dict()), names)
