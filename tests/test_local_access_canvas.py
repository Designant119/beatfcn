import unittest

import torch

from beatfcn.model.local_access_canvas import AccessBlock, AccessCanvas
from tests.test_local_mamba_canvas import small_config, synthetic_inputs


class LocalAccessCanvasTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(391)

    def test_both_schedules_have_identical_parameters_and_initial_values(self):
        torch.manual_seed(44)
        global_model = AccessCanvas(small_config(), density_reference=4, access='global')
        torch.manual_seed(44)
        mixed_model = AccessCanvas(small_config(), density_reference=4, access='mixed')
        self.assertEqual(set(global_model.state_dict()), set(mixed_model.state_dict()))
        for name, value in global_model.state_dict().items():
            torch.testing.assert_close(value, mixed_model.state_dict()[name], atol=0, rtol=0)

    def test_window_blocks_distant_information_and_masks_padding(self):
        window = AccessBlock(small_config(), 'W', radius=2).eval()
        global_block = AccessBlock(small_config(), 'G', radius=2).eval()
        global_block.load_state_dict(window.state_dict())
        x = torch.randn(1, 12, 32)
        valid = torch.tensor([[True] * 11 + [False]])
        changed = x.clone()
        changed[:, 10] += torch.randn(32) * 10
        changed[:, 11] = float('nan')
        with torch.no_grad():
            w0, w1 = window(x, valid), window(changed, valid)
            g0, g1 = global_block(x, valid), global_block(changed, valid)
        torch.testing.assert_close(w0[:, :8], w1[:, :8], atol=0, rtol=0)
        self.assertGreater(float((g0[:, :8] - g1[:, :8]).abs().max()), 1e-4)
        self.assertTrue(torch.isfinite(w1).all())
        self.assertTrue(torch.isfinite(g1).all())
        self.assertEqual(float(w1[:, 11].abs().sum()), 0.)
        self.assertFalse(window.attention_mask(valid)[..., 11].any())

    def test_convolution_is_bidirectional_and_has_finite_parameter_gradients(self):
        block = AccessBlock(small_config(), 'C')
        x = torch.randn(1, 12, 32, requires_grad=True)
        valid = torch.ones(1, 12, dtype=torch.bool)
        block(x, valid)[:, 5].square().sum().backward()
        self.assertGreater(float(x.grad[:, 6:9].abs().sum()), 0.)
        self.assertEqual(float(x.grad[:, 9:].abs().sum()), 0.)
        for parameter in block.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_canvas_forward_keeps_mel_only_and_event_output_contract(self):
        model = AccessCanvas(small_config(), density_reference=4, access='mixed').eval()
        inputs, _ = synthetic_inputs()
        inputs = {k: v for k, v in inputs.items() if not k.startswith(('local', 'song'))}
        output = model(inputs)
        self.assertEqual(tuple(output['count'].shape), (1, 200, 4, 17))
        self.assertEqual(tuple(output['offset'].shape), (1, 200, 4, 16, 56))
        self.assertTrue(all(torch.isfinite(v).all() for v in output.values()))
        sum(v.mean() for v in output.values()).backward()
        self.assertIsNotNone(model.temporal[0].depthwise.weight.grad)
        self.assertIsNotNone(model.temporal[2].attention.q.weight.grad)


if __name__ == '__main__':
    unittest.main()
