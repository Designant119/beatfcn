import unittest

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest
from beatfcn.model.full_event_model import Attention, FullEventConfig
from beatfcn.model.local_mamba_canvas import MambaCanvas
from beatfcn.model.mamba2_reference import Mamba2Reference
from beatfcn.train.local_typed_canvas import request_layout


def small_config():
    return FullEventConfig(d=32, heads=4, kv_heads=2, ffn=48, encoder_layers=1,
                           decoder_layers=0, tap_dim=16, bands=4, dropout=0.,
                           mel_bins=8, ssm_state=8, ssm_headdim=16)


def synthetic_inputs():
    request = EventRequest('phira', 0, 120_000, 4_000_000)
    inputs = dict(
        local=torch.randn(1, 3, 5, 4, 16),
        local_times=torch.arange(5)[None].float() * .04 + .02,
        local_valid=torch.tensor([[True, True, True, False, False]]),
        song=torch.randn(1, 5, 16),
        song_times=torch.arange(5)[None].float() * .5 + .25,
        song_valid=torch.tensor([[True, True, True, True, False]]),
        mel=torch.randn(1, 12, 8),
        mel_times=torch.arange(12)[None].float() * .01 + .005,
        mel_valid=torch.tensor([[True] * 10 + [False] * 2]),
        reference=torch.randn(1, 4, 16),
        reference_times=torch.tensor([[-.2, .1, .4, .8]]),
        reference_valid=torch.tensor([[True, True, True, False]]),
        controls=torch.zeros(1, 8, 6),
        mode=torch.tensor([1]),
    )
    inputs.update({k: torch.from_numpy(np.asarray(v))[None]
                   for k, v in request_layout(request).items()})
    return inputs, request


class MambaCanvasTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(20261027)

    def test_module_tree_has_no_attention_or_autoregressive_vocabulary(self):
        model = MambaCanvas(small_config(), density_reference=4)
        self.assertFalse(any(isinstance(module, Attention) for module in model.modules()))
        self.assertFalse(hasattr(model, 'vocab'))
        for name, _ in model.named_modules():
            self.assertNotIn('token', name)
            self.assertNotIn('decoder', name)
            self.assertNotIn('field_embedding', name)

    def test_output_contract_is_finite_and_differentiable(self):
        model = MambaCanvas(small_config(), density_reference=4)
        inputs, _ = synthetic_inputs()
        output = model(inputs)
        self.assertEqual(set(output), {'count', 'base_count', 'existence', 'offset',
                                      'tail_block', 'tail_cell', 'tail_ms'})
        expected = {
            'count': (1, 200, 4, 17), 'base_count': (1, 200, 4, 17),
            'existence': (1, 200, 4), 'offset': (1, 200, 4, 16, 56),
            'tail_block': (1, 200, 16, 601), 'tail_cell': (1, 200, 16, 50),
            'tail_ms': (1, 200, 16, 40),
        }
        for name, shape in expected.items():
            self.assertEqual(output[name].shape, shape, name)
            self.assertTrue(torch.isfinite(output[name]).all(), name)
        sum(value.float().mean() for value in output.values()).backward()
        parameters = dict(model.named_parameters())
        for name in ('encoders.0.0.mixer.in_proj.weight',
                     'address.0.mixer.in_proj.weight',
                     'temporal.0.mixer.in_proj.weight',
                     'offset.weight', 'tail_block.weight'):
            gradient = parameters[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(float(gradient.norm()), 0, name)

    def test_invalid_audio_reference_padding_never_enters_state_or_grid(self):
        model = MambaCanvas(small_config(), density_reference=4).eval()
        inputs, _ = synthetic_inputs()
        changed = {key: value.clone() for key, value in inputs.items()}
        for key in ('mel', 'song', 'reference'):
            changed[key][~changed[key + '_valid']] = float('nan')
            changed[key + '_times'][~changed[key + '_valid']] = float('nan')
        changed['local'][:, :, 3:] = float('nan')
        changed['local_times'][:, 3:] = float('nan')
        with torch.no_grad():
            expected = model(inputs)
            actual = model(changed)
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], atol=1e-6, rtol=1e-6)
            memory = model.encode(inputs)
        self.assertEqual(memory['audio'].shape[1], 10)
        self.assertTrue(memory['audio_valid'].all())
        torch.testing.assert_close(memory['audio_times'], inputs['mel_times'][:, :10])

    def test_default_mixers_are_bidirectional_ssd_within_parameter_budget(self):
        with torch.device('meta'):
            model = MambaCanvas(density_reference=4)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        self.assertGreaterEqual(parameters, 8_000_000)
        self.assertLessEqual(parameters, 14_000_000)
        blocks = [block for stream in model.encoders for block in stream]
        blocks += list(model.address) + list(model.temporal)
        self.assertEqual(len(model.encoders), 3)
        self.assertEqual(len(model.address), 2)
        self.assertEqual(len(model.temporal), 2)
        for block in blocks:
            self.assertTrue(block.bidirectional)
            self.assertIsInstance(block.mixer, Mamba2Reference)
            self.assertEqual(block.mixer.dim, model.c.d)
            self.assertEqual(block.mixer.inner, 2 * model.c.d)
            self.assertEqual(block.mixer.d_state, model.c.ssm_state)
            self.assertEqual(block.mixer.headdim, model.c.ssm_headdim)

    def test_canvas_loss_and_relative_generation_consume_public_outputs(self):
        from beatfcn.model.local_mamba_canvas import canvas_loss
        from beatfcn.train.local_typed_canvas import encode_canvas
        from beatfcn.train.relative_canvas import generate_relative

        model = MambaCanvas(small_config(), density_reference=4)
        inputs, request = synthetic_inputs()
        events = [RhythmEvent(4_000, 2_012_000, 'hold'),
                  RhythmEvent(5_000, 5_000, 'flick')]
        target = {key: torch.from_numpy(value)[None]
                  for key, value in encode_canvas(events, request).items()}
        loss, terms = canvas_loss(model(inputs), target, inputs)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(value) for value in terms.values()))
        loss.backward()
        self.assertGreater(float(model.frame_read.weight.grad.norm()), 0)

        # Isolate the decoder interface from untrained count quality by selecting
        # the empty count class; no decoding rule is changed in the model.
        with torch.no_grad():
            model.quantity.weight.zero_()
            model.quantity.bias.zero_()
            model.quantity.bias.view(4, 17)[:, 0] = 10
            generated = generate_relative(model, inputs, request)
        self.assertTrue(generated['complete'], generated['failure'])
        self.assertEqual(generated['events'], [])


if __name__ == '__main__':
    unittest.main()
