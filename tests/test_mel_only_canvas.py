import unittest

import torch

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_typed_canvas import canvas_loss
from beatfcn.model.mel_only_canvas import MelOnlyCanvas
from beatfcn.train.relative_canvas import generate_relative
from tests.test_local_mamba_canvas import small_config, synthetic_inputs


class MelOnlyCanvasTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(71)

    def test_pretrained_streams_are_disconnected_from_output_and_gradients(self):
        model = MelOnlyCanvas(small_config(), density_reference=4).eval()
        inputs, _ = synthetic_inputs()
        changed = {key: value.clone() for key, value in inputs.items()}
        changed['local'][:] = float('nan')
        changed['local_times'][:] = float('nan')
        changed['song'][:] = float('nan')
        changed['song_times'][:] = float('nan')
        with torch.no_grad():
            expected = model(inputs)
            actual = model(changed)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], atol=1e-6, rtol=1e-6)
        for name in ('local_projection', 'song_projection', 'band_queries', 'band_embedding', 'layer_mix'):
            self.assertFalse(hasattr(model, name), name)

    def test_real_canvas_loss_and_decoder_contract(self):
        from beatfcn.model.event_canvas import RhythmEvent
        from beatfcn.train.local_typed_canvas import encode_canvas

        model = MelOnlyCanvas(small_config(), density_reference=4)
        inputs, request = synthetic_inputs()
        target = {key: torch.from_numpy(value)[None] for key, value in encode_canvas(
            [RhythmEvent(4_000, 2_012_000, 'hold'), RhythmEvent(5_000, 5_000, 'flick')], request).items()}
        loss, detail = canvas_loss(model(inputs), target, inputs)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(value) for value in detail.values()))
        loss.backward()
        for name in ('mel_projection.weight', 'mel_context_projection.weight', 'offset.weight'):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(float(gradient.norm()), 0, name)
        with torch.no_grad():
            model.quantity.weight.zero_()
            model.quantity.bias.zero_()
            model.quantity.bias.view(4, 17)[:, 0] = 10
            result = generate_relative(model, inputs, request)
        self.assertTrue(result['complete'], result['failure'])
        self.assertEqual(result['events'], [])


if __name__ == '__main__':
    unittest.main()
