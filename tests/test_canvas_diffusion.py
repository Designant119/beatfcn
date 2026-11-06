import unittest

import numpy as np
import torch
from torch.nn import functional as F

from beatfcn.model.canvas_diffusion import (
    ABSENT,
    MASK,
    AbsorbingCanvasDiffusion,
    ancestral_sample,
    categorical_reverse,
    corrupt_packages,
    diffusion_loss,
    legal_tail_nll,
    reconstruction_probe,
)
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.train.local_typed_canvas import encode_canvas, request_layout
from beatfcn.train.relative_canvas import relative_targets


class CanvasDiffusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(41)

    def test_absorbing_forward_and_reverse_match_hand_computation(self):
        clean = torch.tensor([0, 2, 1])
        self.assertTrue(torch.equal(corrupt_packages(clean, 0, 4), clean))
        self.assertTrue(torch.equal(corrupt_packages(clean, 4, 4), torch.full_like(clean, MASK)))
        logits = torch.zeros(3)
        probabilities = categorical_reverse(logits, torch.tensor(3), 2)
        torch.testing.assert_close(probabilities, torch.tensor([1 / 6, 1 / 6, 1 / 6, 1 / 2]))
        final = categorical_reverse(logits, torch.tensor(3), 1)
        torch.testing.assert_close(final, torch.tensor([1 / 3, 1 / 3, 1 / 3, 0.]))
        carry = categorical_reverse(logits, torch.tensor(2), 2)
        torch.testing.assert_close(carry, torch.tensor([0., 0., 1., 0.]))

    def test_joint_tail_nll_matches_explicit_legal_enumeration(self):
        block = torch.tensor([.2, -.4, .7], requires_grad=True)
        cell = torch.tensor([.3, -.8], requires_grad=True)
        ms = torch.tensor([-.1, .5, .9], requires_grad=True)
        lo, hi, target = 2, 14, 7
        values = torch.arange(lo, hi + 1)
        scores = block[values // 6] + cell[(values % 6) // 3] + ms[values % 3]
        expected = -scores[target - lo] + torch.logsumexp(scores, 0)
        actual = legal_tail_nll(block, cell, ms, lo, hi, target, block_size=6, cell_size=3)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(block.grad).all())
        self.assertTrue(torch.isfinite(cell.grad).all())
        self.assertTrue(torch.isfinite(ms.grad).all())

    def _case(self):
        request = EventRequest('phira', 0, 80_000, 4_000_000)
        events = [RhythmEvent(4_000, 2_012_000, 'hold'),
                  RhythmEvent(5_000, 5_000, 'flick'),
                  RhythmEvent(5_000, 5_000, 'flick')]
        inputs = dict(local=torch.randn(1, 3, 2, 4, 16),
            local_times=torch.tensor([[.02, .06]]), local_valid=torch.ones(1, 2, dtype=torch.bool),
            song=torch.randn(1, 4, 16), song_times=torch.arange(4)[None] * .5 + .25,
            song_valid=torch.ones(1, 4, dtype=torch.bool), mel=torch.randn(1, 8, 8),
            mel_times=torch.arange(8)[None] * .01 + .005, mel_valid=torch.ones(1, 8, dtype=torch.bool),
            reference=torch.zeros(1, 1, 16), reference_times=torch.zeros(1, 1),
            reference_valid=torch.ones(1, 1, dtype=torch.bool), controls=torch.zeros(1, 8, 6),
            mode=torch.tensor([1]))
        inputs.update({k: torch.from_numpy(np.asarray(v))[None] for k, v in request_layout(request).items()})
        raw = encode_canvas(events, request)
        target = {k: torch.from_numpy(v)[None] for k, v in raw.items()}
        target = relative_targets(target, [request])
        return request, inputs, target

    def test_noise_and_time_change_real_forward_and_loss_has_gradients(self):
        request, inputs, target = self._case()
        config = FullEventConfig(d=32, heads=4, kv_heads=2, ffn=48, encoder_layers=1,
            decoder_layers=0, tap_dim=16, bands=4, dropout=0., mel_bins=8)
        model = AbsorbingCanvasDiffusion(config, density_reference=4., steps=4)
        noisy = model.corrupt(target, 2)
        output = model(inputs, noisy, torch.tensor([2]))
        changed_time = model(inputs, noisy, torch.tensor([1]))
        self.assertFalse(torch.allclose(output['offset0'], changed_time['offset0']))
        shuffled = dict(noisy)
        shuffled['offset'] = noisy['offset'].flip(-1)
        changed_noise = model(inputs, shuffled, torch.tensor([2]))
        self.assertFalse(torch.allclose(output['offset0'], changed_noise['offset0']))
        loss, detail = diffusion_loss(output, target, noisy, inputs, [request], torch.tensor([2]))
        loss.backward()
        for name in ('frame_read.weight', 'offset0.weight', 'tail_block.weight', 'time_ids.weight'):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(float(gradient.norm()), 0, name)
        self.assertGreater(float(detail['vb']), 0)
        self.assertGreater(float(detail['auxiliary']), 0)
        probe = reconstruction_probe(model, inputs, noisy, torch.tensor([2]), request)
        self.assertTrue(probe['complete'], probe.get('failure'))
        self.assertEqual(probe['decode_kind'], 'reconstruction_probe')
        balanced, balanced_detail = diffusion_loss(output, target, noisy, inputs, [request], torch.tensor([2]),
            balance_positive=True)
        self.assertTrue(torch.isfinite(balanced))
        self.assertGreater(float(balanced_detail['positive_vb']), 0)
        self.assertGreater(float(balanced_detail['absent_vb']), 0)
        self.assertEqual(float(balanced_detail['balanced']), 1.)
        self.assertNotEqual(float(balanced_detail['vb']), float(detail['vb']))

    def test_ancestral_generation_preserves_duplicates_and_legal_support(self):
        request, inputs, target = self._case()
        config = FullEventConfig(d=32, heads=4, kv_heads=2, ffn=48, encoder_layers=1,
            decoder_layers=0, tap_dim=16, bands=4, dropout=0., mel_bins=8)
        model = AbsorbingCanvasDiffusion(config, density_reference=4., steps=3).eval()
        result = ancestral_sample(model, inputs, request, seed=9)
        self.assertTrue(result['complete'], result.get('failure'))
        self.assertEqual(result['decode_kind'], 'ancestral')
        self.assertEqual(result['steps'], 3)
        allowed = inputs['canvas_head_allowed'][0].numpy()
        heads = inputs['canvas_head_us'][0].numpy()
        for event in result['events']:
            frame = (event['onset_us'] - request.start_us) // 40_000
            self.assertIn(event['onset_us'], heads[frame, allowed[frame]])
            if event['type'] == 'hold':
                self.assertGreater(event['end_us'], event['onset_us'])
                self.assertLessEqual(event['end_us'], request.duration_us)
        self.assertEqual(len(result['events']), result['predicted_count'])
        self.assertEqual(len(result['unlock_trace']), 3)
        noisy = model.corrupt(target, 2)
        resumed = ancestral_sample(model, inputs, request, seed=9, initial_noisy=noisy, start_step=2)
        self.assertTrue(resumed['complete'], resumed.get('failure'))
        self.assertEqual(len(resumed['unlock_trace']), 2)
        self.assertEqual([row['t'] for row in resumed['unlock_trace']], [2, 1])


if __name__ == '__main__':
    unittest.main()
