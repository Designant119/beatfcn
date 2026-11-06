import unittest

import torch

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_music_research import BudgetAllocationCanvas, BudgetTemporal, ContextMatchingCanvas


class MusicResearchTests(unittest.TestCase):
    def test_complete_budget_with_shared_boundary(self):
        c = FullEventConfig(d=128, heads=4, kv_heads=2, ffn=384, encoder_layers=1,
                            decoder_layers=0, dropout=0., mel_bins=128)
        models = []
        for width, depth in ((192, 8), (144, 16)):
            torch.manual_seed(12)
            models.append(BudgetAllocationCanvas(c, density_reference=6., width=width, depth=depth))
        counts = [sum(p.numel() for p in m.parameters()) for m in models]
        self.assertLess(max(counts) / min(counts), 1.05)
        first, second = [m.state_dict() for m in models]
        for key in first:
            if not key.startswith('temporal.'):
                self.assertTrue(torch.equal(first[key], second[key]), key)
        self.assertEqual(len(models[0].temporal[0].blocks), 8)
        self.assertEqual(len(models[1].temporal[0].blocks), 16)

    def test_matching_context_never_contains_center_audio(self):
        c = FullEventConfig(d=32, heads=4, kv_heads=2, ffn=64, encoder_layers=1,
                            decoder_layers=0, dropout=0., mel_bins=128)
        model = ContextMatchingCanvas(c, density_reference=6.).eval()
        times = torch.arange(1600)[None] * .01 - 4 + .005
        inputs = dict(mel=torch.randn(1, 1600, 128), mel_times=times,
                      mel_valid=torch.ones_like(times, dtype=torch.bool), canvas_width_s=torch.tensor([8.]))
        local, context = model.matching_views(inputs)
        self.assertFalse(bool((local & context).any()))
        self.assertEqual(int(local.sum()), 200)
        with torch.no_grad():
            before = model.context_embedding(inputs)
            changed = inputs['mel'].clone()
            changed[:, local] += 100.
            after = model.context_embedding({**inputs, 'mel': changed})
        self.assertTrue(torch.equal(before, after))
        shifted = {**inputs, 'mel_times': inputs['mel_times'] + 2,
                   'canvas_width_s': inputs['canvas_width_s'] + 4}
        with torch.no_grad():
            self.assertTrue(torch.allclose(model.context_embedding(inputs), model.context_embedding(shifted), atol=1e-6))
            self.assertTrue(torch.allclose(model.local_embedding(inputs), model.local_embedding(shifted), atol=1e-6))

    def test_budget_trunk_excludes_invalid_nan_before_projection(self):
        c = FullEventConfig(d=32, heads=4, kv_heads=2, ffn=64, dropout=0.)
        trunk = BudgetTemporal(c, 192, 8)
        x = torch.randn(1, 7, 32)
        x[:, 5:] = float('nan')
        valid = torch.tensor([[True] * 5 + [False] * 2])
        result = trunk(x, valid)
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertTrue(torch.equal(result[:, 5:], torch.zeros_like(result[:, 5:])))
        result.sum().backward()
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in trunk.parameters()))


if __name__ == '__main__':
    unittest.main()
