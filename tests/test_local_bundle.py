import unittest

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import FIELDS, EventRequest
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_bundle import LocalBundleModel, bundle_loss, read_grid
from beatfcn.train.local_bundle import annotated_sequence


class LocalBundleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(27)
        torch.set_num_threads(2)
        c = FullEventConfig(d=32, heads=4, kv_heads=2, ffn=48, encoder_layers=1,
                            decoder_layers=2, tap_dim=16, bands=4, dropout=0., mel_bins=8)
        self.model = LocalBundleModel(c).eval()
        self.inputs = dict(local=torch.randn(1, 3, 8, 4, 16),
            local_times=torch.arange(8)[None] * .04 + .02, local_valid=torch.ones(1, 8, dtype=torch.bool),
            song=torch.randn(1, 10, 16), song_times=torch.arange(10)[None] * .5 + .25,
            song_valid=torch.ones(1, 10, dtype=torch.bool), mel=torch.randn(1, 32, 8),
            mel_times=torch.arange(32)[None] * .01 + .005, mel_valid=torch.ones(1, 32, dtype=torch.bool),
            reference=torch.zeros(1, 1, 16), reference_times=torch.zeros(1, 1),
            reference_valid=torch.ones(1, 1, dtype=torch.bool), controls=torch.zeros(1, 8, 6),
            mode=torch.tensor([1]), origin_s=torch.tensor([2.]), end_s=torch.tensor([4.]),
            duration_s=torch.tensor([8.]))
        self.request = EventRequest('phira', 2_000_000, 4_000_000, 8_000_000)
        events = [RhythmEvent(2_050_000, 6_120_000, 'hold'), RhythmEvent(2_050_000, 2_050_000, 'drag'),
                  RhythmEvent(2_170_000, 2_170_000, 'flick')]
        self.seq = {k: torch.from_numpy(v)[None] for k, v in annotated_sequence(events, self.request).items()}

    def test_full_cached_logits_counts_and_future_independence(self):
        keys = ('tokens', 'fields', 'times', 'valid')
        with torch.no_grad():
            memory = self.model.encode(self.inputs)
            full = self.model.decode(*(self.seq[k] for k in keys), memory)
            count = self.model.last_group_log_count.clone()
            cache = self.model.start_cache(memory)
            pieces, sizes = [], []
            for i in range(full.shape[1]):
                pieces.append(self.model.decode(*(self.seq[k][:, i:i+1] for k in keys), memory, cache=cache))
                sizes.append(self.model.last_group_log_count.clone())
            torch.testing.assert_close(full, torch.cat(pieces, 1), atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(count, torch.cat(sizes, 1), atol=1e-5, rtol=1e-5)
            changed = {k: v.clone() for k, v in self.seq.items()}
            changed['tokens'][:, 10:] = 1
            changed['times'][:, 10:] += 2
            other = self.model(self.inputs, changed)
            torch.testing.assert_close(full[:, :10], other[:, :10], atol=1e-5, rtol=1e-5)
        self.assertEqual(self.seq['group_log_count'][self.seq['count_valid']].tolist(),
                         np.log1p([2, 1]).astype(np.float32).tolist())

    def test_masked_features_and_real_gradient_paths(self):
        self.inputs['mel_valid'][:, -3:] = False
        changed = {k: v.clone() for k, v in self.inputs.items()}
        changed['mel'][:, -3:] = 10000
        a = self.model(self.inputs, self.seq)
        b = self.model(changed, self.seq)
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        loss = bundle_loss(self.model, b, self.seq)
        loss.backward()
        for name in ('audio_read.weight', 'time_score.weight', 'action_heads.kind.weight',
                     'group_size.weight', 'condition_direct.2.weight'):
            grad = dict(self.model.named_parameters())[name].grad
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(float(grad.norm()), 0, name)

    def test_grid_clock_interpolation_and_no_outside_evidence(self):
        values = torch.tensor([[[0.], [10.], [999.]]])
        times = torch.tensor([[.005, .015, 0.]])
        valid = torch.tensor([[True, True, False]])
        result = read_grid(values, times, valid, torch.tensor([[.01, .015, .03, -.1]]), .01)
        torch.testing.assert_close(result.flatten(), torch.tensor([5., 10., 0., 0.]))


if __name__ == '__main__':
    unittest.main()
