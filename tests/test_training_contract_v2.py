from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from beatfcn.train.density_curve import requested_density_curve
from beatfcn.train.mel_normalization import fit_mel_statistics, MelNormalizedFeatures
from beatfcn.train.training_objective import objective_terms, update_learning_rate
from beatfcn.train.training_panel import build_training_panel
from tests.test_loader_pipeline import make_panel
from tests import test_full_event_model as fixtures
from beatfcn.model.full_event_model import FullEventModel, event_loss


class TrainingContractV2Tests(unittest.TestCase):
    def test_caller_density_preserves_high_values_zero_and_gaps(self):
        segments = [dict(start_us=8_000_000, end_us=8_050_000, density=0),
            dict(start_us=8_100_000, end_us=8_200_000, density=32)]
        row = requested_density_curve(segments, 8_000_000, 9_000_000)
        np.testing.assert_allclose(row['intensity_times'], [.025, .15])
        np.testing.assert_allclose(row['intensity'][:, 1:], [[0, .05], [.1, .2]])
        self.assertEqual(row['intensity'][0, 0], 0)
        self.assertGreater(row['intensity'][-1, 0], 1)
        self.assertFalse(requested_density_curve([], 0, 1_000_000)['intensity_valid'].any())
        with self.assertRaises(ValueError):
            requested_density_curve([dict(start_us=0, end_us=100, density=float('nan'))], 0, 1000)

    def test_train_only_mel_statistics_and_frame_clock_are_used(self):
        with tempfile.TemporaryDirectory() as folder:
            panel = make_panel(Path(folder))
            audio = 'a'*64
            stats = fit_mel_statistics(panel.features, [audio])
            wrapped = MelNormalizedFeatures(panel.features, stats, train_audios=[audio], forbidden_audios=[])
            row = wrapped.window(audio, 0, 8_000_000)
            raw = panel.features.window(audio, 0, 8_000_000)
            np.testing.assert_allclose(row['mel'], (raw['mel']-np.asarray(stats['mean'], np.float32))/np.asarray(stats['std'], np.float32))
            with self.assertRaises(ValueError):
                fit_mel_statistics(panel.features, [audio], [audio])
            with self.assertRaises(ValueError):
                MelNormalizedFeatures(panel.features, stats, train_audios=[audio], forbidden_audios=[audio])
            times = np.load(Path(folder)/'features'/audio/'times_s.npy')
            np.testing.assert_allclose(raw['local_times'], times[:len(raw['local_times'])], atol=1e-6)

    def test_new_training_entry_rejects_legacy_clock_and_missing_version(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_panel(root)
            plan = dict(tape='tape.json', mert_features=str(root/'features'), mel_normalization='none')
            with self.assertRaisesRegex(ValueError, 'version'):
                build_training_panel(root, plan)
            plan['data_contract_version'] = 2
            with self.assertRaisesRegex(ValueError, 'native-clock'):
                build_training_panel(root, plan)
            d = json.loads((root/'tape.json').read_text())
            d.update(clock_contract='native-v1', drag_policy_id='all-events-1s-gt12-drag-only-v2')
            (root/'tape.json').write_text(json.dumps(d))
            panel = build_training_panel(root, plan)
            self.assertEqual(len(panel.records), 3)

    def test_qk_norm_gain_objective_and_cached_decode(self):
        fixture = fixtures.EventModelTests()
        fixture.setUp()
        model = FullEventModel(replace(fixture.c, qk_norm=True, learnable_audio_scale=True))
        model.eval()
        x, y = fixture.inputs, fixture.seq
        full = model(x, y)
        memory = model.encode(x)
        cache = model.start_cache(memory)
        pieces = [model.decode(y['tokens'][:, i:i+1], y['fields'][:, i:i+1],
            y['times'][:, i:i+1], y['valid'][:, i:i+1], memory, cache=cache)
            for i in range(y['tokens'].shape[1])]
        torch.testing.assert_close(full, torch.cat(pieces, 1), atol=2e-5, rtol=2e-5)
        terms = objective_terms(full, y, type_weight=.1, z_weight=1e-4)
        terms['total'].backward()
        for name, p in model.named_parameters():
            if 'q_norm' in name or 'k_norm' in name or 'input_scale' in name:
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
        torch.testing.assert_close(objective_terms(full, y)['total'], event_loss(full, y))

    def test_warmup_cosine_uses_optimizer_updates(self):
        p = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([p])
        rates = [update_learning_rate(optimizer, i, 6, base_lr=.01,
            warmup_updates=2, minimum_lr=.001) for i in range(6)]
        self.assertEqual(rates[:2], [.005, .01])
        self.assertAlmostEqual(rates[-1], .001)
        self.assertTrue(all(a >= b for a, b in zip(rates[2:], rates[3:])))


if __name__ == '__main__':
    unittest.main()
