import unittest

import torch

from beatfcn.train.audio_dependence import freeze_teacher_panel
from scripts.teacher_ce_probe import compare, kind_calibration, probe, real_audio_protocol
from tests.test_benchmark_v2 import fixture


class ProbeTest(unittest.TestCase):
    def setUp(self):
        self.model, self.panel = fixture()
        self.protocol = freeze_teacher_panel(self.panel, windows=6)

    def test_per_field_and_mode_on_real_audio_only(self):
        result = probe(self.model, self.panel, self.protocol, device='cpu', batch_size=2)
        self.assertTrue(result['complete'])
        self.assertEqual(result['skipped_no_features'], 0)
        self.assertEqual(result['windows'], 6)
        self.assertEqual(set(result['field_ce']), {'all', 'phira', 'mania_4k'})
        for field in ('group', 'head_block', 'head_cell', 'head_ms', 'tails', 'joint'):
            self.assertGreater(result['field_ce']['all'][field], 0)
        self.assertIsNotNone(result['field_ce']['phira']['kind'])
        self.assertIsNotNone(result['field_ce']['mania_4k']['kind'])
        self.assertNotIn('mania', result['field_ce']['mania_4k'])

    def test_donors_are_not_needed_and_songs_without_features_are_skipped(self):
        first = self.protocol['windows'][0]['audio_sha256']
        self.panel.missing_audio = {first}
        real, skipped = real_audio_protocol(self.protocol, self.panel)
        self.assertEqual(real['conditions'], ['real'])
        self.assertGreaterEqual(skipped, 1)
        self.assertFalse(any('foreign_index' in w or w['audio_sha256'] == first for w in real['windows']))
        result = probe(self.model, self.panel, self.protocol, device='cpu', batch_size=2)
        self.assertEqual(result['skipped_no_features'], skipped)
        self.assertEqual(result['windows'], 6-skipped)

    def test_kind_calibration_marginals_are_distributions_and_low_temperature_sharpens(self):
        real, _ = real_audio_protocol(self.protocol, self.panel)
        calibration = kind_calibration(self.model, self.panel, real, device='cpu', batch_size=2)
        self.assertGreater(calibration['tokens'], 0)
        self.assertAlmostEqual(sum(calibration['target_share'].values()), 1, places=5)
        for marginal in calibration['teacher_forced_marginal'].values():
            self.assertAlmostEqual(sum(marginal.values()), 1, places=4)
        one, cold = (calibration['teacher_forced_marginal'][k] for k in ('1.0', '0.3'))
        self.assertGreater(max(cold.values()), max(one.values()))

    def test_gross_kill_uses_the_baseline(self):
        baseline = probe(self.model, self.panel, self.protocol, device='cpu', batch_size=2)
        same = probe(self.model, self.panel, self.protocol, device='cpu', batch_size=2, baseline=baseline)
        self.assertAlmostEqual(same['versus_baseline']['worst_joint_delta_nats'], 0., places=5)
        self.assertFalse(same['versus_baseline']['gross_kill'])
        worse = {m: dict(r, joint=r['joint']-1.) for m, r in baseline['field_ce'].items()}
        self.assertTrue(compare(baseline['field_ce'], worse, .5)['gross_kill'])

    def test_a_shrunk_model_is_probed_without_a_full_tape_model(self):
        from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
        from scripts.evolve_surgery import apply_surgery
        state, config, _ = apply_surgery(self.model.state_dict(), self.model.c.__dict__.copy(), ffn=24)
        small = FullEventModel(FullEventConfig(**config)).eval()
        small.load_state_dict(state)
        result = probe(small, self.panel, self.protocol, device='cpu', batch_size=2)
        self.assertEqual(result['decoder_ffn'], 24)
        self.assertLess(result['parameters'], sum(p.numel() for p in self.model.parameters()))
        self.assertTrue(result['complete'])


if __name__ == '__main__':
    unittest.main()
