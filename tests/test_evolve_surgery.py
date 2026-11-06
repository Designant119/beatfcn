import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from beatfcn.model.full_event_codec import encode_events, EventRequest, teacher_sequence
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from scripts.evolve_surgery import apply_surgery, kept_layers, load_checkpoint_model, macs_per_audio_second, run, shrink_units
from tests.test_full_event_model import inputs

R5 = dict(d=896, heads=14, kv_heads=2, ffn=3584, encoder_layers=3, decoder_layers=16, tap_dim=1024, taps=2, bands=1,
          band_slots=1, mel_bins=128, intensity_dim=3, qk_norm=True, learnable_audio_scale=True, timed_intensity=True,
          onset_head=True, beat_features=True)


def tiny():
    torch.manual_seed(3)
    config = FullEventConfig(d=32, heads=4, kv_heads=2, ffn=48, encoder_layers=1, decoder_layers=6, tap_dim=16, taps=2,
                             bands=1, band_slots=1, dropout=0., qk_norm=True)
    return FullEventModel(config).eval(), config


def batch(config):
    torch.manual_seed(7)
    request = EventRequest('phira', 0, 8_000_000, 10_000_000)
    events = [RhythmEvent(1000, 1000, 'tap'), RhythmEvent(2_000_000, 2_500_000, 'hold'), RhythmEvent(3_000_000, 3_000_000, 'drag')]
    sequence = {k: torch.as_tensor(v)[None] for k, v in teacher_sequence(encode_events(events, request), request).items()}
    return inputs(config), sequence


class KeptLayersTest(unittest.TestCase):
    def test_even_pattern_keeps_first_and_last_for_every_candidate_depth(self):
        for depth in (12, 10, 8, 6):
            keep = kept_layers(16, depth)
            self.assertEqual((len(set(keep)), keep[0], keep[-1]), (depth, 0, 15))
            self.assertEqual(keep, sorted(keep))
            gaps = {b-a for a, b in zip(keep, keep[1:])}
            self.assertLessEqual(max(gaps)-min(gaps), 1)
        self.assertEqual(kept_layers(16, 16), list(range(16)))

    def test_invalid_requests(self):
        with self.assertRaises(ValueError):
            kept_layers(16, 18)
        with self.assertRaises(ValueError):
            kept_layers(16, 8, 'random')


class SurgeryTest(unittest.TestCase):
    def forward(self, state, config_dict):
        model = FullEventModel(FullEventConfig(**config_dict)).eval()
        model.load_state_dict(state, strict=True)
        inp, seq = batch(model.c)
        with torch.no_grad():
            return model(inp, seq)

    def test_depth_unchanged_is_byte_equal_in_outputs(self):
        model, config = tiny()
        state, new_config, record = apply_surgery(model.state_dict(), asdict(config))
        self.assertEqual(new_config, asdict(config))
        self.assertEqual(record['kept_layers'], list(range(6)))
        self.assertTrue(torch.equal(self.forward(state, new_config), self.forward(model.state_dict(), asdict(config))))
        self.assertTrue(all(torch.equal(state[k], v) for k, v in model.state_dict().items()))

    def test_depth_cut_forward_is_valid_and_kept_layers_are_identical(self):
        model, config = tiny()
        original = model.state_dict()
        state, new_config, record = apply_surgery(original, asdict(config), depth=4)
        self.assertEqual(new_config['decoder_layers'], 4)
        self.assertEqual(record['kept_layers'], [0, 2, 3, 5])
        logits = self.forward(state, new_config)
        self.assertTrue(torch.isfinite(logits).all())
        for new, old in enumerate(record['kept_layers']):
            for key in [k for k in original if k.startswith(f'decoder.{old}.')]:
                self.assertTrue(torch.equal(state[key.replace(f'decoder.{old}.', f'decoder.{new}.', 1)], original[key]))
        self.assertTrue(torch.equal(state['output.weight'], original['output.weight']))
        self.assertFalse(any(k.startswith('decoder.4.') or k.startswith('decoder.5.') for k in state))
        self.assertFalse(torch.equal(logits, self.forward(original, asdict(config))))

    def test_ffn_shrink_drops_only_units_that_contribute_nothing(self):
        model, config = tiny()
        keep_units = torch.arange(0, 48, 2)
        with torch.no_grad():
            for block in model.decoder:
                dead = torch.ones(48, dtype=torch.bool)
                dead[keep_units] = False
                block.ffn.down.weight[:, dead] = 0.
        state, new_config, _ = apply_surgery(model.state_dict(), asdict(config), ffn=24)
        self.assertEqual(new_config['decoder_ffn'], 24)
        self.assertEqual(new_config['ffn'], 48)   # encoders stay
        self.assertEqual(state['decoder.0.ffn.up.weight'].shape, (48, 32))
        self.assertEqual(state['decoder.0.ffn.down.weight'].shape, (32, 24))
        self.assertEqual(state['encoders.0.0.ffn.down.weight'].shape, (32, 48))
        torch.testing.assert_close(self.forward(state, new_config), self.forward(model.state_dict(), asdict(config)),
                                   atol=2e-5, rtol=2e-5)

    def test_shrink_keeps_gate_value_and_projection_of_the_same_units(self):
        torch.manual_seed(0)
        up, down = torch.randn(16, 5), torch.randn(5, 8)
        small_up, small_down, keep = shrink_units(up, down, 3)
        score = up[:8].norm(dim=1)*up[8:].norm(dim=1)*down.norm(dim=0)
        self.assertEqual(sorted(keep.tolist()), sorted(score.topk(3).indices.tolist()))
        self.assertTrue(torch.equal(small_up[:3], up[keep]))
        self.assertTrue(torch.equal(small_up[3:], up[8+keep]))
        self.assertTrue(torch.equal(small_down, down[:, keep]))

    def test_depth_and_ffn_together_and_ffn_cannot_grow(self):
        model, config = tiny()
        state, new_config, _ = apply_surgery(model.state_dict(), asdict(config), depth=3, ffn=32)
        self.assertTrue(torch.isfinite(self.forward(state, new_config)).all())
        with self.assertRaises(ValueError):
            apply_surgery(model.state_dict(), asdict(config), ffn=64)

    def test_file_round_trip_has_no_optimizer_and_loads_like_an_export(self):
        model, config = tiny()
        with tempfile.TemporaryDirectory() as d:
            source, output = Path(d)/'r.pt', Path(d)/'cand.pt'
            torch.save(dict(model=model.state_dict(), config=asdict(config), vocabulary={}, optimizer=dict(state=1),
                            schema='beatfcn.final-model.v1', weights='ema', provenance=dict(plan='p')), source)
            summary = run(source, output, depth=4)
            saved = torch.load(output, weights_only=False)
            self.assertNotIn('optimizer', saved)
            self.assertEqual(saved['surgery']['kept_layers'], [0, 2, 3, 5])
            self.assertEqual(saved['provenance'], dict(plan='p'))
            self.assertEqual(summary['parameters'], sum(v.numel() for v in saved['model'].values()))
            self.assertLess(summary['macs_per_audio_second'], summary['source_macs_per_audio_second'])
            loaded = load_checkpoint_model(output, 'cpu')
            self.assertFalse(loaded.training)
            self.assertEqual(len(loaded.decoder), 4)


class RealShapeTest(unittest.TestCase):
    def test_r5_config_on_the_meta_device_matches_the_export_and_prunes(self):
        with torch.device('meta'):
            model = FullEventModel(FullEventConfig(**R5))
        state, config = model.state_dict(), asdict(model.c)
        self.assertAlmostEqual(sum(v.numel() for v in state.values())/1e6, 321.7, delta=.5)
        pruned, new_config, record = apply_surgery(state, config, depth=8)
        self.assertEqual(record['kept_layers'], [0, 2, 4, 6, 9, 11, 13, 15])
        self.assertEqual(set(pruned), {k for k in pruned if not k.startswith(tuple(f'decoder.{i}.' for i in range(8, 16)))})
        with torch.device('meta'):
            FullEventModel(FullEventConfig(**new_config)).load_state_dict(pruned, strict=True, assign=True)


class MacsTest(unittest.TestCase):
    def test_estimate_reproduces_report_e_for_r5(self):
        macs = macs_per_audio_second(R5)
        self.assertAlmostEqual(macs['decoder_row_step_macs']/1e6, 294, delta=1)
        self.assertAlmostEqual(macs['macs_per_audio_second']*8/1e9, 112+86, delta=2)
        small = macs_per_audio_second(dict(R5, decoder_layers=8))
        narrow = macs_per_audio_second(dict(R5, decoder_ffn=1792))
        self.assertAlmostEqual(small['decoder_row_step_macs'], macs['decoder_row_step_macs']/2)
        self.assertLess(narrow['macs_per_audio_second'], macs['macs_per_audio_second'])


if __name__ == '__main__':
    unittest.main()
