"""Optional charter-timing (beat grid) input: features, model equivalence, data, rollout, checkpoints."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from beatfcn.model.full_event_model import BEAT_FEATURE_DIM, FullEventConfig, FullEventModel
from beatfcn.train.charter_timing import (BEAT_FEATURE_NAMES, SCHEMA, ChartTimingTable, beat_features,
    normalize_charter_timing, timing_dropped, window_beat_features)
from beatfcn.train.counterfactual_exposure import build_exposure
from beatfcn.train.exposure_loader import CounterfactualExposureDataset, plan_loader_settings
from beatfcn.train.local_panel import collate
from beatfcn.train.local_shapes import pad_input_shapes
from test_full_event_review_fixes import make_inputs, make_sequence, tiny
from test_history_context import ChartPanel, _digest, _slots

F = {name: i for i, name in enumerate(BEAT_FEATURE_NAMES)}


def assert_phase(actual, expected, atol=1e-6):
    """Circular comparison of phases in [0, 1)."""
    d = np.abs(np.asarray(actual, np.float64)-np.asarray(expected, np.float64)) % 1
    np.testing.assert_array_less(np.minimum(d, 1-d), atol)


def table_document(charts):
    return dict(schema=SCHEMA, settings={}, coverage={}, charts=charts)


def write_table(directory, charts):
    path = Path(directory)/'timing.json'
    path.write_text(json.dumps(table_document(charts)))
    return path


class FeatureTests(unittest.TestCase):
    # 120 BPM in 3/4 from 1 s, then 150 BPM in 4/4 from 4 s (on the 6th beat).
    SEGMENTS = [(1_000_000, 500_000, 3), (4_000_000, 400_000, 4)]

    def test_phase_at_grid_points_bpm_change_and_meter(self):
        times = np.array([1_000_000, 1_250_000, 1_500_000, 2_500_000, 4_000_000, 4_100_000, 4_200_000,
                          5_600_000, 500_000])
        f = beat_features(times, self.SEGMENTS)
        self.assertEqual(f.shape, (len(times), BEAT_FEATURE_DIM))
        beat = np.arctan2(f[:, F['beat_sin']], f[:, F['beat_cos']]) % (2*np.pi)/(2*np.pi)
        assert_phase(beat, [0, .5, 0, 0, 0, .25, .5, 0, 0])
        bar = np.arctan2(f[:, F['bar_sin']], f[:, F['bar_cos']]) % (2*np.pi)/(2*np.pi)
        # 3/4: beat 1 of the bar at 1.5 s -> 1/3; 2.5 s = beat 3 -> bar start; after the change 4/4
        # restarts at the red line: 5.6 s = 4 beats later -> bar start. 0.5 s extends segment 0 backwards
        # (beat -1 -> bar phase 2/3).
        assert_phase(bar, [0, 1/6, 1/3, 0, 0, 1/16, 1/8, 0, 2/3])
        np.testing.assert_allclose(f[5, F['beat2_cos']], -1, atol=1e-6)  # 1/4 beat = half the 2x period
        np.testing.assert_allclose(f[5, F['beat4_cos']], 1, atol=1e-6)   # 1/4 beat is on the 4x grid
        np.testing.assert_allclose(f[6, F['beat2_cos']], 1, atol=1e-6)   # 1/2 beat is on the 2x grid
        np.testing.assert_allclose(f[:, F['log_beat']], np.log([1, 1, 1, 1, .8, .8, .8, .8, 1]), atol=1e-6)
        self.assertTrue(np.all(f[:, F['present']] == 1))

    def test_signed_offsets_to_quarter_and_third(self):
        # 120 BPM: 1/4 = 125 ms, 1/3 = 166.667 ms.
        f = beat_features([1_003_000, 997_000, 1_125_000+20_000, 1_166_667+40_000], [(1_000_000, 500_000, 4)])
        np.testing.assert_allclose(f[:, F['offset_quarter']], [3/25, -3/25, 20/25, -1.], atol=1e-5)
        np.testing.assert_allclose(f[:, F['offset_third']][:2], [3/25, -3/25], atol=1e-5)
        self.assertAlmostEqual(float(f[3, F['offset_third']]), 1., places=4)  # clipped at +25 ms

    def test_absent_and_invalid_frames_are_zero(self):
        self.assertFalse(beat_features(np.arange(10)*10_000., None).any())
        inputs = dict(mel_times=np.array([-.005, .005, .015], np.float32), mel_valid=np.array([False, True, True]))
        beat = window_beat_features(inputs, 2_000_000, [(0, 500_000, 4)])
        self.assertFalse(beat[0].any())
        expected = beat_features(np.array([.005, .015], np.float32).astype(np.float64)*1e6+2_000_000,
                                 [(0, 500_000, 4)])
        np.testing.assert_array_equal(beat[1:], expected)

    def test_dropout_is_deterministic_per_request_and_epoch(self):
        draws = [timing_dropped(3, r, e, .3) for r in range(200) for e in range(5)]
        self.assertEqual(draws, [timing_dropped(3, r, e, .3) for r in range(200) for e in range(5)])
        self.assertAlmostEqual(np.mean(draws), .3, delta=.05)
        self.assertNotEqual(draws, [timing_dropped(4, r, e, .3) for r in range(200) for e in range(5)])
        self.assertFalse(any(timing_dropped(3, r, 0, 0.) for r in range(50)))
        self.assertTrue(all(timing_dropped(3, r, 0, 1.) for r in range(50)))
        self.assertEqual(normalize_charter_timing('x.json'), dict(path='x.json', sha256=None, dropout=.3))
        with self.assertRaises(ValueError):
            normalize_charter_timing(dict(path='x', dropout=2))


class ModelTests(unittest.TestCase):
    def models(self, **kw):
        torch.manual_seed(0)
        off = FullEventModel(tiny(mel_bins=8, intensity_dim=3, **kw)).eval()
        rng_off = torch.get_rng_state()
        torch.manual_seed(0)
        on = FullEventModel(tiny(mel_bins=8, intensity_dim=3, beat_features=True, **kw)).eval()
        self.assertTrue(torch.equal(rng_off, torch.get_rng_state()))
        return off, on

    def test_default_config_adds_nothing_and_flag_needs_mel(self):
        off, on = self.models()
        self.assertEqual(set(on.state_dict()) - set(off.state_dict()),
                         {'beat_projection.weight', 'beat_projection.bias'})
        for key, value in off.state_dict().items():
            self.assertTrue(torch.equal(value, on.state_dict()[key]), key)
        self.assertFalse(on.beat_projection.weight.any() or on.beat_projection.bias.any())
        with self.assertRaises(ValueError):
            FullEventConfig(beat_features=True)

    def test_zero_weights_with_timing_present_equal_flag_off_bitwise(self):
        off, on = self.models(onset_head=True)
        inp = make_inputs(off.c, batch=2)
        seq = make_sequence(batch=2)
        g = torch.Generator().manual_seed(1)
        timed = dict(inp, beat=torch.randn(2, inp['mel'].shape[1], BEAT_FEATURE_DIM, generator=g))
        with torch.no_grad():
            reference = off(inp, seq)
            for candidate in (timed, inp, dict(timed, beat_valid=torch.ones(2, inp['mel'].shape[1], dtype=torch.bool))):
                self.assertTrue(torch.equal(on(candidate, seq), reference))
            memory_off, memory_on = off.encode(inp), on.encode(timed)
            self.assertTrue(torch.equal(memory_off['values'], memory_on['values']))
            self.assertTrue(torch.equal(memory_off['onset_logits'], memory_on['onset_logits']))

    def test_trained_projection_makes_timing_a_distinct_input_and_gets_gradients(self):
        _, on = self.models()
        on.train()
        inp = make_inputs(on.c)
        seq = make_sequence()
        beat = torch.zeros(1, inp['mel'].shape[1], BEAT_FEATURE_DIM)
        beat[..., -1] = 1
        logits = on(dict(inp, beat=beat), seq)
        loss = logits.float().square().mean()
        loss.backward()
        self.assertGreater(float(on.beat_projection.weight.grad.abs().sum()), 0)
        with torch.no_grad():
            on.beat_projection.weight.normal_()
            on.eval()
            self.assertFalse(torch.equal(on(dict(inp, beat=beat), seq), on(inp, seq)))
            # Absent (missing key) == explicit all-zero features.
            self.assertTrue(torch.equal(on(inp, seq), on(dict(inp, beat=torch.zeros_like(beat)), seq)))
        with self.assertRaises(ValueError):
            on(dict(inp, beat=beat[:, :-1]), seq)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = write_table(self.tmp.name, {
            'phira:1': dict(valid=True, status='valid', segments=[[250_000., 500_000., 4], [20_000_000., 400_000., 3]]),
            'mania:2': dict(valid=False, status='timing_invalid', segments=[[0., 500_000., 4]])})

    def tearDown(self):
        self.tmp.cleanup()

    def dataset(self, timing=True, dropout=.3, seed=3):
        panel = ChartPanel()
        exposure = build_exposure(list(range(len(panel.records))), steps=30, seed=seed)
        settings = dict(path=str(self.path), dropout=dropout) if timing else None
        return CounterfactualExposureDataset(panel, exposure, microbatch=4, mel_bins=2,
                                             charter_timing=settings), exposure

    def sample(self, dataset, step, slot, index, timing=True):
        key = step.get('occurrences', step['epochs'])[slot] if timing else None
        return dataset.sample(index, step['rates'][slot], edit_seed=step['edit_seeds'][slot],
                              curve_specified=step['curve_specified'][slot],
                              specified_fields=step['specified_condition_fields'][slot], timing_key=key)

    def test_loader_attaches_the_window_slice_of_the_charts_own_timing(self):
        dataset, exposure = self.dataset()
        table = ChartTimingTable.load(self.path)
        seen = set()
        for step, slot, index in _slots(exposure):
            row, audit = self.sample(dataset, step, slot, index)
            record = dataset.panel.records[index]
            info = audit['charter_timing']
            inputs = row['inputs']
            self.assertEqual(inputs['beat'].shape, (len(inputs['mel_times']), BEAT_FEATURE_DIM))
            dropped = timing_dropped(exposure['seed'], index, step['epochs'][slot], .3)
            self.assertEqual(info['dropped'], dropped)
            if record['chart_id'] == 'mania:2':
                self.assertEqual((info['present'], info['reason']), (False, 'timing_invalid'))
                self.assertFalse(inputs['beat'].any())
            elif dropped:
                self.assertEqual((info['present'], info['reason']), (False, 'dropout'))
                self.assertFalse(inputs['beat'].any())
            else:
                self.assertTrue(info['present'])
                start = record['request']['start_us']
                expected = beat_features(np.asarray(inputs['mel_times'], np.float64)*1e6+start,
                                         table.segments('phira:1'))
                np.testing.assert_array_equal(inputs['beat'], expected)
                seen.add(start >= 20_000_000)
        self.assertEqual(seen, {False, True})  # windows on both sides of the BPM change

    def test_default_path_is_byte_identical_and_plan_settings_unchanged(self):
        base, _ = self.dataset(timing=False)
        timed, _ = self.dataset(timing=True, dropout=1.)
        plain = [base[i] for i in range(4)]
        for a, b in zip(plain, [timed[i] for i in range(4)]):
            for (_, ia, sa), (_, ib, sb) in zip(a['batches'], b['batches']):
                self.assertNotIn('beat', ia)
                self.assertFalse(ib['beat'].any())  # dropout 1: absent state everywhere
                for key in ia:
                    self.assertTrue(torch.equal(ia[key], ib[key]), key)
        self.assertEqual(_digest(plain), _digest([base[i] for i in range(4)]))
        plan = dict(config=dict(mel_bins=2))
        self.assertNotIn('charter_timing', plan_loader_settings(plan))
        with self.assertRaises(ValueError):
            plan_loader_settings(dict(plan, charter_timing=dict(path='x')))
        got = plan_loader_settings(dict(config=dict(mel_bins=2, beat_features=True), charter_timing=dict(path='x')))
        self.assertEqual(got['charter_timing'], dict(path='x', sha256=None, dropout=.3))

    def test_collate_and_fixed_shapes_pad_beat_with_mel(self):
        dataset, exposure = self.dataset()
        step = exposure['steps'][0]
        rows = [self.sample(dataset, step, s, i)[0] for s, i in enumerate(step['indices'][:2])]
        mel = len(rows[0]['inputs']['mel'])
        inputs, _ = collate(rows, input_shapes=dict(mel=mel+7))
        self.assertEqual(inputs['beat'].shape, (2, mel+7, BEAT_FEATURE_DIM))
        self.assertFalse(inputs['beat'][:, mel:].any())
        padded = pad_input_shapes(collate(rows)[0], dict(mel=mel+5))
        self.assertEqual(padded['beat'].shape[1], mel+5)
        self.assertTrue(torch.equal(padded['beat'][:, :mel], inputs['beat'][:, :mel]))

    def test_evaluation_base_inputs_with_and_without_timing(self):
        from beatfcn.train import audio_dependence as A
        panel = ChartPanel()
        panel.features = None
        table = ChartTimingTable.load(self.path)
        fake = dict(panel.row(0)['inputs'])
        with patch.object(A, 'request_inputs', lambda *a, **k: (dict(fake), None)):
            plain, request = A.base_inputs(panel, 0)
            timed, _ = A.base_inputs(panel, 0, charter_timing=table)
        self.assertNotIn('beat', plain)
        np.testing.assert_array_equal(timed['beat'], window_beat_features(fake, request.start_us,
                                                                          table.segments('phira:1')))
        self.assertTrue(timed['beat'][:, -1].all())


    def test_evaluation_table_only_with_plan_key_and_model_flag(self):
        from beatfcn.train.charter_timing import evaluation_table
        on, off = SimpleNamespace(c=SimpleNamespace(beat_features=True)), SimpleNamespace(c=SimpleNamespace())
        plan = dict(charter_timing=dict(path=str(self.path)))
        self.assertIsNone(evaluation_table({}, on))
        self.assertIsNone(evaluation_table(plan, off))
        self.assertIsNone(evaluation_table(plan, on, choice='without'))
        table = evaluation_table(plan, on)
        self.assertIsNotNone(table.segments('phira:1'))
        self.assertIsNone(table.segments('mania:2'))
        self.assertIs(evaluation_table(plan, on), table)


class RolloutTests(unittest.TestCase):
    def inputs(self, features, audio, request, controls, *, sources=()):
        mel = np.arange(-50, 850)*.01+.005
        return dict(local=np.zeros((3, 1, 8, 1024), np.float32), mode=np.asarray(1), controls=controls,
                    reference=np.zeros((1, 16), np.float32), mel=np.zeros((len(mel), 2), np.float32),
                    mel_times=mel.astype(np.float32), mel_valid=np.ones(len(mel), bool)), []

    def run_rollout(self, **kw):
        from beatfcn.model.local_song_rollout import generate_complete_songs
        seen = []

        def predict(model, inputs, reqs, cap, **_):
            seen.append(({k: v.clone() for k, v in inputs.items()}, list(reqs)))
            return [dict(complete=True, tokens=[], events=[]) for _ in reqs]
        songs = [dict(chart_id='a', audio_sha256='a'*64, mode='phira', duration_us=17_000_000),
                 dict(chart_id='b', audio_sha256='b'*64, mode='phira', duration_us=9_000_000)]
        with patch('beatfcn.model.local_song_rollout.request_inputs', self.inputs):
            result = generate_complete_songs(SimpleNamespace(training=False), None, songs, device='cpu',
                                             generator=predict, **kw)
        return result, seen

    def test_timing_provider_adds_per_core_beat_and_default_has_none(self):
        _, plain = self.run_rollout()
        self.assertTrue(all('beat' not in inputs for inputs, _ in plain))
        segments = {'a': [(100_000, 500_000, 4)], 'b': None}
        result, seen = self.run_rollout(timing_provider=lambda state, request: segments[state['chart_id']])
        for inputs, requests in seen:
            for row, request in enumerate(requests):
                song = 'a' if request.duration_us == 17_000_000 else 'b'
                times = inputs['mel_times'][row].double().numpy()*1e6+request.start_us
                expected = beat_features(times, segments[song])
                np.testing.assert_array_equal(inputs['beat'][row].numpy(), expected)
        cores = {s['chart_id']: s['cores'] for s in result['songs']}
        self.assertTrue(all(c['charter_timing']['present'] for c in cores['a']))
        self.assertFalse(any(c['charter_timing']['present'] for c in cores['b']))
        self.assertEqual(len(cores['a']), 3)


class CheckpointTests(unittest.TestCase):
    def test_extend_checkpoint_round_trip_loads_strictly_and_reproduces_outputs(self):
        from dataclasses import asdict
        from scripts.extend_checkpoint import extend_checkpoint
        torch.manual_seed(0)
        config = tiny(mel_bins=8, intensity_dim=3)
        source = FullEventModel(config).eval()
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp)/'src.pt', Path(tmp)/'dst.pt'
            ema = {k: v.clone()+1 for k, v in source.state_dict().items()}
            torch.save(dict(model=source.state_dict(), config=asdict(config), vocabulary={}, optimizer={'x': 1},
                            ema=dict(model=ema, decay=.999), updates=7), src)
            info = extend_checkpoint(src, dst)
            self.assertTrue(info['ema'])
            self.assertEqual(info['dropped'], ['optimizer'])
            saved = torch.load(dst, weights_only=False)
            self.assertTrue(saved['config']['beat_features'])
            self.assertNotIn('optimizer', saved)
            self.assertEqual(saved['updates'], 7)
            model = FullEventModel(FullEventConfig(**saved['config'])).eval()
            current = model.state_dict()
            self.assertEqual(set(saved['model']), set(current))  # the runner's init_weights check
            self.assertEqual(set(saved['ema']['model']), set(current))
            model.load_state_dict(saved['model'], strict=True)
            inp, seq = make_inputs(config), make_sequence()
            beat = torch.randn(1, inp['mel'].shape[1], BEAT_FEATURE_DIM)
            with torch.no_grad():
                self.assertTrue(torch.equal(model(dict(inp, beat=beat), seq), source(inp, seq)))
            with self.assertRaises(FileExistsError):
                extend_checkpoint(src, dst)
            with self.assertRaises(ValueError):
                extend_checkpoint(dst, Path(tmp)/'again.pt')


class BuilderTests(unittest.TestCase):
    """Real local raw chart + tape target; the full-database clock shift via a receipt."""

    def setUp(self):
        from scripts.run_workflow_benchmark import RAW_ROOT, ROOT, TAPE, raw_chart_path
        self.root = ROOT
        tape = ROOT/TAPE
        if not tape.is_file():
            self.skipTest('local 580 tape not present')
        manifest = json.loads(tape.read_text())
        self.raw_root = ROOT/RAW_ROOT
        self.chart = next(c for c in manifest['charts'] if c['chart_id'] == 'osu:1042313')
        if not raw_chart_path(self.raw_root, self.chart['chart_id']).is_file():
            self.skipTest('raw chart not present')

    def test_local_chart_valid_and_receipt_shift_moves_the_grid(self):
        import hashlib
        from scripts.build_chart_timing import chart_entry
        from scripts.run_workflow_benchmark import ClockShifts
        base = chart_entry(self.root, self.raw_root, self.chart, None)
        self.assertEqual(base['status'], 'valid')
        self.assertEqual((base['clock_shift_us'], base['clock_shift_source']), (0, 'none'))
        self.assertEqual(base['meter_source'], 'osu_timing_point')
        with tempfile.TemporaryDirectory() as tmp:
            with np.load(self.root/self.chart['target_path']) as z:
                arrays = {k: z[k].copy() for k in z.files}
            arrays['onset_us'] = arrays['onset_us']+14_000
            arrays['end_us'] = arrays['end_us']+14_000
            target = Path(tmp)/'t.npz'
            np.savez(target, **arrays)
            chart = dict(self.chart, target_path=str(target), target_sha256=None, canonical_sha256='c'*64)
            key = hashlib.sha256(f"{chart['chart_id']}:{'c'*64}".encode()).hexdigest()
            receipt = Path(tmp)/'receipts'/'targets'/key/'receipt.json'
            receipt.parent.mkdir(parents=True)
            receipt.write_text(json.dumps(dict(chart_id=chart['chart_id'], canonical_sha256='c'*64,
                                               statistics=dict(model_shift_us=14_000))))
            shifted = chart_entry(Path('/'), self.raw_root, chart,
                                  ClockShifts(Path(tmp)/'receipts', {chart['chart_id']: 'c'*64}))
            self.assertEqual((shifted['status'], shifted['clock_shift_source'], shifted['clock_shift_us']),
                             ('valid', 'receipt', 14_000))
            np.testing.assert_allclose([s[0] for s in shifted['segments']],
                                       [s[0]+14_000 for s in base['segments']], atol=1e-3)
            estimated = chart_entry(Path('/'), self.raw_root, chart, None)
            self.assertEqual((estimated['status'], estimated['clock_shift_source']), ('valid', 'estimated'))


class GimmickTimingTests(unittest.TestCase):
    def test_stop_bpm_is_timing_invalid_not_a_crash(self):
        from fractions import Fraction
        from scripts import build_chart_timing as B
        def raw(bpm):
            seg = SimpleNamespace(start_sec=Fraction(0), bpm=bpm)
            return SimpleNamespace(timing=SimpleNamespace(segments=[seg], offset_sec=Fraction(0)),
                                   sidecar=None, source_format='phira')
        self.assertEqual(B.timing_segments(raw(Fraction(120)))[0], [[0.0, 500000.0, 4]])
        for bpm in (Fraction(1, 10**400), Fraction(0), Fraction(10**9)):
            with self.assertRaises(B.GimmickTiming):
                B.timing_segments(raw(bpm))
        gimmick = dict(valid=False, mode='mania_4k', source='osu', status='timing_invalid',
                       reason='gimmick timing: unusable BPM')
        cov = B.coverage({'osu:1': gimmick})
        self.assertEqual(cov['gimmick_timing'], 1)


if __name__ == '__main__':
    unittest.main()
