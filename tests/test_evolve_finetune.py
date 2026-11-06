import json
import tempfile
import threading
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest, EventVocabulary, encode_events, teacher_sequence
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.train.counterfactual_density import edit_events_to_multiplier
from beatfcn.train.evolve_objective import distillation_loss, evolve_loss, field_ce_sums, token_weight_table
from beatfcn.train.training_objective import exposure_group_loss
from scripts.evolve_finetune import Progress, ensure_exposure, finetune, freeze_encoders, layer_dropped, layerdrop_keep, loader_workers, load_recipe
from tests.test_alternative_context import SiblingPanel
from tests.test_full_event_model import inputs

LOSS = dict(type_weight=1., z_weight=.0001, first_token_weight=1., onset_weight=.1)
PLANS = Path('/Users/yukoval/yukoval-projects/beatfcn/reports/evolve-20261104/plans')


def tiny_config(**kw):
    return FullEventConfig(**{**dict(d=16, heads=2, kv_heads=1, ffn=32, encoder_layers=1, decoder_layers=4, tap_dim=4,
                                     taps=2, bands=1, band_slots=1, dropout=0., mel_bins=2, intensity_dim=3,
                                     qk_norm=True), **kw})


class Run:
    """A candidate spec plus an r5-shaped base plan, on the tiny fake panel."""

    def __init__(self, directory, **spec):
        self.dir = Path(directory)
        self.panel = SiblingPanel()
        torch.manual_seed(1)
        self.config = spec.pop('config', tiny_config())
        self.init = self.dir/'init.pt'
        torch.save(dict(model=FullEventModel(self.config).state_dict(), config=asdict(self.config), vocabulary={}), self.init)
        (self.dir/'tags.json').write_text(json.dumps(dict(tags={str(i): r['mode'] for i, r in enumerate(self.panel.records)})))
        (self.dir/'base.json').write_text(json.dumps(dict(
            accum_steps=4, physical_microbatch_requests=8, seed=9, cpu_threads=1, loss=LOSS,
            counterfactual=dict(treatment_mix=None, condition_dropout=None), mode_weights=dict(mania_4k=1., phira=2.),
            optimizer=dict(weight_decay=.1, betas=[.9, .95]), exposure_song_chunk=1, counterfactual_cell_us=100_000,
            counterfactual_curve_cell_us=2_000_000)))
        self.spec = dict(id='t', base_plan=str(self.dir/'base.json'), init=str(self.init), out_dir=str(self.dir/'out'),
                         learning_rate=3e-3, schedule=dict(horizon_updates=12, warmup_updates=2, minimum_lr=5e-4),
                         ema=dict(decay=.9), light_checkpoint_every=100, light_checkpoint_keep=2,
                         exposure=dict(path=str(self.dir/'exposure.json'), seed=11, subdivision_tags=str(self.dir/'tags.json'),
                                       unconditional_native=True), **spec)

    def __call__(self, updates, **kw):
        return finetune(self.spec, updates=updates, device='cpu', panel=self.panel, progress_every_s=.05, log_every=1, **kw)

    def progress(self):
        return json.loads((Path(self.spec['out_dir'])/'progress.json').read_text())

    def checkpoint(self, updates):
        return torch.load(Path(self.spec['out_dir'])/f'step-{updates:06d}.pt', weights_only=False)


class FineTuneRunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.quiet = patch('builtins.print')
        self.quiet.start()
        self.addCleanup(self.quiet.stop)

    def test_run_writes_progress_and_a_light_checkpoint_loadable_like_an_export(self):
        run = Run(self.tmp.name)
        self.assertEqual(run(3), 'done')
        progress = run.progress()
        self.assertEqual((progress['phase'], progress['updates'], progress['target_updates']), ('done', 3, 3))
        self.assertGreater(progress['rate_updates_per_s'], 0)
        self.assertEqual(progress['eta_s'], 0)
        self.assertIn('head_block', progress['loss_per_field'])
        self.assertGreater(progress['loss'], 0)
        saved = run.checkpoint(3)
        self.assertNotIn('optimizer', saved)
        self.assertEqual((saved['updates'], saved['weights'], saved['ema']['updates']), (3, 'ema', 3))
        model = FullEventModel(FullEventConfig(**saved['config']))
        model.load_state_dict(saved['model'], strict=True)
        self.assertFalse(torch.equal(saved['model']['output.weight'], saved['raw']['output.weight']))

    def test_resume_with_optimizer_continues_the_identical_trajectory(self):
        straight = Run(self.tmp.name)
        straight(4)
        expected = straight.checkpoint(4)
        with tempfile.TemporaryDirectory() as other:
            split = Run(other)
            split(2, save_optimizer=True)
            self.assertIn('optimizer', split.checkpoint(2))
            self.assertEqual(split(4, resume=True), 'done')
            actual = split.checkpoint(4)
        for key, value in expected['raw'].items():
            torch.testing.assert_close(actual['raw'][key], value, atol=1e-6, rtol=1e-5)
        for key, value in expected['model'].items():
            torch.testing.assert_close(actual['model'][key], value, atol=1e-6, rtol=1e-5)
        self.assertEqual(actual['ema']['updates'], 4)

    def test_resume_without_optimizer_rewarms_and_guards_overwrites(self):
        run = Run(self.tmp.name)
        run(2)
        with self.assertRaises(FileExistsError):
            run(4)
        run(3, resume=True)
        self.assertLess(run.progress()['lr'], 3e-3/2+1e-9)
        self.assertEqual(run.checkpoint(3)['updates'], 3)

    def test_resume_needs_a_checkpoint_and_updates_stay_in_the_horizon(self):
        run = Run(self.tmp.name)
        with self.assertRaises(FileNotFoundError):
            run(2, resume=True)
        with self.assertRaises(ValueError):
            run(13)

    def test_deadline_stops_cleanly(self):
        run = Run(self.tmp.name)
        self.assertEqual(run(3, deadline_unix=time.time()-1), 'deadline')
        progress = run.progress()
        self.assertEqual((progress['phase'], progress['updates']), ('deadline', 0))
        self.assertEqual(list(Path(run.spec['out_dir']).glob('step-*.pt')), [])

    def test_light_checkpoints_are_periodic_and_pruned(self):
        run = Run(self.tmp.name)
        run.spec.update(light_checkpoint_every=1, light_checkpoint_keep=2)
        run(4)
        self.assertEqual(sorted(p.name for p in Path(run.spec['out_dir']).glob('step-*.pt')),
                         ['step-000003.pt', 'step-000004.pt'])

    def test_non_finite_guard_fails_the_run_and_records_it(self):
        run = Run(self.tmp.name)
        saved = torch.load(run.init, weights_only=False)
        saved['model']['output.weight'][0, 0] = float('nan')
        torch.save(saved, run.init)
        with self.assertRaises(RuntimeError):
            run(2)
        progress = run.progress()
        self.assertEqual(progress['phase'], 'failed')
        self.assertIn('RuntimeError', progress['error'])

    def test_freeze_encoders_leaves_encoder_weights_untouched(self):
        run = Run(self.tmp.name, freeze_encoders=True)
        run(2)
        init = torch.load(run.init, weights_only=False)['model']
        raw = run.checkpoint(2)['raw']
        self.assertTrue(torch.equal(raw['encoders.0.0.ffn.up.weight'], init['encoders.0.0.ffn.up.weight']))
        self.assertTrue(torch.equal(raw['song_projection.weight'], init['song_projection.weight']))
        self.assertFalse(torch.equal(raw['decoder.0.ffn.up.weight'], init['decoder.0.ffn.up.weight']))
        self.assertFalse(torch.equal(raw['output.weight'], init['output.weight']))

    def test_layerdrop_and_kd_and_kind_weights_run_together(self):
        run = Run(self.tmp.name, layerdrop=.5, kd=dict(teacher='', alpha=.5, temperature=2.),
                  kind_class_weights=dict(drag=2.), field_weights=dict(kind=2.))
        run.spec['kd']['teacher'] = str(run.init)
        self.assertEqual(run(2), 'done')
        saved = run.checkpoint(2)
        self.assertEqual(sum(k.startswith('decoder.') and k.endswith('norm1.weight') for k in saved['raw']), 4)

    def test_grouped_microbatches_cover_every_exposure_step(self):
        run = Run(self.tmp.name)
        recipe = load_recipe(run.spec, asdict(run.config))
        exposure = ensure_exposure(run.spec, recipe, run.panel)
        self.assertEqual(len(exposure['steps']), 48)
        self.assertEqual(exposure['mode_mix']['exposure_slots']['phira']/exposure['mode_mix']['train_requests']['phira'] >=
                         exposure['mode_mix']['exposure_slots']['mania_4k']/exposure['mode_mix']['train_requests']['mania_4k'], True)
        first = json.loads(Path(run.spec['exposure']['path']).read_text())
        self.assertEqual(first['steps'][0], exposure['steps'][0])


class ExposureContractTest(unittest.TestCase):
    def test_exposure_skips_songs_without_features_and_rejects_a_stale_file(self):
        with tempfile.TemporaryDirectory() as d:
            run = Run(d)
            recipe = load_recipe(run.spec, asdict(run.config))
            run.panel.missing_audio = {run.panel.records[-1]['audio_sha256']}
            exposure = ensure_exposure(run.spec, recipe, run.panel)
            used = {run.panel.records[i]['audio_sha256'] for s in exposure['steps'] for i in s['indices']}
            self.assertFalse(used & run.panel.missing_audio)
            self.assertEqual(exposure['excluded_missing_feature_songs'], 1)
            run.panel.missing_audio = set(list(used)[:1])
            with self.assertRaises(ValueError):
                ensure_exposure(run.spec, recipe, run.panel)

    def test_base_plan_identity_is_checked(self):
        with tempfile.TemporaryDirectory() as d:
            run = Run(d, base_plan_sha256='0'*64)
            with self.assertRaises(ValueError):
                load_recipe(run.spec, asdict(run.config))

    def test_worker_count_is_capped_by_the_cgroup_quota(self):
        with patch('pathlib.Path.read_text', return_value='800000 100000\n'):
            self.assertEqual(loader_workers(28), 6)
        with patch('pathlib.Path.read_text', return_value='max 100000\n'):
            self.assertEqual(loader_workers(28), 28)
        with patch('pathlib.Path.read_text', side_effect=OSError):
            self.assertEqual(loader_workers(8), 8)


class ProgressTest(unittest.TestCase):
    def test_thread_keeps_writing_while_the_loop_is_stuck_and_exits_past_the_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'progress.json'
            exited = threading.Event()
            with patch('os._exit', side_effect=lambda code: exited.set()), patch('builtins.print'):
                progress = Progress(path, every_s=.01, deadline_unix=time.time()-5, stuck_grace_s=0.)
                progress.update(phase='train', updates=7, last_update_unix=time.time()-100)
                progress.start()
                self.assertTrue(exited.wait(5))
                progress.stop.set()
            state = json.loads(path.read_text())
            self.assertEqual(state['phase'], 'stuck_past_deadline')
            self.assertGreater(state['since_last_update_s'], 90)

    def test_writes_are_valid_json_at_every_read(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'progress.json'
            progress = Progress(path, every_s=.001, deadline_unix=None)
            progress.start()
            for _ in range(50):
                json.loads(path.read_text())
            progress.close(phase='done')
            self.assertEqual(json.loads(path.read_text())['phase'], 'done')


def tiny_batch(config, requests):
    rows = []
    for events in requests:
        request = EventRequest('phira', 0, 8_000_000, 10_000_000)
        rows.append(teacher_sequence(encode_events(events, request), request))
    length = max(len(r['targets']) for r in rows)

    def pad(key, fill):
        out = []
        for r in rows:
            value = torch.as_tensor(r[key])
            extra = torch.full((length-len(value), *value.shape[1:]), fill, dtype=value.dtype)
            out.append(torch.cat((value, extra)))
        return torch.stack(out)
    sequence = dict(tokens=pad('tokens', 0), fields=pad('fields', 0), times=pad('times', 0.), targets=pad('targets', 0),
                    allowed=pad('allowed', False), valid=pad('valid', False))
    sequence['allowed'][..., 0] |= ~sequence['valid']
    return sequence


class ObjectiveTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        tap, drag = RhythmEvent(1000, 1000, 'tap'), RhythmEvent(2_000_000, 2_000_000, 'drag')
        self.config = tiny_config(mel_bins=0, intensity_dim=0, tap_dim=16, decoder_layers=3)
        self.sequence = tiny_batch(self.config, [[tap, drag]]*8)
        self.vocab = EventVocabulary()
        self.logits = torch.randn(8, self.sequence['targets'].shape[1], self.vocab.size, requires_grad=True)

    def test_off_is_exactly_the_r5_objective(self):
        settings = dict(type_weight=1., z_weight=.0001, first_token_weight=1.)
        self.assertEqual(float(evolve_loss(self.logits, self.sequence, settings)),
                         float(exposure_group_loss(self.logits, self.sequence, settings)))
        self.assertIsNone(token_weight_table(self.vocab))

    def test_unit_weights_change_nothing_and_a_kind_class_weight_only_moves_that_class(self):
        settings = dict(type_weight=1., z_weight=0., first_token_weight=0.)
        base = evolve_loss(self.logits, self.sequence, settings)
        unit = token_weight_table(self.vocab, kind_class_weights=dict(drag=1.))
        torch.testing.assert_close(evolve_loss(self.logits, self.sequence, settings, token_weights=unit), base, atol=1e-5, rtol=1e-5)
        table = token_weight_table(self.vocab, kind_class_weights=dict(drag=3.))
        self.assertEqual(float(table[self.vocab.token('kind', 3)]), 3.)
        self.assertEqual(float(table[self.vocab.token('kind', 0)]), 1.)
        self.assertEqual(float(table[self.vocab.token('head_ms', 1)]), 1.)
        heavy = evolve_loss(self.logits, self.sequence, settings, token_weights=table)
        self.assertGreater(float(heavy), float(base))
        grad = torch.autograd.grad(heavy, self.logits)[0]
        self.assertTrue(torch.isfinite(grad).all())

    def test_field_weight_scales_the_field(self):
        table = token_weight_table(self.vocab, field_weights=dict(group=0.))
        self.assertEqual(float(table[self.vocab.token('group', 1)]), 0.)
        with self.assertRaises(ValueError):
            token_weight_table(self.vocab, field_weights=dict(nope=1.))
        with self.assertRaises(ValueError):
            token_weight_table(self.vocab, kind_class_weights=dict(nope=1.))

    def test_distillation_is_zero_for_the_teacher_and_mixes_by_alpha(self):
        same = distillation_loss(self.logits, self.logits.detach(), self.sequence, 2.)
        self.assertAlmostEqual(float(same), 0., places=4)
        other = torch.randn_like(self.logits)
        kl = distillation_loss(self.logits, other, self.sequence, 2.)
        self.assertGreater(float(kl), 0)
        settings = dict(type_weight=0., z_weight=0., first_token_weight=0.)
        task = evolve_loss(self.logits, self.sequence, settings)
        mixed = evolve_loss(self.logits, self.sequence, settings, teacher_logits=other, kd=dict(alpha=.25, temperature=2.))
        torch.testing.assert_close(mixed, .75*task+.25*kl, atol=1e-5, rtol=1e-5)
        pure = evolve_loss(self.logits, self.sequence, settings, teacher_logits=self.logits.detach(), kd=dict(alpha=1., temperature=2.))
        self.assertAlmostEqual(float(pure), 0., places=4)

    def test_field_sums_exclude_forced_tokens(self):
        total, count = field_ce_sums(self.logits, self.sequence)
        forced = (self.sequence['allowed'].sum(-1) <= 1) & self.sequence['valid']
        self.assertEqual(int(count.sum()), int(self.sequence['valid'].sum()-forced.sum()))
        self.assertTrue(torch.isfinite(total).all())


class LayerDropTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.config = tiny_config(mel_bins=0, intensity_dim=0, tap_dim=16, decoder_layers=4)
        self.model = FullEventModel(self.config).train()
        events = [RhythmEvent(1000, 1000, 'tap')]
        request = EventRequest('phira', 0, 8_000_000, 10_000_000)
        self.sequence = {k: torch.as_tensor(v)[None] for k, v in teacher_sequence(encode_events(events, request), request).items()}
        self.inputs = inputs(self.config)

    def test_keep_always_has_first_and_last_and_matches_p(self):
        rng = np.random.default_rng(0)
        draws = [layerdrop_keep(16, .2, rng) for _ in range(500)]
        self.assertTrue(all(k[0] == 0 and k[-1] == 15 for k in draws))
        self.assertAlmostEqual(np.mean([len(k) for k in draws]), 2+14*.8, delta=.3)
        self.assertEqual(layerdrop_keep(4, 0., rng), [0, 1, 2, 3])

    def test_swap_runs_the_kept_blocks_only_and_restores_the_model(self):
        names = list(self.model.state_dict())
        with layer_dropped(self.model, [0, 3]):
            self.assertEqual(len(self.model.decoder), 2)
            loss = self.model(self.inputs, self.sequence).square().mean()
        loss.backward()
        self.assertEqual(list(self.model.state_dict()), names)
        self.assertIsNotNone(self.model.decoder[0].ffn.up.weight.grad)
        self.assertIsNone(self.model.decoder[1].ffn.up.weight.grad)
        self.assertIsNone(self.model.decoder[2].cross_attn.k.weight.grad)
        self.assertIsNotNone(self.model.decoder[3].cross_attn.k.weight.grad)

    def test_all_kept_equals_the_plain_forward(self):
        self.model.eval()
        with torch.no_grad():
            plain = self.model(self.inputs, self.sequence)
            with layer_dropped(self.model, range(4)):
                swapped = self.model(self.inputs, self.sequence)
        self.assertTrue(torch.equal(plain, swapped))

    def test_freeze_encoders_keeps_the_decode_path_trainable(self):
        freeze_encoders(self.model)
        grads = {n: p.requires_grad for n, p in self.model.named_parameters()}
        self.assertFalse(grads['encoders.0.0.attn.q.weight'])
        self.assertFalse(grads['song_projection.weight'])
        self.assertTrue(grads['decoder.0.self_attn.q.weight'])
        self.assertTrue(grads['token_embedding.weight'])
        self.assertTrue(grads['output.weight'])


class AvoidSameLaneTest(unittest.TestCase):
    def jack_rate(self, events):
        lanes, times = {}, []
        for e in events:
            lanes.setdefault(e.onset_us, set()).add(e.lane)
        times = sorted(lanes)
        return np.mean([bool(lanes[a] & lanes[b]) for a, b in zip(times, times[1:])])

    def source(self):
        rng = np.random.default_rng(0)
        lanes = rng.permutation(np.tile(np.arange(4), 8))
        return [RhythmEvent(250_000*i, 250_000*i, 'tap', int(l)) for i, l in enumerate(lanes)]

    def test_off_is_unchanged_and_on_is_legal_with_far_fewer_same_key_neighbours(self):
        events = self.source()
        plain = edit_events_to_multiplier(events, 0, 8_000_000, 2., 'mania_4k', 10_000_000, seed=5)
        off = edit_events_to_multiplier(events, 0, 8_000_000, 2., 'mania_4k', 10_000_000, seed=5, avoid_same_lane=False)
        self.assertEqual(plain, off)
        on, _ = edit_events_to_multiplier(events, 0, 8_000_000, 2., 'mania_4k', 10_000_000, seed=5, avoid_same_lane=True)
        self.assertEqual(len(on), len(plain[0]))
        self.assertEqual(sorted(e.onset_us for e in on), sorted(e.onset_us for e in plain[0]))
        self.assertLess(self.jack_rate(on), self.jack_rate(plain[0]))
        self.assertLess(self.jack_rate(on), .1)
        request = EventRequest('mania_4k', 0, 8_000_000, 10_000_000)
        self.assertTrue(encode_events(on, request))

    def test_flag_reaches_the_loader_settings_only_when_set(self):
        from beatfcn.train.exposure_loader import plan_loader_settings
        plan = dict(config=dict(mel_bins=0))
        self.assertNotIn('avoid_same_lane', plan_loader_settings(plan))
        self.assertTrue(plan_loader_settings(dict(plan, avoid_same_lane=True))['avoid_same_lane'])


class AnalysisHelpersTest(unittest.TestCase):
    def test_same_key_stats_uses_the_report_d_definition(self):
        from scripts.analyze_type_mix import same_key_stats
        events = [RhythmEvent(t, t, 'tap', lane) for t, lane in
                  ((0, 0), (100_000, 0), (200_000, 1), (200_000, 2), (300_000, 2), (900_000, 2))]
        self.assertEqual(same_key_stats(events).tolist(), [1, 1, 3])

    def test_chart_table_maps_request_indices_to_charts(self):
        from scripts.analyze_type_mix import chart_table
        firsts, total = chart_table([dict(duration_us=8_000_000), dict(duration_us=8_000_001), dict(duration_us=1)])
        self.assertEqual((firsts.tolist(), total), ([0, 1, 3], 4))


@unittest.skipUnless(PLANS.is_dir(), 'plans live in the reports tree')
class PlanFileTest(unittest.TestCase):
    def test_plans_are_consistent(self):
        files = sorted(PLANS.glob('C*.json'))
        self.assertGreaterEqual(len(files), 6)
        for path in files:
            spec = json.loads(path.read_text())
            self.assertEqual(spec['id'], path.stem)
            self.assertEqual(spec['schedule']['horizon_updates'], 2400)
            self.assertEqual(spec['base_plan'], '/data/round5-level/plan.json')
            token_weight_table(EventVocabulary(), spec.get('field_weights'), spec.get('kind_class_weights'))
            if spec['surgery']:
                self.assertEqual(spec['surgery']['source'], '/data/beatfcn-r5-l322m.pt')
                self.assertNotEqual(spec['init'], spec['surgery']['source'])
            else:
                self.assertEqual(spec['init'], '/data/beatfcn-r5-l322m.pt')


if __name__ == '__main__':
    unittest.main()
