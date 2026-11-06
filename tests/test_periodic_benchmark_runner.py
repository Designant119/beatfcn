"""Runner hook for the periodic benchmark v2 screen (tiny model, no DB/GPU)."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from beatfcn.train.ema import ParameterEMA
from beatfcn.train.periodic_requirements import periodic_benchmark_step


def _drift(model, ema, steps=3):
    for _ in range(steps):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * .1)
        ema.update()


class PeriodicBenchmarkStepTests(unittest.TestCase):
    def setUp(self):
        from tests.test_benchmark_v2 import fixture
        self.model, self.panel = fixture()
        self.model.train()
        self.ema = ParameterEMA(self.model, .5)
        _drift(self.model, self.ema)

    def test_not_due_is_a_noop(self):
        report = dict(optimizer_updates=7, updates=14)
        plan = dict(benchmark_v2=dict(every_optimizer_updates=5))
        self.assertIsNone(periodic_benchmark_step(self.model, self.panel, plan, report, root='.',
                                                  device='cpu', ema=self.ema, screen=self.fail))
        self.assertNotIn('benchmark_v2_evaluations', report)

    def fail(self, *args, **kwargs):
        raise AssertionError('must not run')

    def test_uses_ema_weights_restores_state_and_names_the_weights(self):
        live = [p.detach().clone() for p in self.model.parameters()]
        seen = {}

        def screen(model, panel, plan, *, root, device, update, checkpoint):
            seen.update(training=model.training, checkpoint=checkpoint, update=update,
                        first=next(model.parameters()).detach().clone())
            torch.rand(5)  # consumes the global RNG; must be restored
            model.eval()
            return dict(path='x.json', complete=True)

        report = dict(optimizer_updates=10, updates=20, checkpoint='ckpt/step-0020.pt', checkpoint_sha256='cd'*32)
        state = torch.get_rng_state()
        entry = periodic_benchmark_step(self.model, self.panel, dict(benchmark_v2=dict(every_optimizer_updates=5),
                                        ema=dict(decay=.5)), report, root='.', device='cpu', ema=self.ema,
                                        screen=screen)
        self.assertEqual(entry['status'], 'done')
        self.assertEqual(entry['weights'], 'ema')
        self.assertEqual(entry['ema_updates'], 3)
        torch.testing.assert_close(seen['first'], self.ema.shadow[0])
        self.assertEqual(seen['checkpoint']['sha256'], 'cd'*32)
        self.assertEqual(seen['checkpoint']['weights'], 'ema')
        self.assertTrue(self.model.training)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for p, before in zip(self.model.parameters(), live):
            self.assertTrue(torch.equal(p, before))
        self.assertEqual(report['benchmark_v2_evaluations'], [entry])

    def test_raw_weights_without_ema_and_stale_checkpoint_is_not_claimed(self):
        live = next(self.model.parameters()).detach().clone()
        seen = {}

        def screen(model, panel, plan, *, root, device, update, checkpoint):
            seen.update(first=next(model.parameters()).detach().clone(), checkpoint=checkpoint)
            return dict(path='y.json')

        report = dict(optimizer_updates=5, updates=10, checkpoint='ckpt/step-0008.pt', checkpoint_sha256='ef'*32)
        entry = periodic_benchmark_step(self.model, self.panel, dict(benchmark_v2=dict(every_optimizer_updates=5)),
                                        report, root='.', device='cpu', ema=None, screen=screen)
        self.assertEqual(entry['weights'], 'raw')
        self.assertNotIn('sha256', seen['checkpoint'])
        self.assertEqual(seen['checkpoint']['weights'], 'raw')
        self.assertTrue(torch.equal(seen['first'], live))

    def test_failure_is_recorded_not_raised_and_state_restored(self):
        def screen(model, *args, **kwargs):
            model.eval()
            with torch.no_grad():
                next(model.parameters()).add_(1.)  # inside EMA swap: undone by restore
            raise RuntimeError('boom')

        live = next(self.model.parameters()).detach().clone()
        report = dict(optimizer_updates=5, updates=10)
        entry = periodic_benchmark_step(self.model, self.panel, dict(benchmark_v2=dict(every_optimizer_updates=5)),
                                        report, root='.', device='cpu', ema=self.ema, screen=screen)
        self.assertEqual(entry['status'], 'failed')
        self.assertIn('RuntimeError: boom', entry['error'])
        self.assertTrue(self.model.training)
        self.assertTrue(torch.equal(next(self.model.parameters()), live))
        bad = periodic_benchmark_step(self.model, self.panel, dict(benchmark_v2=dict(
            every_optimizer_updates=5, weights='ema')), dict(optimizer_updates=5, updates=10), root='.',
            device='cpu', ema=None, screen=self.fail)
        self.assertEqual(bad['status'], 'failed')

    def test_real_screen_end_to_end_on_tiny_model_with_ema(self):
        from beatfcn.train.audio_dependence import freeze_teacher_panel
        from beatfcn.train.checkpoint_resume import file_sha
        from beatfcn.train.learning_evaluation import freeze_generation_panel
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'teacher.json').write_text(json.dumps(freeze_teacher_panel(self.panel, windows=4)))
            (root/'generation.json').write_text(json.dumps(freeze_generation_panel(
                self.panel, per_mode_positive=1, per_mode_silence=1, anchors_per_mode=1)))
            plan = dict(ema=dict(decay=.5), benchmark_v2=dict(every_optimizer_updates=5, max_wall_s=120,
                teacher_protocol='teacher.json', teacher_protocol_sha256=file_sha(root/'teacher.json'),
                generation_protocol='generation.json', generation_protocol_sha256=file_sha(root/'generation.json'),
                history_dir='bench', latest_output='bench/latest.json',
                readouts=[dict(name='sample_t1', temperature=1., seeds=[0, 1]), dict(name='greedy')]))
            state = torch.get_rng_state()
            report = dict(optimizer_updates=5, updates=10)
            entry = periodic_benchmark_step(self.model, self.panel, plan, report, root=root, device='cpu',
                                            ema=self.ema)
            self.assertEqual(entry['status'], 'done', entry.get('error'))
            self.assertEqual(entry['weights'], 'ema')
            self.assertTrue(self.model.training)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            value = json.loads((root/'bench/latest.json').read_text())
            self.assertIn('weights_fingerprint', value['checkpoint'])
            self.assertEqual(value['checkpoint']['weights'], 'ema')
            rows = value['checkpoints'][0]['benchmark']['rows']
            modes = {r.get('rng_mode') for r in rows if r['readout']['name'] == 'sample_t1'}
            self.assertEqual(modes, {'per_row'})


class PlanBuilderBenchmarkTests(unittest.TestCase):
    def test_defaults_match_frozen_panels_of_the_same_tape_only(self):
        import argparse
        from scripts.prepare_native_counterfactual_training import benchmark_settings
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'tape.json').write_text('{"a": 1}')
            from beatfcn.train.checkpoint_resume import file_sha
            sha = file_sha(root/'tape.json')
            for name in ('teacher-panel.json', 'generation-panel.json'):
                (root/name).write_text(json.dumps(dict(tape_sha256=sha)))
            args = argparse.Namespace(benchmark_every_updates=250, benchmark_dir=str(root),
                                      benchmark_max_wall_s=600., name='x')
            bench, reason = benchmark_settings(args, str(root/'tape.json'), ema=True)
            self.assertIsNone(reason)
            self.assertEqual(bench['every_optimizer_updates'], 250)
            self.assertEqual(bench['weights'], 'ema')
            self.assertEqual([r['name'] for r in bench['readouts']], ['sample_t1', 'greedy'])
            self.assertEqual(bench['readouts'][0]['seeds'], [0, 1, 2, 3])
            self.assertEqual(bench['readouts'][0]['temperature'], 1.)
            (root/'tape.json').write_text('{"a": 2}')
            with self.assertRaises(ValueError):
                benchmark_settings(args, str(root/'tape.json'), ema=True)
            args.benchmark_every_updates = 0
            self.assertIsNone(benchmark_settings(args, str(root/'tape.json'), ema=True)[0])


if __name__ == '__main__':
    unittest.main()
