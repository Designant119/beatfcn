"""Benchmark v2: synthetic metric cases with expected values + plumbing on a tiny model."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.train import requirement_diagnostics as D


def ev(t_ms, kind='tap', lane=0, end_ms=None):
    t = int(t_ms*1000)
    return dict(onset_us=t, end_us=t if end_ms is None else int(end_ms*1000), type=kind, lane=lane)


REQ = dict(mode='mania_4k', start_us=0, end_us=8_000_000, duration_us=10_000_000)
# A musically plausible human window: 250 ms grid, one hold.
LANES = (0, 2, 1, 3, 3, 0, 2, 1, 0, 1, 3, 2, 2, 3, 0, 1, 1, 0, 3, 2, 0, 3, 1, 2, 3, 1, 0, 2, 1, 3)
HUMAN = [ev(250*i+125, lane=LANES[i]) for i in range(30)] + [ev(7700, 'hold', 1, 7950)]


class AccuracyMetricTests(unittest.TestCase):
    def test_perfect_copy(self):
        result = D.window_accuracy(dict(complete=True, events=HUMAN), HUMAN, REQ)
        for ms in D.TOLERANCES_MS:
            self.assertEqual(D.f1_of(result['full'][f'note_{ms}ms']), 1.)
            self.assertEqual(D.f1_of(result['full'][f'onset_{ms}ms']), 1.)
        self.assertEqual(D.f1_of(result['full']['hold_joint_50ms']), 1.)
        self.assertEqual(result['coverage_fraction'], 1.)
        # The 250 ms grid shifted by whole seconds lands on itself (onset F1 stays
        # high: timing alone needs no audio), but lanes do not, so note F1 drops.
        self.assertGreater(D.f1_of(result['null_shift']['onset_50ms']), .8)
        self.assertLess(D.f1_of(result['null_shift']['note_50ms']), .5)

    def test_empty_output_is_zero_not_undefined(self):
        result = D.window_accuracy(dict(complete=True, events=[]), HUMAN, REQ)
        self.assertEqual(D.f1_of(result['full']['note_50ms']), 0.)
        self.assertEqual(D.f1_of(result['full']['hold_joint_50ms']), 0.)

    def test_shifted_output_has_no_credit_at_20ms_but_some_at_100ms(self):
        shifted = [dict(e, onset_us=e['onset_us']+60_000, end_us=e['end_us']+60_000) for e in HUMAN]
        result = D.window_accuracy(dict(complete=True, events=shifted), HUMAN, REQ)
        self.assertEqual(D.f1_of(result['full']['note_20ms']), 0.)
        self.assertEqual(D.f1_of(result['full']['note_50ms']), 0.)
        self.assertEqual(D.f1_of(result['full']['note_100ms']), 1.)

    def test_dense_random_output_scores_like_its_own_null_shift(self):
        rng = np.random.default_rng(0)
        dense = sorted((ev(float(t), lane=int(l)) for t, l in zip(rng.uniform(0, 8000, 400), rng.integers(0, 4, 400))),
                       key=lambda e: e['onset_us'])
        result = D.window_accuracy(dict(complete=True, events=dense), HUMAN, REQ)
        real, null = D.f1_of(result['full']['note_50ms']), D.f1_of(result['null_shift']['note_50ms'])
        self.assertLess(abs(real-null), .05)
        stats = D.human_likeness([(dense, HUMAN, REQ)])
        self.assertGreater(stats['short_gap_fraction_model'], .25)
        self.assertEqual(stats['short_gap_fraction_human'], 0.)
        self.assertGreater(stats['ioi_js_nats'], .3)

    def test_prefix_and_coverage_of_truncated_output(self):
        prefix = [e for e in HUMAN if e['onset_us'] < 2_000_000]
        result = D.window_accuracy(dict(complete=False, events=prefix), HUMAN, REQ)
        self.assertAlmostEqual(result['coverage_fraction'], (prefix[-1]['onset_us']+1)/8e6)
        self.assertEqual(D.f1_of(result['prefix']['note_50ms']), 1.)
        self.assertLess(D.f1_of(result['full']['note_50ms']), .5)

    def test_hold_pooling_counts_one_sided_windows(self):
        rows = [D.match_counts([], [ev(100, 'hold', 0, 500)], 'mania_4k'),  # missed hold
                D.match_counts([], [], 'mania_4k'),                           # neither side
                D.match_counts([ev(100, 'hold', 0, 500)], [ev(100, 'hold', 0, 500)], 'mania_4k')]
        pooled = D.pool_counts(rows)['hold_joint_50ms']
        self.assertEqual((pooled['tp'], pooled['pred'], pooled['true']), (1, 1, 2))
        self.assertAlmostEqual(pooled['f1'], 2/3)

    def test_js_status_counts_empty_sides(self):
        self.assertEqual(D.js_with_status([0, 0], [0, 0]), (0., 'both_empty'))
        self.assertEqual(D.js_with_status([0, 0], [1, 0]), (D.JS_MAX_NATS, 'one_side_empty'))
        value, status = D.js_with_status([1, 0], [1, 0])
        self.assertEqual((value, status), (0., 'defined'))


class HumanLikenessAndHackTests(unittest.TestCase):
    def test_sub_30ms_gaps_are_visible(self):
        stacked = [ev(100), ev(115, lane=1), ev(130, lane=2), ev(500)]
        stats = D.human_likeness([(stacked, HUMAN, REQ)])
        self.assertAlmostEqual(stats['sub30ms_gap_fraction_model'], 2/3)
        self.assertAlmostEqual(stats['short_gap_fraction_model'], 2/3)

    def test_chords_are_not_gaps(self):
        self.assertEqual(len(D.onset_gaps_ms([ev(100, lane=0), ev(100, lane=1), ev(300)])), 1)

    def test_template_output_across_windows(self):
        template = [ev(250*i+125, lane=0) for i in range(8)]
        outputs = [([dict(e, onset_us=e['onset_us']+s, end_us=e['end_us']+s) for e in template], s, f'song{k}')
                   for k, s in enumerate((0, 8_000_000, 16_000_000))]
        result = D.template_similarity(outputs, 'mania_4k')
        self.assertEqual((result['pairs'], result['mean'], result['identical_pairs']), (3, 1., 3))
        empty = D.template_similarity([([], 0, 'a'), ([], 8_000_000, 'b')], 'mania_4k')
        self.assertEqual(empty['mean'], 1.)   # all-empty output is a perfect template
        same_song = D.template_similarity([(template, 0, 'a'), (template, 0, 'a')], 'mania_4k')
        self.assertEqual(same_song['pairs'], 0)


def fixture(windows_per_mode=3, silent_per_mode=1, span_us=8_000_000, duration_us=20_000_000):
    from tests.test_full_event_model import inputs
    config = FullEventConfig(d=32, heads=4, kv_heads=2, ffn=48, encoder_layers=1, decoder_layers=2,
                             tap_dim=16, taps=2, bands=1, dropout=0., intensity_dim=3)
    torch.manual_seed(5)
    base = {k: v[0].numpy() for k, v in inputs(config).items()
            if k not in ('controls', 'mode', 'reference', 'reference_valid', 'reference_times')}
    base.update(mel=np.zeros((31, 128), np.float32), mel_times=np.arange(31, dtype=np.float32)*.01,
                mel_valid=np.ones(31, bool))

    class Features:
        records = {}
        def window(self, audio, *_):
            result = {k: v.copy() for k, v in base.items()}
            # Audio-specific structure (not a constant offset, which layer norm removes).
            rng = np.random.default_rng(int(audio[0], 16))
            for stream in ('local', 'song', 'mel'):
                result[stream] = result[stream]+rng.normal(size=result[stream].shape).astype(np.float32)
            return result

    class Panel:
        records, charts, targets_by_index = [], [], []
        features = Features()
        def targets(self, index):
            return self.targets_by_index[index]

    panel = Panel()
    index = 0
    for mode in ('phira', 'mania_4k'):
        for k in range(windows_per_mode+silent_per_mode):
            audio = f'{index+1:x}'*64
            positive = k < windows_per_mode
            request = asdict(EventRequest(mode, 0, span_us, duration_us))
            request['copy_times'], request['known_carry'] = [], []
            lane = -1 if mode == 'phira' else 0
            truth = [RhythmEvent(250_000*i+125_000, 250_000*i+125_000, 'tap', lane if lane < 0 else i % 4)
                     for i in range(12)] + [RhythmEvent(5_000_000, 5_500_000, 'hold', lane if lane < 0 else 1)]
            truth = truth if positive else []
            panel.features.records[audio] = dict(duration_us=duration_us)
            types = {}
            for e in truth:
                types[e.type] = types.get(e.type, 0)+1
            panel.records.append(dict(index=index, chart_id=f'chart{index}', audio_sha256=audio, component=audio,
                mode=mode, split='val', request=request, chart_index=index, event_count=len(truth), types=types))
            panel.charts.append(dict(target_sha256=f'{index:064x}'))
            panel.targets_by_index.append(truth)
            index += 1
    return FullEventModel(config).eval(), panel


class PlumbingTests(unittest.TestCase):
    def setUp(self):
        self.model, self.panel = fixture()

    def test_teacher_screen_reports_per_field_gaps_for_every_window(self):
        from beatfcn.train.audio_dependence import freeze_teacher_panel, run_teacher_screen, CONDITIONS
        protocol = freeze_teacher_panel(self.panel, windows=8)
        self.assertEqual(len(protocol['windows']), 8)
        self.assertEqual(protocol, freeze_teacher_panel(self.panel, windows=8))   # deterministic
        for window in protocol['windows']:
            donor = self.panel.records[window['foreign_index']]
            self.assertNotEqual(donor['audio_sha256'], window['audio_sha256'])
            self.assertEqual(donor['mode'], window['mode'])
            self.assertIn(abs(window['roll_s']), (1., 2., 3.))
        result = run_teacher_screen(self.model, self.panel, protocol, device='cpu', batch_size=3)
        summary = result['summary']
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['measured_windows'], 8)
        for record in result['records']:
            self.assertEqual(set(record['conditions']), set(CONDITIONS))
        for condition in ('foreign', 'zero_local_mel', 'roll'):
            gap = summary['gaps'][condition]['head_time']
            self.assertIsNotNone(gap['pooled_gap_nats'])
            self.assertTrue(math.isfinite(gap['pooled_gap_nats']))
        # A model whose audio streams are disconnected has exactly zero gaps.
        deaf = FullEventModel(self.model.c).eval()
        with torch.no_grad():
            deaf.local_projection.weight.zero_(); deaf.local_projection.bias.zero_()
            deaf.song_projection.weight.zero_(); deaf.song_projection.bias.zero_()
            deaf.mel_source.zero_() if hasattr(deaf, 'mel_source') else None
        deaf_summary = run_teacher_screen(deaf, self.panel, protocol, device='cpu', batch_size=8)['summary']
        self.assertLess(abs(deaf_summary['gaps']['foreign']['head_time']['pooled_gap_nats']), 1e-5)

    def test_onset_head_is_scored_when_present(self):
        from dataclasses import fields, replace
        from beatfcn.train.audio_dependence import freeze_teacher_panel, run_teacher_screen
        if 'onset_head' not in {f.name for f in fields(FullEventConfig)}:
            self.skipTest('model has no onset head option')
        model = FullEventModel(replace(self.model.c, onset_head=True)).eval()
        result = run_teacher_screen(model, self.panel, freeze_teacher_panel(self.panel, windows=4), device='cpu')
        onset = result['summary']['onset_head_f1_50ms']
        self.assertEqual(set(onset), {'real', 'foreign', 'zero_local_mel', 'roll'})
        self.assertIn('any', onset['real']['per_type'])
        self.assertEqual(onset['real']['true'], sum(r['event_count'] > 0 and len({
            e.onset_us for e in self.panel.targets(r['index'])}) for r in result['records']))
        self.assertIsNone(run_teacher_screen(self.model, self.panel, freeze_teacher_panel(self.panel, windows=2),
                                             device='cpu')['summary']['onset_head_f1_50ms'])

    def test_generation_benchmark_batches_all_cases_and_keeps_raw_outputs(self):
        from beatfcn.train import learning_evaluation as L
        protocol = L.freeze_generation_panel(self.panel, per_mode_positive=3, per_mode_silence=1, anchors_per_mode=1)
        self.assertEqual(len(protocol['samples']), 8)
        self.assertEqual(len(protocol['anchors']), 2)
        calls = []
        original = L.generate_batch
        def counted(model, inputs, requests, max_tokens, **kwargs):
            calls.append((len(requests), max_tokens, sorted(kwargs)))
            return original(model, inputs, requests, max_tokens, **kwargs)
        L.generate_batch = counted
        try:
            result = L.run_generation_benchmark(self.model, self.panel, protocol, device='cpu',
                deadline=time.monotonic()+120, max_batch_rows=64)
        finally:
            L.generate_batch = original
        cases = 8+2*(len(L.AUDIO_CONDITIONS)+len(L.DENSITY_CONDITIONS))
        readouts = 1 if result['readouts_skipped'] else 5   # greedy only if sampling is unsupported
        self.assertEqual(result['expected_rows'], cases*readouts)
        self.assertEqual(sum(c[0] for c in calls), cases*readouts)
        self.assertEqual(len(calls), readouts if not result['readouts_skipped'] else 1)
        self.assertEqual(result['token_budget'], 2*result['longest_teacher_tokens'])
        for row in result['rows']:
            self.assertIn('tokens', row['prediction'])
            self.assertIn('events', row['prediction'])
            if not row['prediction']['complete']:
                self.assertIsNotNone(row['failure'])
        greedy = result['readouts']['greedy']['per_seed']['None']
        for category in ('accuracy', 'human_likeness', 'condition_response', 'hacks', 'audio_grounding'):
            self.assertIn('meaning', greedy[category])
        json.dumps(L.sanitize(result), allow_nan=False)

    def test_sampled_readout_uses_per_row_generators_when_supported(self):
        from beatfcn.train import learning_evaluation as L
        protocol = L.freeze_generation_panel(self.panel, per_mode_positive=2, per_mode_silence=0, anchors_per_mode=0)
        seen = []
        original, cached = L.generate_batch, dict(L._ROW_GENERATORS)

        def per_row(model, inputs, requests, max_tokens, *, deadline=None, temperature=None, top_p=None,
                    generator=None):
            if temperature is not None:
                self.assertIsInstance(generator, list)
                self.assertEqual(len(generator), len(requests))
                seen.append([g.initial_seed() for g in generator])
            return original(model, inputs, requests, max_tokens, deadline=deadline)
        L.generate_batch, L._ROW_GENERATORS['supported'] = per_row, None
        try:
            result = L.run_generation_benchmark(self.model, self.panel, protocol, device='cpu',
                deadline=time.monotonic()+60, readouts=(dict(name='sample_t1', temperature=1., seeds=(0, 1, 2)),))
        finally:
            L.generate_batch = original
            L._ROW_GENERATORS.update(cached)
        summary = result['readouts']['sample_t1']
        self.assertEqual(summary['rng_modes'], {'per_row': 12})
        self.assertEqual(len(seen), 1)
        self.assertEqual(len(set(seen[0])), 12)   # distinct, case- and seed-specific seeds
        self.assertEqual(set(summary['per_seed']), {'0', '1', '2'})
        self.assertEqual(summary['mean_sd_over_seeds']['accuracy.note_f1_50ms']['seeds'], 3)
        self.assertIn('ORACLE', summary['oracle']['label'])

    def test_guidance_readout_runs_with_unconditional_inputs(self):
        from beatfcn.train import learning_evaluation as L
        protocol = L.freeze_generation_panel(self.panel, per_mode_positive=1, per_mode_silence=0, anchors_per_mode=1)
        result = L.run_generation_benchmark(self.model, self.panel, protocol, device='cpu',
            deadline=time.monotonic()+60, readouts=(dict(name='cfg2_greedy', guidance=2.0),))
        rows = result['rows']
        self.assertTrue(rows)
        self.assertTrue(all(r['prediction']['decoding'].get('guidance') == 2.0 for r in rows
                            if not r['prediction'].get('not_started')))

    def test_real_sampler_is_reproducible_for_fixed_chunking(self):
        from beatfcn.train import learning_evaluation as L
        if not L.sampling_supported():
            self.skipTest('generate_batch has no sampling arguments')
        protocol = L.freeze_generation_panel(self.panel, per_mode_positive=2, per_mode_silence=0, anchors_per_mode=0)
        run = lambda: [r['prediction']['tokens'] for r in L.run_generation_benchmark(self.model, self.panel, protocol,
            device='cpu', deadline=time.monotonic()+60,
            readouts=(dict(name='sample_t1', temperature=1., seeds=(0, 1)),))['rows']]
        self.assertEqual(run(), run())

    def test_watchdog_is_a_failure_not_a_truncated_success(self):
        from beatfcn.train import learning_evaluation as L
        protocol = L.freeze_generation_panel(self.panel, per_mode_positive=1, per_mode_silence=0, anchors_per_mode=0)
        result = L.run_generation_benchmark(self.model, self.panel, protocol, device='cpu',
            deadline=time.monotonic()-1, readouts=(dict(name='greedy'),))
        self.assertEqual(result['complete_rows'], 0)
        self.assertEqual(set(result['failure_counts']), {'watchdog'})
        accuracy = result['readouts']['greedy']['per_seed']['None']['accuracy']
        self.assertEqual(accuracy['complete_fraction'], 0.)
        self.assertEqual(accuracy['note_f1_50ms'], 0.)

    def test_score_readout_flags_empty_template_and_silence_hacks(self):
        from beatfcn.train import learning_evaluation as L
        protocol = L.freeze_generation_panel(self.panel, per_mode_positive=3, per_mode_silence=1, anchors_per_mode=1)
        rows = []
        for case in L.build_cases(self.panel, protocol):
            sample = next(s for s in protocol['samples'] if s['index'] == case['index'])
            truth = [asdict(e) for e in self.panel.targets(case['index'])]
            prediction = dict(complete=True, events=[], tokens=[])
            row = dict(case, readout=dict(name='x'), readout_seed=None, truth=truth, request=sample['request'],
                       audio_sha256=sample['audio_sha256'], prediction=prediction, requested_curve=None)
            row['failure'] = L.classify(prediction, case['case_role'])
            row['scores'] = D.window_accuracy(prediction, truth, sample['request'])
            rows.append(row)
        summary = L.score_readout(rows)
        self.assertEqual(summary['hacks']['empty_positive_rate'], 1.)
        self.assertTrue(summary['hacks']['all_positive_empty'])
        self.assertEqual(summary['accuracy']['note_f1_50ms'], 0.)
        self.assertEqual(summary['accuracy']['complete_fraction'], 1.)  # complete, yet a failure
        self.assertEqual(summary['hacks']['silence_identical_to_music'], 2)
        self.assertEqual(summary['hacks']['foreign_identical_fraction'], 1.)
        for mode in ('phira', 'mania_4k'):
            self.assertEqual(summary['hacks']['template'][mode]['model']['mean'], 1.)


class PeriodicScreenTests(unittest.TestCase):
    def test_periodic_screen_writes_atomic_complete_output(self):
        from beatfcn.train.audio_dependence import freeze_teacher_panel
        from beatfcn.train.learning_evaluation import freeze_generation_panel
        from beatfcn.train.periodic_requirements import run_periodic_screen, benchmark_due
        from beatfcn.train.checkpoint_resume import file_sha
        model, panel = fixture()
        model.train()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'teacher.json').write_text(json.dumps(freeze_teacher_panel(panel, windows=6)))
            (root/'generation.json').write_text(json.dumps(freeze_generation_panel(
                panel, per_mode_positive=2, per_mode_silence=1, anchors_per_mode=1)))
            plan = dict(benchmark_v2=dict(every_optimizer_updates=10, max_wall_s=120,
                teacher_protocol='teacher.json', teacher_protocol_sha256=file_sha(root/'teacher.json'),
                generation_protocol='generation.json', generation_protocol_sha256=file_sha(root/'generation.json'),
                history_dir='bench', latest_output='bench/latest.json',
                readouts=[dict(name='sample_t1', temperature=1., seeds=[0, 1]), dict(name='greedy')]))
            self.assertFalse(benchmark_due(5, plan['benchmark_v2']))
            self.assertTrue(benchmark_due(20, plan['benchmark_v2']))
            state = torch.random.get_rng_state()
            meta = run_periodic_screen(model, panel, plan, root=root, device='cpu', update=20,
                                       checkpoint=dict(path='x.pt', sha256='ab'*32))
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
            self.assertTrue(model.training)
            value = json.loads((root/'bench/latest.json').read_text())
            self.assertEqual(value['schema'], 'benchmark-v2-periodic')
            self.assertEqual(value['optimizer_updates'], 20)
            self.assertEqual(value['checkpoint']['sha256'], 'ab'*32)
            self.assertIn('decoding', value)
            entry = value['checkpoints'][0]
            self.assertTrue(entry['teacher_screen']['summary']['complete'])
            self.assertEqual(len(entry['benchmark']['rows']), entry['benchmark']['expected_rows'])
            self.assertEqual(value['status'], 'done')
            self.assertEqual(Path(meta['path']).read_text(), (root/'bench/latest.json').read_text())
            self.assertFalse(list((root/'bench').glob('*.partial')))
            with self.assertRaises(FileExistsError):
                run_periodic_screen(model, panel, plan, root=root, device='cpu', update=20,
                                    checkpoint=dict(path='x.pt', sha256='ab'*32))


class CheckpointScreenTests(unittest.TestCase):
    def test_v1_plan_entrypoint_uses_batched_v2_engine(self):
        from beatfcn.train.checkpoint_resume import file_sha
        from beatfcn.train import learning_evaluation as L
        model, panel = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = L.freeze_generation_panel(panel, per_mode_positive=2, per_mode_silence=1, anchors_per_mode=0)
            protocol = dict(schema='learning-watch-fixed-panel-v1', tape_sha256='t'*64,
                            samples=protocol['samples'])   # v1 shape: samples only
            (root/'panel.json').write_text(json.dumps(protocol))
            torch.save(dict(model=model.state_dict(), config=asdict(model.c), updates=32,
                            vocabulary=asdict(model.vocab)), root/'step.pt')
            (root/'plan.json').write_text('{}')
            checkpoint = dict(path='step.pt', sha256=file_sha(root/'step.pt'), exposure_step=32)
            plan = dict(tape='tape', inputs_sha256=dict(tape='t'*64), accum_steps=16, resume_checkpoint=checkpoint,
                learning_eval=dict(protocol='panel.json', protocol_sha256=file_sha(root/'panel.json'),
                    output='eval.json', plan_path='plan.json', checkpoints=[checkpoint], max_wall_s=120,
                    checkpoint_wall_s=100, per_case_s=12, max_tokens=2048,
                    readouts=[dict(name='sample_t1', temperature=1., seeds=[0, 1]), dict(name='greedy')]))
            model.train()
            meta = L.run_checkpoint_screen(model, panel, plan, root=root, device='cpu')
            self.assertTrue(model.training)
            value = json.loads((root/'eval.json').read_text())
        self.assertEqual(value['schema'], 'learning-watch-evaluation-v2')
        self.assertEqual(value['decoding']['ignored_v1_settings'], dict(per_case_s=12, max_tokens=2048))
        entry = value['checkpoints'][0]
        self.assertEqual(entry['optimizer_updates'], 2)
        self.assertEqual(len(entry['records']), 6)   # legacy rows: one per frozen sample
        self.assertEqual(len(entry['benchmark']['anchors']), 4)   # derived: two per mode
        self.assertEqual(entry['summary']['expected'], 6)
        self.assertIn('silence_false_events_per_s', entry['summary'])
        self.assertEqual(meta['checkpoints'], 1)


class DashboardV2Tests(unittest.TestCase):
    def test_dashboard_reads_v2_categories_and_old_schema(self):
        from scripts.monitor_600m import State
        from scripts.monitor_600m_dashboard import snapshot, render_text
        model, panel = fixture()
        from beatfcn.train.audio_dependence import freeze_teacher_panel, run_teacher_screen
        from beatfcn.train import learning_evaluation as L
        protocol = L.freeze_generation_panel(panel, per_mode_positive=2, per_mode_silence=1, anchors_per_mode=1)
        records, benchmark = L.evaluate_model(model, panel, protocol, device='cpu', deadline=time.monotonic()+120,
                                              readouts=(dict(name='greedy'),))
        teacher = run_teacher_screen(model, panel, freeze_teacher_panel(panel, windows=4), device='cpu')
        checkpoint = lambda opt: L.sanitize(dict(optimizer_updates=opt, sha256=str(opt), records=records,
            summary=L.summarize(records, len(protocol['samples'])), benchmark=benchmark, teacher_screen=teacher))
        state = State()
        state.evaluation = dict(schema='learning-watch-evaluation-v2', protocol_sha256='fixed',
                                checkpoints=[checkpoint(1), checkpoint(2)])
        view = snapshot(state, dict(quality_eval_protocol=dict(panel_sha256='fixed')))
        categories = {c['key'] for c in view['benchmark'] if c['key'] != '_identity'}
        self.assertEqual(categories, {'accuracy', 'human_likeness', 'condition_response', 'hacks', 'audio_grounding'})
        for category in (c for c in view['benchmark'] if 'metrics' in c):
            for m in category['metrics']:
                self.assertIn(m['direction'], (-1, 0, 1))
        text = render_text(view)
        self.assertIn('人类相似', text)
        self.assertIn('音频依赖', text)
        # Old-schema rows still render (no benchmark block).
        state.evaluation = dict(protocol_sha256='fixed', checkpoints=[dict(optimizer_updates=1, sha256='1',
            records=[dict(case_role='positive', teacher_joint_ce=1., prediction=dict(complete=True, events=[]))],
            summary=dict(expected=1, hold_joint_f1_50ms=0., silence_false_events=0, per_type={}))])
        view = snapshot(state, dict(quality_eval_protocol=dict(panel_sha256='fixed')))
        self.assertEqual(view['benchmark'], [])
        render_text(view)


if __name__ == '__main__':
    unittest.main()
