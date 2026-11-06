"""Scheduler identity and benchmark safety: grouped / continuous == legacy per (seed, song, core) on a tiny CPU model.

Run from the repo root: python -m unittest tests.test_sched_identity
"""
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import time
from unittest import mock
import unittest

import numpy as np
import torch

from beatfcn.model import full_event_generate as gen
from beatfcn.model import local_song_rollout as rollout
from beatfcn.model.full_event_codec import EventRequest, EventState, pattern_digits
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.model.local_song_rollout import generate_complete_song_samples
from scripts import bench_sched
from scripts import run_workflow_benchmark as W

SCHEDULERS = (('legacy', 1), ('legacy', 7), ('grouped', 3), ('grouped', 7), ('grouped', 40),
              ('continuous', 1), ('continuous', 3), ('continuous', 7), ('continuous', 18), ('continuous', 40))
SONGS = [dict(chart_id='m1', audio_sha256='a'*64, mode='mania_4k', duration_us=19_500_000),
         dict(chart_id='p1', audio_sha256='b'*64, mode='phira', duration_us=13_000_000),
         dict(chart_id='m2', audio_sha256='c'*64, mode='mania_4k', duration_us=9_000_000)]
PANEL = dict(songs=[dict(s, song_id=s['chart_id']) for s in SONGS], split='val')


def tiny_model(**kw):
    torch.manual_seed(2)
    config = dict(d=32, heads=4, kv_heads=2, ffn=48, encoder_layers=1, decoder_layers=2, tap_dim=16, bands=4, taps=3,
                  dropout=0.)
    config.update(kw)
    model = FullEventModel(FullEventConfig(**config)).eval()
    v = model.vocab
    bias = torch.zeros(v.size)
    bias[v.token('group', 0)] = -1.
    for code in range(1, 81):
        if 2 in pattern_digits(code):
            bias[v.token('mania', code)] += 2.5
    bias[v.token('kind', 1)] += 2.
    model.output.register_forward_hook(lambda m, a, out: out + bias.to(out.device))
    return model


def strip(results, *, drop=('batch_wall_s',)):
    """Whole samples without wall clocks."""
    return [[dict({k: v for k, v in state.items() if k != 'cores'},
                  cores=[{k: v for k, v in core.items() if k not in drop} for core in state['cores']])
             for state in sorted(result['songs'], key=lambda s: s['chart_id'])] for result in results]


class Harness(unittest.TestCase):
    seeds = [4, 5, 6, 7, 8, 9]
    token_cap = 240

    @classmethod
    def setUpClass(cls):
        cls.model = tiny_model()
        cls.features = bench_sched.SyntheticFeatures(cls.model.c, SONGS)

    def run_samples(self, scheduler, batch, *, model=None, generator_options=None, **kw):
        options = dict(generator_options or {})
        temperature = options.pop('temperature', 1.)
        options.setdefault('stats', {})
        generator = W.make_generator(temperature, options.pop('top_p', None), self.seeds[0],
                                     options.pop('field_temperature', None), options.pop('decode_options', None), **options)
        kw.setdefault('token_cap', self.token_cap)
        return generate_complete_song_samples(model or self.model, self.features, SONGS, self.seeds, device='cpu',
                                              batch_size=batch, generator=generator, scheduler=scheduler, **kw)

    def assert_all_same(self, *, drop=('batch_wall_s',), **kw):
        reference = strip(self.run_samples('legacy', 7, **kw), drop=drop)
        for scheduler, batch in SCHEDULERS:
            with self.subTest(scheduler=scheduler, batch=batch):
                self.assertEqual(strip(self.run_samples(scheduler, batch, **kw), drop=drop), reference)
        return reference


class SchedulerIdentityTests(Harness):
    def test_cold_mode_mix_and_hold_carry(self):
        reference = self.assert_all_same()
        cores = [core for sample in reference for state in sample for core in state['cores']]
        self.assertTrue(any(core['known_carry'] for core in cores), 'a hold must cross a core boundary')
        self.assertEqual({state['mode'] for sample in reference for state in sample}, {'mania_4k', 'phira'})
        self.assertTrue(all(state['complete'] for sample in reference for state in sample))

    def test_history_reference_uses_the_exact_previous_draft(self):
        reference = self.assert_all_same(history_reference=True)
        cores = [core for sample in reference for state in sample for core in state['cores']]
        used = [core for core in cores if any((row.get('source_id') or '').startswith('generated:')
                                              for row in core['reference_provenance'])]
        self.assertTrue(used, 'history_reference must reach the model')

    def test_token_cap_failures(self):
        reference = self.assert_all_same(token_cap=70)
        failed = [state for sample in reference for state in sample if state['failure'] == 'core_token_limit']
        finished = [state for sample in reference for state in sample if state['complete']]
        self.assertTrue(failed and finished, 'need both failing and finishing pairs at this cap')
        self.assertTrue(all(state['cores'][-1]['token_limit_reached'] and len(state['cores'][-1]['tokens']) == 70
                            for state in failed))

    def test_greedy_field_temperature_top_p(self):
        self.assert_all_same(generator_options=dict(temperature=None))
        self.assert_all_same(generator_options=dict(field_temperature={'kind': .3, 'mania': .5, 'head_ms': 2.}))
        self.assert_all_same(generator_options=dict(top_p=.9))

    def test_pattern_constraints_and_onset_fusion(self):
        self.assert_all_same(generator_options=dict(decode_options=dict(
            pattern_constraints=dict(max_chord=1, min_same_lane_ms=120.))))
        self.assert_all_same(model=tiny_model(onset_head=True), generator_options=dict(onset_fusion=.5))

    def test_guidance_with_some_rows_conditioned(self):
        controls = np.random.default_rng(0).standard_normal((self.model.c.controls, self.model.c.control_dim)).astype(np.float32)

        def provider(state, request):
            return dict(inputs=dict(controls=controls), record='asked') if state['chart_id'] != 'p1' else None
        self.assert_all_same(condition_provider=provider, generator_options=dict(guidance=2.))
        self.assert_all_same(condition_provider=provider)

    def test_onset_probe_is_shared_and_recorded(self):
        """Tokens are identical; the probe floats differ in the last bit between batch compositions, for legacy too:
        the encoder's fp32 reduction order depends on the padded batch, which is why bit identity is not claimed."""
        model = tiny_model(onset_head=True)
        probe = W.onset_probe(model)
        self.assert_all_same(model=model, on_core_inputs=probe, drop=('batch_wall_s', 'probe'))
        probes = {}
        for scheduler, batch in (('legacy', 7), ('continuous', 7), ('grouped', 7)):
            results = self.run_samples(scheduler, batch, model=model, on_core_inputs=probe)
            probes[scheduler] = np.array([p for result in results for state in sorted(result['songs'], key=lambda s: s['chart_id'])
                                          for core in state['cores'] for p in core['probe']['onset_probs']])
        self.assertGreater(len(probes['legacy']), 0)
        for scheduler in ('continuous', 'grouped'):
            self.assertLess(np.abs(probes[scheduler]-probes['legacy']).max(), 1e-5)

    def test_dead_end_row(self):
        class DeadEnd(EventState):
            def allowed_tokens(self):
                dead = self.request.mode == 'phira' and self.request.start_us == 8_000_000 and len(self.tokens) >= 3
                return () if dead else super().allowed_tokens()
        with mock.patch.object(gen, 'EventState', DeadEnd):
            reference = self.assert_all_same()
        failed = [state for sample in reference for state in sample if state['failure'] is not None]
        self.assertTrue(failed)
        self.assertTrue(all(state['chart_id'] == 'p1' and state['failure'] == 'core_token_limit'
                            and len(state['cores'][-1]['tokens']) == 3 for state in failed))

    def test_consensus_of_scheduled_samples(self):
        drafts = {}
        for scheduler, batch in (('legacy', 7), ('grouped', 7), ('continuous', 7)):
            results = self.run_samples(scheduler, batch)
            samples = [[W.draft_tuples(song, {s['chart_id']: s for s in result['songs']}[song['chart_id']])
                        for result in results] for song in SONGS]
            drafts[scheduler] = [W.consensus_draft(sample, song['mode'], song['duration_us'])[0]
                                 for song, sample in zip(SONGS, samples)]
        self.assertEqual(drafts['legacy'], drafts['grouped'])
        self.assertEqual(drafts['legacy'], drafts['continuous'])
        self.assertTrue(any(drafts['legacy']))

    def test_logit_difference_between_batch_and_seat_decode_is_round_off(self):
        """The same rows through one batch and through 2 refilled seats: the logit difference is the only way a token
        can flip. tests the bound on the tiny model; G1 (scripts/bench_sched.py) measures flips on the real one."""
        measured = measure_logit_difference(self.model, self.features)
        self.assertGreater(measured['logits_compared'], 500)
        self.assertEqual(measured['token_differences'], 0)
        self.assertLess(measured['max_abs_logit_difference'], 1e-4)


def measure_logit_difference(model, features, *, seats=2, cap=240, copies=8):
    """Teacher-free comparison of per-(request, position) logits. Same seeds, same requests."""
    requests, rows = [], []
    for _ in range(copies):
        for song in SONGS:
            for start in range(0, song['duration_us'], 8_000_000):
                request = EventRequest(song['mode'], start, min(start+8_000_000, song['duration_us']), song['duration_us'])
                state = dict(song, events=[], next_start_us=start)
                requests.append(request)
                rows.append(rollout._build_core_row(state, request, features, history_reference=False,
                                                    intensity_provider=None, condition_provider=None,
                                                    timing_provider=None, alternative_provider=None)['row'])
    inputs = rollout._device_rows(rows, None, 'cpu')
    generators = lambda: [torch.Generator().manual_seed(100+i) for i in range(len(requests))]  # noqa: E731
    captured = []
    handle = model.output.register_forward_hook(lambda m, a, out: captured.append(out[:, 0].clone()))
    try:
        with torch.no_grad():
            batch = gen.generate_batch(model, inputs, requests, cap, temperature=1., generator=generators())
            reference = {(row, step): logits[row] for step, logits in enumerate(captured)
                         for row in range(len(requests)) if step < len(batch[row]['tokens'])}
            decoder = gen.ContinuousDecoder(model, seats, cap, temperature=1.)
            waiting = list(range(len(requests)))
            encoded = [decoder.encode({k: v[row:row+1] for k, v in inputs.items()}, [requests[row]])[0]
                       for row in range(len(requests))]
            on_seat, tokens, worst, compared = {}, {}, 0., 0
            draws = generators()
            while waiting or on_seat:
                for seat in decoder.free_seats():
                    if waiting:
                        row = waiting.pop(0)
                        decoder.seat(seat, requests[row], encoded[row], draws[row])
                        on_seat[seat] = row
                captured.clear()
                positions = decoder.positions.copy()
                finished = decoder.step()
                for seat, row in on_seat.items():
                    worst = max(worst, (captured[0][seat]-reference[row, int(positions[seat])]).abs().max().item())
                    compared += 1
                for seat in finished:
                    tokens[on_seat.pop(seat)] = decoder.release(seat)['tokens']
    finally:
        handle.remove()
    return dict(max_abs_logit_difference=worst, logits_compared=compared,
                token_differences=sum(tokens[row] != batch[row]['tokens'] for row in range(len(requests))))


class ContinuousBehaviourTests(Harness):
    def test_encoder_dedupe_counts(self):
        unique = sum(math.ceil(song['duration_us']/8_000_000) for song in SONGS)
        for scheduler in ('grouped', 'continuous'):
            stats = {}
            self.run_samples(scheduler, 6, generator_options=dict(stats=stats))
            self.assertLess(stats['encoded_rows'], unique*len(self.seeds))
            if scheduler == 'continuous':
                self.assertEqual(stats['encoded_rows'], unique, 'one encoder row per (song, core) in cold mode')
        stats = {}
        self.run_samples('continuous', 6, history_reference=True, generator_options=dict(stats=stats))
        self.assertEqual(stats['encoded_rows'], unique*len(self.seeds), 'history_reference rows differ per seed')

    def test_songs_finish_in_order(self):
        """Earlier songs always get free seats first: with B = N no song waits behind a later one, and a song only
        starts once every sample of the previous song is decoding or done."""
        started, finished = [], []

        def order(state, seed):
            if len(state['cores']) == 1 and state['chart_id'] not in started:
                started.append(state['chart_id'])
            if state['complete'] and len(finished) < 18:
                finished.append(state['chart_id'])
        self.run_samples('continuous', len(self.seeds), on_progress=order)
        self.assertEqual(started, ['m1', 'p1', 'm2'])
        self.assertEqual(finished[:4], ['m1']*4, 'the first finishes are all from the first song')
        last = {song: len(finished)-1-finished[::-1].index(song) for song in ('m1', 'p1', 'm2')}
        self.assertLess(last['m1'], last['m2'])
        self.assertLess(last['p1'], last['m2'])

    def test_grouped_window_bounds_songs_in_flight(self):
        batches = []
        real = rollout._grouped_batch

        def spy(active, batch_size, previous_tokens):
            chosen = real(active, batch_size, previous_tokens)
            batches.append({state['chart_id'] for state in chosen})
            return chosen
        with mock.patch.object(rollout, '_grouped_batch', spy):
            self.run_samples('grouped', len(self.seeds))
        self.assertTrue(all('m2' not in batch or 'm1' not in batch for batch in batches), batches)

    def test_stats_are_consistent(self):
        for scheduler, batch in (('legacy', 7), ('grouped', 7), ('continuous', 7)):
            stats = {}
            results = self.run_samples(scheduler, batch, generator_options=dict(stats=stats))
            tokens = sum(len(core['tokens']) for result in results for state in result['songs'] for core in state['cores'])
            self.assertEqual(stats['tokens'], tokens)
            self.assertGreaterEqual(stats['billed_row_steps'], tokens)
            self.assertAlmostEqual(stats['occupancy'], tokens/stats['billed_row_steps'])
            for key in ('decode_s', 'grammar_s', 'select_s', 'encoder_s', 'steps'):
                self.assertGreater(stats[key], 0, key)

    def test_compaction_keeps_tokens_and_cuts_billing(self):
        stats = {}
        self.run_samples('continuous', 18, generator_options=dict(stats=stats))
        self.assertLess(stats['billed_row_steps'], 18*stats['steps'], 'seats must shrink when pairs run out')

    def test_seat_table_grows_for_longer_memory(self):
        decoder = gen.ContinuousDecoder(self.model, 2, 8)
        c = self.model.c
        request = EventRequest('mania_4k', 0, 8_000_000, 8_000_000)
        short = self._row(c, 3)
        longer = self._row(c, gen.MEMORY_LENGTH_STEP + 3)
        decoder.seat(0, request, decoder.encode(short, [request])[0])
        first = decoder.memory_length
        decoder.step()
        decoder.seat(1, request, decoder.encode(longer, [request])[0])
        self.assertGreater(decoder.memory_length, first)
        decoder.step()
        decoder.release(0), decoder.release(1)

    @staticmethod
    def _row(c, song_rows):
        frames = 4
        return dict(local=torch.randn(1, c.taps, frames, c.bands, c.tap_dim), local_times=torch.zeros(1, frames),
                    local_valid=torch.ones(1, frames, dtype=torch.bool), song=torch.randn(1, song_rows, c.tap_dim),
                    song_times=torch.zeros(1, song_rows), song_valid=torch.ones(1, song_rows, dtype=torch.bool),
                    reference=torch.zeros(1, 1, c.reference_dim), reference_times=torch.zeros(1, 1),
                    reference_valid=torch.zeros(1, 1, dtype=torch.bool), controls=torch.zeros(1, c.controls, c.control_dim),
                    mode=torch.tensor([0]))

    def test_autocast_is_off_by_default(self):
        request = EventRequest('mania_4k', 0, 8_000_000, 8_000_000)
        for autocast, dtype in ((None, torch.float32), ('bf16', torch.bfloat16)):
            decoder = gen.ContinuousDecoder(self.model, 1, 8, autocast=autocast)
            decoder.seat(0, request, decoder.encode(self._row(self.model.c, 3), [request])[0])
            decoder.step()
            self.assertEqual(decoder.decoders[0].cache['keys'].dtype, dtype)
        results = self.run_samples('continuous', 7, generator_options=dict(decode_options=dict(autocast='bf16')))
        self.assertTrue(all(state['complete'] for result in results for state in result['songs']))
        results = self.run_samples('legacy', 7, generator_options=dict(decode_options=dict(autocast='bf16')))
        self.assertTrue(all(state['complete'] for result in results for state in result['songs']))

    def test_compile_decode_option(self):
        reference = strip(self.run_samples('legacy', 7))
        compiled = strip(self.run_samples('continuous', 7, generator_options=dict(decode_options=dict(compile_decode=True))))
        self.assertEqual(compiled, reference)


class FakeGraphStep(gen._GraphStep):
    """_GraphStep on the CPU: replay re-runs the step into the captured output buffer. Covers the staging, per-row
    position, bucket and seat-move logic; the real capture and its eager self-check only exist on CUDA."""

    def _capture(self, length):
        self._run(length)
        output = self._run(length)
        step = self

        class Replay:
            def replay(self):
                output.copy_(step._run(length))
        self.graphs[length] = (Replay(), output)
        self.captures += 1
        return output


class GraphPathTests(Harness):
    def setUp(self):
        patches = [mock.patch.object(torch.Tensor, 'pin_memory', lambda tensor: tensor),
                   mock.patch.object(torch.cuda, 'graph_pool_handle', lambda: None),
                   mock.patch.object(gen, '_GraphStep', FakeGraphStep)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_seat_table_through_the_graph_step(self):
        class GraphDecoder(gen.ContinuousDecoder):
            def __init__(self, *args, **kw):
                super().__init__(*args, **kw)
                self.use_graph = True
        reference = strip(self.run_samples('legacy', 7))
        for batch in (3, 18):
            stats = {}
            with mock.patch.object(rollout, 'ContinuousDecoder', GraphDecoder):
                out = self.run_samples('continuous', batch, generator_options=dict(stats=stats))
            self.assertEqual(strip(out), reference)
            self.assertGreater(stats['graph_captures'], 0)

    def test_grouped_memory_geometry_is_stable(self):
        model = self.model
        c = model.c

        def memory(rows):
            inputs = ContinuousBehaviourTests._row(c, rows)
            return gen._encode(model, inputs)
        with torch.no_grad():
            first = gen._step_decoder(model, memory(5), 8, True, 'conditional', stable_memory=True)
            self.assertEqual(first.cache['memory_mask'].shape[-1] % gen.MEMORY_LENGTH_STEP, 0)
            self.assertIs(gen._step_decoder(model, memory(3), 8, True, 'conditional', stable_memory=True), first)
            longer = gen._step_decoder(model, memory(gen.MEMORY_LENGTH_STEP+1), 8, True, 'conditional', stable_memory=True)
            self.assertIsNot(longer, first)
            self.assertIs(gen._step_decoder(model, memory(3), 8, True, 'conditional', stable_memory=True), longer)
        gen._GRAPH_RUNNERS.clear()


class ResumeTests(Harness):
    seeds = [4, 5, 6]

    def uninterrupted(self, scheduler, batch):
        return strip(self.run_samples(scheduler, batch))

    def test_killed_run_resumes_to_identical_samples(self):
        for scheduler, batch in (('legacy', 7), ('grouped', 7), ('continuous', 7)):
            with self.subTest(scheduler=scheduler), tempfile.TemporaryDirectory() as folder:
                out = Path(folder)
                writer = W.ProgressWriter(out, SONGS, self.seeds)

                def killed(state, seed):
                    writer(state, seed)
                    if len(writer.written) >= 4:
                        raise KeyboardInterrupt
                with self.assertRaises(KeyboardInterrupt):
                    self.run_samples(scheduler, batch, on_progress=killed)
                writer.samples.close()
                saved = W.load_samples(out/'samples.jsonl')
                self.assertEqual(len(saved), 4)
                with open(out/'samples.jsonl', 'a') as handle:  # a torn last line
                    handle.write('{"seed": 4, "state": {"chart')
                completed = W.load_samples(out/'samples.jsonl', repair=True)
                self.assertEqual(set(completed), set(saved))
                self.assertTrue((out/'samples.jsonl').read_text().endswith('\n'))
                writer = W.ProgressWriter(out, SONGS, self.seeds, completed=completed)
                resumed = self.run_samples(scheduler, batch, completed=completed, on_progress=writer)
                writer.samples.close()
                self.assertEqual(strip(resumed), self.uninterrupted(scheduler, batch))
                self.assertEqual(len(W.load_samples(out/'samples.jsonl')), 9)
                self.assertEqual(writer.cores, writer.total)
                self.assertEqual(writer.pairs, 9)

    def test_wall_limit_pairs_are_retried_and_token_cap_pairs_kept(self):
        self.assertTrue(W.finished_pair(dict(complete=True, failure=None)))
        self.assertTrue(W.finished_pair(dict(complete=False, failure='core_token_limit')))
        self.assertTrue(W.finished_pair(dict(complete=False, failure='independent_legality')))
        self.assertFalse(W.finished_pair(dict(complete=False, failure='evaluation_wall_limit')))
        self.assertFalse(W.finished_pair(dict(complete=False, failure='generation_deadline')))

    def test_deadline_mid_run_then_resume(self):
        for scheduler, batch in (('legacy', 7), ('grouped', 7), ('continuous', 7)):
            with self.subTest(scheduler=scheduler), tempfile.TemporaryDirectory() as folder:
                out = Path(folder)
                writer = W.ProgressWriter(out, SONGS, self.seeds)
                clock = [0.]

                def ticks():
                    clock[0] += 1.
                    return clock[0]
                with mock.patch.object(rollout.time, 'monotonic', ticks), mock.patch.object(gen.time, 'monotonic', ticks):
                    partial = self.run_samples(scheduler, batch, max_wall_s=60., on_progress=writer)
                writer.samples.close()
                states = [state for result in partial for state in result['songs']]
                failures = {state['failure'] for state in states}
                self.assertTrue(failures & {'evaluation_wall_limit', 'generation_deadline'}, failures)
                saved = W.load_samples(out/'samples.jsonl')
                self.assertEqual(len(saved), 9, 'partial pairs are kept on disk too')
                completed = {k: v for k, v in saved.items() if W.finished_pair(v)}
                self.assertLess(len(completed), 9)
                resumed = self.run_samples(scheduler, batch, completed=completed)
                self.assertEqual(strip(resumed), self.uninterrupted(scheduler, batch))

    def test_sequential_path_resume(self):
        model, features = self.model, self.features
        generator = W.make_generator(1., None, 4)
        full = rollout.generate_complete_songs(model, features, SONGS, device='cpu', batch_size=3, token_cap=240,
                                               generator=generator)
        kept = {'m1': json.loads(json.dumps(next(s for s in full['songs'] if s['chart_id'] == 'm1')))}
        resumed = rollout.generate_complete_songs(model, features, SONGS, device='cpu', batch_size=3, token_cap=240,
                                                  generator=W.make_generator(1., None, 4), completed=kept)
        self.assertEqual(strip([resumed])[0][1:], strip([full])[0][1:])
        self.assertEqual(strip([resumed])[0][0]['events'], strip([full])[0][0]['events'])


class ProgressWriterTests(unittest.TestCase):
    def test_lines_snapshot_and_phases(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            song = dict(chart_id='m', duration_us=16_000_001)
            writer = W.ProgressWriter(out, [song], [0, 1], interval_s=0.)
            state = dict(chart_id='m', complete=False, failure=None, cores=[dict(tokens=[1, 2, 3])])
            writer(state, 0)
            self.assertFalse((out/'samples.jsonl').exists())
            state['cores'].append(dict(tokens=[4]))
            state['complete'] = True
            writer(state, 0)
            writer(state, 0)  # the rollout reports every state once more at the end
            self.assertEqual(len((out/'samples.jsonl').read_text().splitlines()), 1)
            snapshot = json.loads((out/'progress.json').read_text())
            self.assertEqual((snapshot['phase'], snapshot['cores_done'], snapshot['cores_total'], snapshot['tokens'],
                              snapshot['pairs_done'], snapshot['pairs_total']), ('decode', 2, 6, 4, 1, 2))
            self.assertIn('eta_s', snapshot)
            self.assertIn('tok_s', snapshot)
            writer.phase('score', total=4)
            writer.advance()
            snapshot = json.loads((out/'progress.json').read_text())
            self.assertEqual((snapshot['phase'], snapshot['done'], snapshot['total']), ('score', 1, 4))
            writer.close()
            self.assertEqual(json.loads((out/'progress.json').read_text())['phase'], 'done')
            self.assertFalse(list(out.glob('*.tmp')))

    def test_a_silent_decode_does_not_refresh_progress_but_a_silent_phase_does(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            writer = W.ProgressWriter(out, [dict(chart_id='m', duration_us=8_000_000)], [0], interval_s=.02)
            writer.write()
            first = json.loads((out/'progress.json').read_text())['at']
            time.sleep(.1)
            self.assertEqual(json.loads((out/'progress.json').read_text())['at'], first)
            writer.phase('score', total=1)
            first = json.loads((out/'progress.json').read_text())['at']
            time.sleep(.2)
            self.assertGreater(json.loads((out/'progress.json').read_text())['at'], first)
            writer.close()


class DeadlineTests(unittest.TestCase):
    def test_absolute_deadline_ignores_sample_count(self):
        now = time.monotonic()
        args = mock.Mock(process_start=now-500., max_wall_s=1000., flush_reserve_s=300.)
        self.assertAlmostEqual(W.remaining_wall_s(args), 200., delta=1.)
        args = mock.Mock(process_start=now-900., max_wall_s=1000., flush_reserve_s=300.)
        self.assertEqual(W.remaining_wall_s(args), 0.)

    def test_consensus_votes(self):
        samples = [[(1_000_000, 1_000_000, 'tap', 0)], [(1_000_000, 1_000_000, 'tap', 0)], [], [], [], []]
        self.assertEqual(W.consensus_draft(samples, 'mania_4k', 8_000_000)[0], [], 'default need = ceil(6/2)')
        self.assertEqual(len(W.consensus_draft(samples, 'mania_4k', 8_000_000, min_votes=2)[0]), 1)
        self.assertEqual(W.consensus_draft(samples, 'mania_4k', 8_000_000, min_votes=3)[0], [])


def synthetic_tape(root, songs):
    charts = []
    for song in songs:
        onsets = np.arange(500_000, song['duration_us']-500_000, 400_000)
        path = root/f"{song['chart_id']}.npz"
        np.savez(path, onset_us=onsets, end_us=onsets, kind=np.zeros(len(onsets), int), lane=np.arange(len(onsets)) % 4)
        charts.append(dict(chart_id=f"osu:{song['chart_id']}", audio_sha256=song['audio_sha256'], mode=song['mode'],
                           split='val', duration_us=song['duration_us'], component=song['chart_id'], stratum='density/low',
                           requests=[dict(event_count=len(onsets))], target_path=path.name,
                           target_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (root/'tape.json').write_text(json.dumps(dict(complete=True, failures=[], charts=charts)))


class EndToEndTests(unittest.TestCase):
    """The benchmark entry point with a tiny random model standing in for the smoke model."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        synthetic_tape(self.root, SONGS)
        model = tiny_model()
        self.features = bench_sched.SyntheticFeatures(model.c, SONGS)
        self.smoke = lambda args, panel: (model, self.features, panel['songs'], dict(path=None, sha256=None), 0, 100_000)

    def tearDown(self):
        self.folder.cleanup()

    def base(self, out, *extra):
        return ['prog', '--root', str(self.root), '--tape', 'tape.json', '--raw-root', 'raw', '--out-dir', str(out),
                '--bootstrap', '5', '--token-cap', '240', '--seed', '3', '--consensus', '4', '--batch-size', '6',
                '--human-baseline', str(self.root/'human.json'), *extra]

    def invoke(self, argv):
        (self.root/'human.json').write_text(json.dumps(dict(human_playability={})))
        with mock.patch('sys.argv', argv), mock.patch.object(W, 'local_smoke_model', self.smoke), \
                contextlib.redirect_stdout(io.StringIO()):
            W.main()

    def test_no_score_then_score_from_and_resume(self):
        out = self.root/'decode'
        self.invoke(self.base(out, '--local-smoke', '--no-score', '--scheduler', 'continuous'))
        self.assertFalse((out/'workflow-benchmark.json').exists())
        lines = (out/'samples.jsonl').read_text().splitlines()
        self.assertEqual(len(lines), 12)
        meta = json.loads((out/'run-meta.json').read_text())
        self.assertGreater(meta['decode_stats']['billed_row_steps'], 0)
        self.assertEqual(json.loads((out/'progress.json').read_text())['phase'], 'done')
        self.assertEqual(len(json.loads((out/'drafts-samples.json').read_text())), 4)
        # Laptop: no model at all.
        scored = self.root/'scored'
        self.score_from(out, scored, '--consensus-votes', '2')
        report = json.loads((scored/'workflow-benchmark.json').read_text())
        self.assertEqual(report['decoding']['scheduler'], 'continuous')
        self.assertEqual(report['decoding']['consensus']['min_votes'], 2)
        self.assertEqual(report['decode_stats']['billed_row_steps'], meta['decode_stats']['billed_row_steps'])
        self.assertEqual(json.loads((scored/'progress.json').read_text())['phase'], 'done')
        # A new run in the same directory must ask for --resume; a resume decodes nothing and scores.
        with self.assertRaises(FileExistsError):
            self.invoke(self.base(out, '--local-smoke', '--scheduler', 'continuous'))
        self.invoke(self.base(out, '--local-smoke', '--scheduler', 'continuous', '--resume'))
        self.assertEqual(len((out/'samples.jsonl').read_text().splitlines()), 12)
        direct = json.loads((out/'workflow-benchmark.json').read_text())
        self.assertEqual(direct['decode_stats']['tokens'], 0, 'a resume of a finished run decodes nothing')
        scores = lambda r: [(row['song_id'], row['min_ref_cer']) for row in r['readout_rows']['consensus_4']['draft_rows']]  # noqa: E731
        self.assertEqual(scores(direct), scores(report), 'scoring saved samples == scoring in the decode run')
        # A song with a missing sample is left out instead of being scored as insertions.
        kept = [line for line in lines if not ('"seed": 3,' in line and '"chart_id": "mania_4k:cccc' in line)]
        self.assertEqual(len(kept), len(lines)-1)
        (out/'samples.jsonl').write_text('\n'.join(kept)+'\n')
        fewer = self.root/'fewer'
        self.score_from(out, fewer)
        self.assertEqual(len(json.loads((fewer/'workflow-benchmark.json').read_text())['panel']['songs']), 2)

    def score_from(self, source, out, *extra):
        (self.root/'human.json').write_text(json.dumps(dict(human_playability={})))
        with mock.patch.object(W, 'checkpoint_model', side_effect=AssertionError), \
                mock.patch.object(W, 'local_smoke_model', side_effect=AssertionError), \
                mock.patch('sys.argv', self.base(out, '--score-from', str(source), *extra)), \
                contextlib.redirect_stdout(io.StringIO()):
            W.main()

    def test_resume_refuses_other_decoding(self):
        out = self.root/'decode'
        self.invoke(self.base(out, '--local-smoke', '--no-score'))
        with self.assertRaises(SystemExit):
            self.invoke(self.base(out, '--local-smoke', '--resume', '--no-score', '--temperature', '.5'))

    def test_past_deadline_skips_scoring_and_keeps_samples(self):
        out = self.root/'late'
        with mock.patch.object(W, 'PROCESS_START', time.monotonic()-1000.):
            self.invoke(self.base(out, '--local-smoke', '--max-wall-s', '1000', '--flush-reserve-s', '100'))
        self.assertFalse((out/'workflow-benchmark.json').exists())
        self.assertEqual({state['failure'] for state in W.load_samples(out/'samples.jsonl').values()},
                         {'evaluation_wall_limit'})
        self.assertTrue((out/'drafts-samples.json').exists())

    def test_deadline_during_scoring_keeps_samples(self):
        out = self.root/'mid'
        def remaining(args):
            return 0. if args.progress.phase_name == 'score' and args.progress.done >= 1 else 100.
        with mock.patch.object(W, 'remaining_wall_s', remaining):
            self.invoke(self.base(out, '--local-smoke', '--scheduler', 'continuous'))
        self.assertFalse((out/'workflow-benchmark.json').exists())
        self.assertEqual(json.loads((out/'progress.json').read_text())['phase'], 'done')
        self.assertEqual(len((out/'samples.jsonl').read_text().splitlines()), 12)

    def test_field_temperature_is_validated(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.invoke(self.base(self.root/'bad', '--local-smoke', '--field-temperature', '{"kinds": 0.5}'))
            with self.assertRaises(SystemExit):
                self.invoke(self.base(self.root/'bad', '--local-smoke', '--consensus-votes', '9'))


class BenchSchedTests(unittest.TestCase):
    def test_tiny_bench_reports_rate_and_identity(self):
        with contextlib.redirect_stdout(io.StringIO()):
            report = bench_sched.main(['--tiny', '--device', 'cpu', '--batch-sizes', '4,12', '--samples', '3', '--songs', '3',
                                       '--token-cap', '128', '--arm-wall-s', '60'])
        self.assertEqual(len(report['arms']), 6)
        self.assertGreater(report['cores_compared'], 0)
        self.assertEqual(report['cores_different'], 0)
        self.assertTrue(all(arm['tok_s'] > 0 and arm['stats']['billed_row_steps'] > 0 for arm in report['arms']))


if __name__ == '__main__':
    unittest.main()
