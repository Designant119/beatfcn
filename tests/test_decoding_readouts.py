"""Decoding readouts D3/D4/D6, history reference and the silence split (tiny CPU models).

D3 onset shallow fusion (generate_batch onset_fusion; fast == reference),
D4 rollout condition_provider + CFG plumbing (workflow --density-request/--guidance),
D6 consensus drafts (constructed samples), --history-reference wiring,
true-silence vs uncharted-music split (benchmark v2 windows + whole songs).
"""
from dataclasses import replace
import math
from types import SimpleNamespace
import time
import unittest
from unittest import mock
from unittest.mock import patch

import numpy as np
import torch

from beatfcn.model import full_event_generate as gen
from beatfcn.model.full_event_codec import EventRequest, EventState
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.model.full_event_model import FullEventModel
from beatfcn.train import requirement_diagnostics as D
import scripts.run_workflow_benchmark as wb
from test_fast_generate import FakeFeatures, both, hold_heavy, make_inputs, requests_mixed, row_seeded, tiny


def fake_memory(requests, peaks_s, frames=800, hop_s=.01):
    """Onset head memory: p ~ 1 at the given request-relative seconds, ~0 elsewhere."""
    times = torch.arange(frames)[None].float().repeat(len(requests), 1) * hop_s
    logits = torch.full((len(requests), frames, 5), -20.)
    for peak in peaks_s:
        logits[:, int(round(peak / hop_s)), 0] = 20.
    return dict(onset_logits=logits, onset_times=times, onset_valid=torch.ones(len(requests), frames, dtype=torch.bool))


# ---------------------------------------------------------------- D3
class OnsetFusionBiasTests(unittest.TestCase):
    def test_tables_and_bias_per_head_phase(self):
        request = EventRequest('mania_4k', 2_000_000, 10_000_000, 20_000_000)
        (lo, table), = gen._onset_tables(fake_memory([request], [1.22]), [request], 20)
        self.assertEqual(lo, 2000)
        self.assertEqual(len(table), 8000)
        hit = 3220 - lo
        self.assertTrue(np.all(table[hit-20:hit+21] > .99))           # +-20 ms dilation
        self.assertLess(table[hit-21], 1e-6)
        self.assertLess(table[hit+21], 1e-6)
        state = EventState(request)
        state.apply(state.vocab.token('group', 1))                       # -> head_block
        bias = gen._onset_bias(state, (lo, table), .5)
        values = state.allowed_values()
        self.assertEqual(len(bias), len(values))
        good = .5 * math.log(1 + 1e-4)
        bad = .5 * math.log(1e-4)
        self.assertAlmostEqual(float(bias[list(values).index(1)]), good, places=4)  # block 1 = 2-4 s holds 3.22 s
        self.assertTrue(all(abs(float(b) - bad) < 1e-3 for v, b in zip(values, bias) if v != 1))
        state.apply(state.vocab.token('head_block', 1))                  # -> head_cell (block 1 = 2..4 s)
        bias = gen._onset_bias(state, (lo, table), .5)
        cells = list(state.allowed_values())
        expect = [good if (1220 - 20 <= 40 * c + 39 and 40 * c <= 1220 + 20) else bad for c in cells]
        np.testing.assert_allclose(bias, expect, atol=1e-3)
        self.assertEqual(sum(b > -1 for b in bias), 2)  # cells 30 and 31 (3200-3279 ms)
        state.apply(state.vocab.token('head_cell', 30))                  # 3200..3239 ms
        bias = gen._onset_bias(state, (lo, table), .5)
        ms = 3200 + np.asarray(state.allowed_values())
        np.testing.assert_allclose(bias, [good if abs(t - 3220) <= 20 else bad for t in ms], atol=1e-3)
        state.apply(state.vocab.token('head_ms', 34))                    # -> mania: not a head-time field
        self.assertIsNone(gen._onset_bias(state, (lo, table), .5))

    def test_copy_times_use_their_exact_time(self):
        request = EventRequest('phira', 0, 8_000_000, 8_000_000, copy_times=(1_000_000, 5_500_000))
        tables = gen._onset_tables(fake_memory([request], [5.5]), [request], 20)
        state = EventState(request)
        state.apply(state.vocab.token('group', 1))
        values = list(state.allowed_values())
        bias = gen._onset_bias(state, tables[0], 1.)
        copy_bias = {v - state.vocab.blocks: b for v, b in zip(values, bias) if v >= state.vocab.blocks}
        self.assertAlmostEqual(float(copy_bias[1]), math.log(1 + 1e-4), places=4)
        self.assertAlmostEqual(float(copy_bias[0]), math.log(1e-4), places=3)

    def test_validation(self):
        for bad in (dict(), dict(weight=0.), dict(weight=.5, extra=1), dict(weight=.5, window_ms=2.5), .5):
            with self.assertRaises(ValueError):
                gen._onset_fusion_rule(bad)
        rule = gen._onset_fusion_rule(dict(weight=.5))
        self.assertEqual((rule['weight'], rule['window_ms'], rule['floor']), (.5, 20, 1e-4))


class OnsetFusionGenerationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.c = tiny(qk_norm=True, mel_bins=8, intensity_dim=3, timed_intensity=True, onset_head=True)
        self.model = hold_heavy(FullEventModel(self.c).eval())
        self.requests = requests_mixed()
        self.inputs = make_inputs(self.c, batch=4)

    def test_fast_equals_reference_with_fusion(self):
        fusion = dict(onset_fusion=dict(weight=.5))
        ref, fast = both(self.model, self.inputs, self.requests, 300, **fusion)
        self.assertEqual(ref, fast)
        self.assertEqual(ref[0]['decoding']['onset_fusion']['weight'], .5)
        for kw in (dict(temperature=1.), dict(temperature=.8, top_p=.9, field_temperature={'kind': .3, 'head_ms': .5})):
            gens = [torch.Generator().manual_seed(i) for i in range(4)]
            ref, fast = both(self.model, self.inputs, self.requests, 300, generator=gens, **kw, **fusion)
            self.assertEqual(ref, fast)
        with mock.patch.object(gen, '_batched_sampling_ok', lambda g: False):
            gens = [torch.Generator().manual_seed(i) for i in range(4)]
            ref, fast = both(self.model, self.inputs, self.requests, 200, generator=gens, temperature=1., **fusion)
            self.assertEqual(ref, fast)
        ref, fast = both(self.model, self.inputs, self.requests, 200, temperature=1.,
                         generator=torch.Generator().manual_seed(3), **fusion)
        self.assertEqual(ref, fast)
        uncond = dict(self.inputs, controls=torch.zeros_like(self.inputs['controls']))
        ref, fast = both(self.model, self.inputs, self.requests, 150, guidance=2., unconditional_inputs=uncond, **fusion)
        self.assertEqual(ref, fast)

    def test_off_by_default_and_changes_output_when_on(self):
        plain = generate_batch(self.model, self.inputs, self.requests, 300)
        self.assertNotIn('onset_fusion', plain[0]['decoding'])
        fused = generate_batch(self.model, self.inputs, self.requests, 300, onset_fusion=dict(weight=3.))
        self.assertNotEqual([r['tokens'] for r in plain], [r['tokens'] for r in fused])

    def test_strong_fusion_pulls_heads_onto_onset_peaks(self):
        peaks = [.5, 1.25, 3., 6.4]
        original = self.model.encode

        def encode(inputs):
            memory = original(inputs)
            memory.update(fake_memory([None] * len(inputs['mode']), peaks))
            return memory
        requests = [EventRequest('mania_4k', 0, 8_000_000, 8_000_000), EventRequest('phira', 0, 8_000_000, 8_000_000)]
        inputs = {k: v[:2] for k, v in self.inputs.items()}
        for fast in (False, True):
            with mock.patch.object(self.model, 'encode', encode):
                out = generate_batch(self.model, inputs, requests, 400, onset_fusion=dict(weight=8.), fast=fast)
            # After the last peak every candidate has the same (floor) bias: the model chooses freely there.
            heads = [e['onset_us'] / 1e6 for row in out for e in row['events'] if e['onset_us'] / 1e6 <= peaks[-1] + .02]
            self.assertGreater(len({round(h, 3) for h in heads}), 3)
            self.assertTrue(all(min(abs(h - p) for p in peaks) <= .0205 for h in heads), heads)

    def test_requires_onset_head(self):
        model = FullEventModel(replace(self.c, onset_head=False)).eval()
        for fast in (False, True):
            with self.assertRaisesRegex(ValueError, 'onset head'):
                generate_batch(model, self.inputs, self.requests, 10, onset_fusion=dict(weight=.5), fast=fast)


# ---------------------------------------------------------------- D4 rollout hook
class ConditionProviderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.c = tiny(taps=3)
        self.model = hold_heavy(FullEventModel(self.c).eval(), end_bias=-1.)
        self.songs = [dict(chart_id='m', audio_sha256='a' * 64, mode='mania_4k', duration_us=19_500_000),
                      dict(chart_id='p', audio_sha256='b' * 64, mode='phira', duration_us=13_000_000)]
        self.features = FakeFeatures(self.c, {s['audio_sha256']: s['duration_us'] for s in self.songs})

    def test_default_rollout_never_passes_unconditional_inputs(self):
        from beatfcn.model.local_song_rollout import generate_complete_songs
        calls = []

        def generator(model, inputs, requests, max_tokens, *, deadline=None):  # no unconditional kwarg accepted
            calls.append(len(requests))
            return generate_batch(model, inputs, requests, max_tokens, deadline=deadline)
        generate_complete_songs(self.model, self.features, self.songs, device='cpu', batch_size=2, token_cap=200,
                                generator=generator)
        self.assertTrue(calls)

    def test_provider_conditions_rows_and_supplies_unconditional_batch(self):
        from beatfcn.model.local_song_rollout import generate_complete_songs
        seen = []

        def provider(state, request):
            if state['mode'] != 'mania_4k':
                return None
            return dict(inputs=dict(controls=np.full((self.c.controls, self.c.control_dim), 3., np.float32)),
                        record=dict(label='charter_intent_density', start=request.start_us))

        def generator(model, inputs, requests, max_tokens, *, deadline=None, unconditional_inputs=None):
            seen.append((inputs['controls'].clone(), None if unconditional_inputs is None
                         else unconditional_inputs['controls'].clone(), [r.mode for r in requests]))
            return generate_batch(model, inputs, requests, max_tokens, deadline=deadline,
                                  guidance=2., unconditional_inputs=unconditional_inputs)
        result = generate_complete_songs(self.model, self.features, self.songs, device='cpu', batch_size=2,
                                         token_cap=200, generator=generator, condition_provider=provider)
        for cond, uncond, modes in seen:
            self.assertIsNotNone(uncond)
            for row, mode in enumerate(modes):
                if mode == 'mania_4k':
                    self.assertTrue(torch.all(cond[row] == 3.))
                    self.assertFalse(torch.all(uncond[row] == 3.))
                else:
                    self.assertTrue(torch.equal(cond[row], uncond[row]))
        mania = next(s for s in result['songs'] if s['mode'] == 'mania_4k')
        self.assertTrue(all(c['condition']['label'] == 'charter_intent_density' for c in mania['cores']))
        phira = next(s for s in result['songs'] if s['mode'] == 'phira')
        self.assertTrue(all('condition' not in c for c in phira['cores']))

    def test_parallel_samples_with_provider_equal_separate_rollouts(self):
        from beatfcn.model.local_song_rollout import generate_complete_song_samples, generate_complete_songs

        def provider(state, request):
            return dict(inputs=dict(controls=np.full((self.c.controls, self.c.control_dim), .5, np.float32)))

        def plain(seed, fast):  # conditioning only (no CFG): drop the unconditional batch
            inner = row_seeded(seed, fast=fast)
            return lambda *a, unconditional_inputs=None, **k: inner(*a, **k)
        pooled = generate_complete_song_samples(self.model, self.features, self.songs, [4, 5], device='cpu',
                                                batch_size=64, token_cap=200, generator=plain(0, True),
                                                condition_provider=provider)
        alone = generate_complete_songs(self.model, self.features, self.songs, device='cpu', batch_size=1,
                                        token_cap=200, generator=plain(5, False), condition_provider=provider)
        strip = lambda r: [[e for e in st['events']] for st in r['songs']]  # noqa: E731
        self.assertEqual(strip(pooled[1]), strip(alone))
        self.assertNotEqual(strip(pooled[0]), strip(pooled[1]))

    def test_reranker_scores_conditioned_cores_against_the_request(self):
        from beatfcn.model.rerank import score_song
        tap = lambda t: dict(onset_us=t, end_us=t, type='tap', lane=-1)  # noqa: E731
        state = dict(duration_us=8_000_000, complete=True, failure=None,
                     events=[tap(t) for t in range(0, 8_000_000, 500_000)],
                     cores=[dict(start_us=0, end_us=8_000_000, complete=True)])
        plain = score_song(state)[1]['sum_density']
        cond = dict(state, cores=[dict(state['cores'][0], condition=dict(segments=[
            dict(start_us=0, end_us=8_000_000, density=6.)]))])
        self.assertEqual(plain, 0.)
        self.assertAlmostEqual(score_song(cond)[1]['sum_density'], 4.)   # produced 2/s vs requested 6/s

    def test_density_provider_renders_reference_curve(self):
        intent = {'s': dict(chart_id='osu:1', stratum='x',
                            notes=[(t, t, 'tap', 0) for t in range(0, 4_000_000, 250_000)])}
        provider = wb.density_condition_provider(intent, cell_us=100_000)
        request = EventRequest('mania_4k', 0, 8_000_000, 20_000_000)
        out = provider(dict(chart_id='s', mode='mania_4k'), request)
        self.assertEqual(out['record']['label'], 'charter_intent_density')
        self.assertEqual([s['density'] for s in out['record']['segments']], [4., 4., 0., 0.])
        self.assertTrue(out['inputs']['intensity_valid'].all())
        self.assertIn('controls', out['inputs'])


# ---------------------------------------------------------------- workflow wiring
def fake_inputs_sources(seen):
    def fake(features, audio, request, controls, *, sources=()):
        seen.append((request.start_us, sources))
        return dict(local=np.zeros((3, 1, 8, 1024), np.float32), mode=np.asarray(1),
                    controls=controls, reference=np.zeros((1, 16), np.float32)), []
    return fake


def recording_generate_batch(calls):
    def stub(model, inputs, requests, max_tokens, *, temperature=None, **kw):
        calls.append(dict(kw, temperature=temperature))
        return [dict(complete=True, tokens=[], failure=None, events=[]) for _ in requests]
    return stub


SONGS = [dict(song_id='phira:aaaa', audio_sha256='a'*64, mode='phira', duration_us=17_000_000, references=[])]


def wf_args(**kw):
    base = dict(temperature='1', top_p=None, seed=5, device='cpu', batch_size=4, token_cap=64, max_wall_s=60.)
    return SimpleNamespace(**dict(base, **kw))


class WorkflowWiringTests(unittest.TestCase):
    def pool(self, **kw):
        seen, calls = [], []
        with patch('beatfcn.model.local_song_rollout.request_inputs', fake_inputs_sources(seen)), \
             patch('beatfcn.model.full_event_generate.generate_batch', recording_generate_batch(calls)):
            samples, _ = wb.sample_pool(wf_args(**kw), dict(songs=SONGS), SimpleNamespace(training=False, c=None),
                                        None, intensity_dim=0, cell_us=100_000, best_of=1)
        return samples, seen, calls

    def test_history_reference_off_by_default_and_on_when_asked(self):
        samples, seen, calls = self.pool()
        self.assertTrue(all(not sources for _, sources in seen))
        self.assertNotIn('history_reference', samples[0]['decoding'])
        self.assertNotIn('onset_fusion', calls[0])
        samples, seen, _ = self.pool(history_reference=True)
        self.assertTrue(samples[0]['result']['history_reference'])
        self.assertTrue(samples[0]['decoding']['history_reference']['enabled'])
        later = [sources for start, sources in seen if start > 0]
        self.assertTrue(later and all(s and s[0].role == 'model_history' for s in later))

    def test_onset_fusion_and_guidance_forwarded(self):
        samples, _, calls = self.pool(onset_fusion=.5)
        self.assertEqual(calls[0]['onset_fusion'], dict(weight=.5))
        self.assertEqual(samples[0]['decoding']['onset_fusion']['weight'], .5)
        calls = []
        g = wb.make_generator(1., None, 0, guidance=2.)
        req = [EventRequest('phira', 0, 8_000_000, 8_000_000)]
        with patch('beatfcn.model.full_event_generate.generate_batch', recording_generate_batch(calls)):
            g(None, dict(local=torch.zeros(1)), req, 8)
            g(None, dict(local=torch.zeros(1)), req, 8, unconditional_inputs={'x': 1})
        self.assertNotIn('guidance', calls[0])
        self.assertEqual((calls[1]['guidance'], calls[1]['unconditional_inputs']), (2., {'x': 1}))

    def test_cfg_weight_rules(self):
        self.assertIsNone(wb.cfg_weight(SimpleNamespace(guidance=2., density_request=None)))
        self.assertIsNone(wb.cfg_weight(SimpleNamespace(guidance=1., density_request='reference')))
        self.assertEqual(wb.cfg_weight(SimpleNamespace(guidance=2.5, density_request='reference')), 2.5)
        dec = wb.draft_decoding(SimpleNamespace(top_p=None, token_cap=64, density_request='reference', guidance=1.),
                                1., 0)
        self.assertEqual(dec['density_request']['label'], 'charter_intent_density')
        self.assertIn('plain', dec['density_request']['cfg'])


# ---------------------------------------------------------------- D6
def taps(*items):
    return [(int(t*1000), int(t*1000), 'tap', lane) for t, lane in items]


class ConsensusTests(unittest.TestCase):
    def test_mania_majority_median_holds_and_overlap(self):
        s0 = taps((1000, 0)) + [(3000_000, 3500_000, 'hold', 2)] + [(4000_000, 5000_000, 'hold', 3)] + taps((4500, 3))
        s1 = taps((1010, 0), (2000, 1)) + [(3005_000, 3600_000, 'hold', 2)] + [(4000_000, 5000_000, 'hold', 3)] \
            + taps((4500, 3))
        s2 = taps((995, 1)) + [(3000_000, 3700_000, 'hold', 2)]
        notes, info = wb.consensus_draft([s0, s1, s2], 'mania_4k', 10_000_000)
        self.assertEqual(info['min_votes'], 2)
        self.assertEqual(notes, [(1000_000, 1000_000, 'tap', 0),               # median 1000; lane 0 wins 2:1
                                 (3000_000, 3600_000, 'hold', 2),              # median tail
                                 (4000_000, 4000_000, 'tap', 3),               # hold would contain the 4.5 s head
                                 (4500_000, 4500_000, 'tap', 3)])
        self.assertEqual(info['holds_dropped_for_overlap'], 1)
        self.assertEqual(wb.consensus_legality(notes, 'mania_4k', 10_000_000), [])
        from beatfcn.train.workflow_metrics import playability
        p = playability(notes, 'mania_4k')
        self.assertEqual((p['same_lane_overlap'], p['hold_covers_head']), (0, 0))

    def test_ties_go_to_earliest_sample_and_window(self):
        notes, _ = wb.consensus_draft([taps((1000, 2)), taps((1020, 1))], 'mania_4k', 5_000_000)
        self.assertEqual(notes, [(1000_000, 1000_000, 'tap', 2)])
        apart, info = wb.consensus_draft([taps((1000, 2)), taps((1040, 2)), taps((3000, 0))], 'mania_4k', 5_000_000)
        self.assertEqual(apart, [])                                         # 40 ms apart -> two 1-vote clusters
        self.assertEqual(info['clusters'], 3)

    def test_phira_multiset_and_hold_tail(self):
        a = [(1000_000, 1000_000, 'tap', -1), (1000_000, 1800_000, 'hold', -1)]
        b = [(1010_000, 1010_000, 'tap', -1), (1010_000, 1900_000, 'hold', -1)]
        c = [(1005_000, 1005_000, 'flick', -1)]
        notes, _ = wb.consensus_draft([a, b, c], 'phira', 5_000_000)
        self.assertEqual(notes, [(1005_000, 1800_000, 'hold', -1), (1005_000, 1005_000, 'tap', -1)])
        self.assertEqual(wb.consensus_legality(notes, 'phira', 5_000_000), [])

    def test_random_samples_always_legal(self):
        rng = np.random.default_rng(0)
        for trial in range(20):
            samples = []
            for k in range(int(rng.integers(2, 8))):
                notes, ends = [], [0]*4
                for t in np.sort(rng.choice(np.arange(0, 20_000, 7), 120, replace=False)):
                    lane = int(rng.integers(0, 4))
                    if ends[lane] > t*1000:
                        continue
                    if rng.random() < .3:
                        end = min(int(t+rng.integers(1, 900))*1000, 20_000_000)
                        notes.append((int(t)*1000, end, 'hold', lane))
                        ends[lane] = end
                    else:
                        notes.append((int(t)*1000, int(t)*1000, 'tap', lane))
                samples.append(notes)
            draft, _ = wb.consensus_draft(samples, 'mania_4k', 20_000_000)
            self.assertEqual(wb.consensus_legality(draft, 'mania_4k', 20_000_000), [], trial)

    def test_consensus_readout_beside_single_sample(self):
        song = dict(song_id='mania_4k:x', mode='mania_4k', duration_us=10_000_000, audio_sha256='c'*64)
        ref = [(t, t, 'tap', t//500_000 % 4) for t in range(0, 10_000_000, 500_000)]
        refs = {song['song_id']: [dict(chart_id='osu:1', stratum='medium', coarse_stratum='medium', notes=ref)]}
        samples, evaluated = [], []
        for k, d in enumerate([ref[:5], ref[:15], ref[:15]]):
            state = dict(events=[dict(onset_us=a, end_us=b, type=t, lane=l) for a, b, t, l in d], complete=True)
            samples.append(dict(seed=k, by_id={song['song_id']: state}))
            r, row = wb.evaluate(d, dict(draft_id='model'), song, refs[song['song_id']], grids={}, per_op=False,
                                 split='val')
            evaluated.append({song['song_id']: (r, row)})
        consensus, drafts = wb.consensus_evaluated(samples, [song], refs, {}, per_op=False, split='val')
        readouts = wb.build_readouts(samples, evaluated, [song], samples_n=50, consensus=consensus)
        self.assertEqual(list(readouts), ['single_sample', 'consensus_3'])
        cer = readouts['consensus_3']['aggregate']['min_over_references']['val/mania_4k']['min_ref_cer']['mean']
        self.assertAlmostEqual(cer, .25)
        self.assertEqual(drafts[song['song_id']]['info']['legality'], [])
        self.assertEqual(readouts['consensus_3']['seeds_used'], {song['song_id']: [0, 1, 2]})


# ---------------------------------------------------------------- silence split
def song_mel(silent_ranges_s, seconds=20, quiet=-10., loud=0.):
    frames = seconds * 100
    mel = np.full((frames, 8), loud, np.float32)
    for lo, hi in silent_ranges_s:
        mel[int(lo*100):int(hi*100)] = quiet
    return mel


class SilenceSplitTests(unittest.TestCase):
    def test_energy_threshold_and_window(self):
        song = D.song_energy(song_mel([(0, 5)]))
        self.assertAlmostEqual(song['threshold'], -9.)
        self.assertTrue(D.near_silent(2_000_000, song))
        self.assertFalse(D.near_silent(7_000_000, song))
        self.assertFalse(D.near_silent(4_960_000, song))                  # music within +50 ms
        self.assertTrue(D.near_silent(4_900_000, song))
        split = D.silence_split([dict(onset_us=1_000_000), dict(onset_us=6_000_000), dict(onset_us=6_500_000)], song)
        self.assertEqual((split['true_silence_events'], split['uncharted_music_events']), (1, 2))

    def test_uncharted_regions(self):
        notes = [(1_000_000, 1_000_000), (2_000_000, 7_000_000), (12_000_000, 12_000_000)]
        regions = D.uncharted_regions(notes, 20_000_000)
        self.assertEqual(regions, [(7_100_000, 11_900_000), (12_100_000, 20_000_000)])
        self.assertEqual(D.uncharted_regions([], 5_000_000), [(0, 5_000_000)])
        song = D.song_energy(song_mel([(15, 20)]))
        split = D.uncharted_split([8_000_000, 9_000_000, 16_000_000, 3_000_000], regions, song)
        self.assertEqual((split['uncharted_notes'], split['true_silence_notes'], split['uncharted_music_notes']),
                         (3, 1, 2))
        self.assertAlmostEqual(split['uncharted_s'], 12.7)

    def test_workflow_evaluate_and_aggregate_carry_split_only_when_measured(self):
        song = dict(song_id='mania_4k:x', mode='mania_4k', duration_us=20_000_000)
        ref = [(t, t, 'tap', 0) for t in range(0, 8_000_000, 500_000)]
        refs = [dict(chart_id='osu:1', stratum='m', coarse_stratum='m', notes=ref)]
        draft = ref + [(12_000_000, 12_000_000, 'tap', 1), (17_000_000, 17_000_000, 'tap', 2)]
        energy = D.song_energy(song_mel([(15, 20)]))
        rows, row = wb.evaluate(draft, dict(draft_id='model'), song, refs, grids={}, per_op=False, split='val',
                                energy=energy)
        self.assertEqual((row['silence_split']['true_silence_notes'], row['silence_split']['uncharted_music_notes']),
                         (1, 1))
        agg = wb.aggregate(rows, [row], samples=20, seed=0)
        self.assertIn('silence_split', agg['min_over_references']['val/mania_4k'])
        self.assertIn('true_silence_events_per_s', agg['by_split+mode']['val/mania_4k'])
        self.assertTrue(any('Silence split' in line for line in wb.aggregate_tables(agg)))
        plain_rows, plain = wb.evaluate(draft, dict(draft_id='model'), song, refs, grids={}, per_op=False, split='val')
        self.assertNotIn('silence_split', plain)
        plain_agg = wb.aggregate(plain_rows, [plain], samples=20, seed=0)
        self.assertNotIn('silence_split', plain_agg['min_over_references']['val/mania_4k'])
        self.assertNotIn('true_silence_events_per_s', plain_agg['by_split+mode']['val/mania_4k'])


class BenchmarkV2Tests(unittest.TestCase):
    def setUp(self):
        from test_benchmark_v2 import fixture
        self.model, self.panel = fixture()

    def test_silence_split_in_hacks_and_old_metric_kept(self):
        from beatfcn.train import learning_evaluation as L
        protocol = L.freeze_generation_panel(self.panel, per_mode_positive=1, per_mode_silence=1, anchors_per_mode=0)
        readouts = (dict(name='sample_t1', temperature=1., seeds=(0,)),)
        plain = L.run_generation_benchmark(self.model, self.panel, protocol, device='cpu',
                                           deadline=time.monotonic()+60, readouts=readouts)
        hacks = plain['readouts']['sample_t1']['per_seed']['0']['hacks']
        self.assertIsNone(hacks['true_silence_events_per_s'])           # fixture features expose no mel
        # Song mel: first 4 s near-silent, rest loud.
        self.panel.features.mel_frames = lambda audio: song_mel([(0, 4)])
        result = L.run_generation_benchmark(self.model, self.panel, protocol, device='cpu',
                                            deadline=time.monotonic()+60, readouts=readouts)
        hacks = result['readouts']['sample_t1']['per_seed']['0']['hacks']
        silence_rows = [r for r in result['rows'] if r['case_role'] == 'natural_silence']
        events = [e for r in silence_rows for e in r['prediction']['events']]
        expect_silent = sum(e['onset_us'] < 3_950_000 for e in events)
        seconds = sum((r['request']['end_us']-r['request']['start_us'])/1e6 for r in silence_rows)
        self.assertAlmostEqual(hacks['true_silence_events_per_s'], expect_silent/seconds)
        self.assertAlmostEqual(hacks['true_silence_events_per_s']+hacks['uncharted_music_events_per_s'],
                               hacks['silence_events_per_s'])
        self.assertEqual(hacks['silence_events_per_s'],
                         plain['readouts']['sample_t1']['per_seed']['0']['hacks']['silence_events_per_s'])
        self.assertIn('hacks.true_silence_events_per_s', L.aggregate_seeds(result['readouts']['sample_t1']['per_seed']))

    def test_onset_fusion_and_guidance_readouts_on_density_anchors(self):
        from beatfcn.train import learning_evaluation as L
        model = FullEventModel(replace(self.model.c, onset_head=True)).eval()
        protocol = L.freeze_generation_panel(self.panel, per_mode_positive=1, per_mode_silence=0, anchors_per_mode=1)
        result = L.run_generation_benchmark(model, self.panel, protocol, device='cpu', deadline=time.monotonic()+60,
            readouts=(dict(name='onset_fusion', temperature=1., seeds=(0,), onset_fusion=.5),
                      dict(name='cfg_w2.5', temperature=1., seeds=(0,), guidance=2.5)))
        fused = [r for r in result['rows'] if r['readout']['name'] == 'onset_fusion']
        self.assertTrue(all(r['prediction']['decoding']['onset_fusion']['weight'] == .5 for r in fused))
        guided = [r for r in result['rows'] if r['readout']['name'] == 'cfg_w2.5']
        density = [r for r in guided if r['condition'].startswith('density_')]
        self.assertTrue(density and all(r['prediction']['decoding']['guidance'] == 2.5 for r in density))
        anchors = result['readouts']['cfg_w2.5']['per_seed']['0']['condition_response']['anchors']
        self.assertTrue(anchors and all('density_high' in a['rates'] for a in anchors))

    def test_benchmark_v2_cli_extra_readouts(self):
        from scripts.run_benchmark_v2 import extra_readouts
        self.assertEqual(extra_readouts(SimpleNamespace(onset_fusion=None, guidance=None, seeds=2)), [])
        out = extra_readouts(SimpleNamespace(onset_fusion=.5, guidance=[1.5, 2.5], seeds=2))
        self.assertEqual([r['name'] for r in out], ['onset_fusion', 'cfg_w1.5', 'cfg_w2.5'])
        self.assertEqual(out[0]['onset_fusion'], dict(weight=.5))


if __name__ == '__main__':
    unittest.main()
