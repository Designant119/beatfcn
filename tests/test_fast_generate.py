"""Fast generation path == reference path (full_event_generate.generate_batch).

Covers: FastEventState supports vs EventState (randomized grammar walks),
static decode logits vs cached decode, greedy / per-row / shared / global-RNG
sampling token equality for mania + Phira incl. holds and incoming carries,
finished / dead-end / deadline rows, guidance, unsupported-model fallback,
whole-song rollout equality and multi-seed rollout == separate rollouts.
CUDA-only graph checks are skipped without a GPU.
"""
from dataclasses import replace
import random
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from beatfcn.model import full_event_generate as gen
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import (EventRequest, EventState, EventVocabulary, FastEventState)
from beatfcn.model.full_event_generate import fast_supported, generate_batch
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel

MPS = torch.backends.mps.is_available()


def tiny(**kw):
    return FullEventConfig(**dict(dict(d=32, heads=4, kv_heads=2, ffn=48, encoder_layers=1, decoder_layers=2,
                                       tap_dim=16, bands=4, dropout=0.), **kw))


def make_inputs(c, batch=1, seed=0, frames=10):
    g = torch.Generator().manual_seed(seed)
    inp = dict(local=torch.randn(batch, c.taps, frames, c.bands, c.tap_dim, generator=g),
               local_times=(torch.arange(frames)[None].float() * .8 - .5).repeat(batch, 1),
               local_valid=torch.ones(batch, frames, dtype=torch.bool),
               song=torch.randn(batch, 5, c.tap_dim, generator=g),
               song_times=(torch.arange(5)[None].float() * 2.).repeat(batch, 1),
               song_valid=torch.tensor([[True] * 4 + [False]]).repeat(batch, 1),
               reference=torch.randn(batch, 3, c.reference_dim, generator=g),
               reference_times=torch.tensor([[-1., .2, .5]]).repeat(batch, 1),
               reference_valid=torch.tensor([[True, True, False]]).repeat(batch, 1),
               controls=torch.randn(batch, c.controls, c.control_dim, generator=g),
               mode=torch.tensor([i % 2 for i in range(batch)]))
    if c.mel_bins:
        inp.update(mel=torch.randn(batch, 24, c.mel_bins, generator=g),
                   mel_times=(torch.arange(24)[None].float() * .3).repeat(batch, 1),
                   mel_valid=torch.ones(batch, 24, dtype=torch.bool))
    if c.intensity_dim:
        inp.update(intensity=torch.rand(batch, 4, c.intensity_dim, generator=g),
                   intensity_times=(torch.arange(4)[None].float() * 2 + .5).repeat(batch, 1),
                   intensity_valid=torch.tensor([[True, False, True, True]]).repeat(batch, 1))
    return inp


def hold_heavy(model, end_bias=-2.):
    """Bias logits (both paths see it) toward holds, long tails and longer streams."""
    v = model.vocab
    bias = torch.zeros(v.size)
    bias[v.token('group', 0)] = end_bias
    bias[v.token('kind', 1)] += 2.
    model.output.register_forward_hook(lambda m, a, out: out + bias.to(out.device))
    return model


def requests_mixed():
    carry = (RhythmEvent(1_000_000, 9_500_000, 'hold', 2),)
    return [EventRequest('mania_4k', 0, 8_000_000, 20_000_000),
            EventRequest('phira', 8_000_000, 16_000_000, 20_000_000,
                         known_carry=(RhythmEvent(3_000_000, 12_000_000, 'hold', -1),)),
            EventRequest('mania_4k', 8_000_000, 16_000_000, 20_000_000, known_carry=carry),
            EventRequest('phira', 16_000_000, 20_000_000, 20_000_000)]


def both(model, inputs, requests, steps, **kw):
    """Run reference and fast with independent copies of any generator(s)."""
    def clone(g):
        if isinstance(g, torch.Generator):
            h = torch.Generator(device=g.device); h.set_state(g.get_state()); return h
        return [clone(x) for x in g]
    out = []
    for fast in (False, True):
        args = dict(kw)
        if args.get('generator') is not None:
            args['generator'] = clone(kw['generator'])
        if args.pop('global_seed', None) is not None:
            torch.manual_seed(kw['global_seed'])
        out.append(generate_batch(model, inputs, requests, steps, fast=fast, **args))
    return out


class FastGrammarTests(unittest.TestCase):
    def walk(self, request, vocab, rng, steps=400):
        fast, ref = FastEventState(request, vocab), EventState(request, vocab)
        for _ in range(steps):
            expected = EventState.allowed_values(ref)
            self.assertEqual(tuple(fast.allowed_values()), tuple(expected))
            tokens = ref.allowed_tokens()
            self.assertEqual(fast.allowed_tokens(), tokens)
            first, count, explicit = fast.allowed_spec()
            self.assertEqual(count, len(tokens))
            self.assertEqual(tuple(range(first, first + count)) if explicit is None else explicit, tokens)
            self.assertEqual(fast.query(), ref.query())
            if fast.done or not tokens:
                break
            # Prefer NEW groups and small time steps so walks cover many groups, holds and carries.
            if ref.phase in ('group', 'member') and len(tokens) == 2 and rng.random() < .93:
                token = tokens[1]
            elif rng.random() < .7:
                token = tokens[min(int(rng.expovariate(.3)), len(tokens) - 1)]
            else:
                token = rng.choice(tokens)
            try:
                ref.apply(token)
            except ValueError as error:  # e.g. max_events: the fast state must fail identically
                with self.assertRaisesRegex(ValueError, str(error)):
                    fast.apply(token)
                return ref
            fast.apply(token)
            self.assertEqual(vars(fast), vars(ref))
        with self.assertRaises(ValueError):  # illegal tokens keep the reference error
            fast.apply(0)
        return ref

    def test_fast_state_matches_reference_on_random_walks(self):
        rng = random.Random(7)
        carry_m = (RhythmEvent(1_000_000, 9_000_000, 'hold', 0), RhythmEvent(500_000, 8_000_500, 'hold', 3))
        carry_p = (RhythmEvent(2_000_000, 9_000_000, 'hold', -1),)
        base = [EventRequest('mania_4k', 0, 8_000_000, 10_000_000),
                EventRequest('mania_4k', 8_000_000, 16_000_000, 16_002_000, known_carry=carry_m),
                EventRequest('phira', 8_000_000, 16_000_000, 30_000_000, known_carry=carry_p),
                EventRequest('phira', 0, 6_000_000, 6_000_000, allowed_types=('tap', 'drag')),
                EventRequest('phira', 0, 8_000_000, 8_000_000, allowed_types=('hold',)),
                EventRequest('mania_4k', 0, 8_000_000, 9_000_000, allowed_types=('tap',)),
                EventRequest('mania_4k', 2_000_000, 8_000_000, 8_000_000,
                             copy_times=(2_000_500, 3_000_000, 7_999_999)),
                EventRequest('phira', 0, 4_000_000, 4_000_000, copy_times=(1_234_567,), max_events=6)]
        vocabs = [EventVocabulary(), EventVocabulary(relative_blocks=True, tail_query=True)]
        events = []
        for vocab in vocabs:
            for request in base:
                for _ in range(6):
                    events += self.walk(request, vocab, rng).events
        holds = [e for e in events if e.type == 'hold']
        self.assertGreater(len(events), 2000)
        # every mode now emits lane -1 (timing-only); both grammars must exercise holds
        self.assertGreater(len(holds), 100)
        self.assertTrue(all(e.lane == -1 for e in events))


class StaticDecodeTests(unittest.TestCase):
    def check(self, device, **flags):
        torch.manual_seed(0)
        c = tiny(**flags)
        model = FullEventModel(c).to(device).eval()
        inp = {k: v.to(device) for k, v in make_inputs(c, batch=3).items()}
        g = torch.Generator().manual_seed(1)
        with torch.no_grad():
            memory = model.encode(inp)
            cache, static = model.start_cache(memory), model.start_static_cache(memory, 32)
            pos = torch.zeros(1, dtype=torch.long, device=device)
            for i in range(20):
                tok = torch.randint(2, model.vocab.size, (3, 1), generator=g).to(device)
                field = torch.randint(0, 10, (3, 1), generator=g).to(device)
                t = (torch.rand(3, 1, generator=g) * 8).to(device)
                a = model.decode(tok, field, t, torch.ones_like(tok, dtype=torch.bool), memory, cache=cache)
                pos.fill_(i)
                exact = model.decode_static(tok, field, t, static, pos, i + 1)
                bucket = model.decode_static(tok, field, t, static, pos, 32)
                torch.testing.assert_close(exact, a, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(bucket, a, rtol=1e-5, atol=1e-5)

    def test_static_decode_matches_cached_decode(self):
        self.check('cpu')
        self.check('cpu', qk_norm=True, timed_memory_values=True, mel_bins=8, intensity_dim=3, timed_intensity=True)

    @unittest.skipUnless(MPS, 'MPS unavailable')
    def test_static_decode_matches_cached_decode_mps(self):
        self.check('mps', qk_norm=True, timed_memory_values=True, mel_bins=8, intensity_dim=3, timed_intensity=True)

    def test_static_cache_refuses_training_and_ssm(self):
        model = FullEventModel(tiny())
        memory = None
        with self.assertRaises(ValueError):
            model.start_static_cache(memory, 8)
        hybrid = SimpleNamespace(c=SimpleNamespace(history_mixer='hybrid'), decode_static=None,
                                 start_static_cache=None)
        self.assertFalse(fast_supported(hybrid))
        self.assertFalse(fast_supported(SimpleNamespace(c=model.c)))  # e.g. other model families
        self.assertTrue(fast_supported(model))


class FastGenerationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.c = tiny(qk_norm=True, mel_bins=8, intensity_dim=3, timed_intensity=True)
        self.model = hold_heavy(FullEventModel(self.c).eval())
        self.requests = requests_mixed()
        self.inputs = make_inputs(self.c, batch=4)

    def assert_same(self, pair):
        ref, fast = pair
        self.assertEqual(ref, fast)
        return ref

    def test_greedy_tokens_identical_mania_phira_holds_carries(self):
        ref = self.assert_same(both(self.model, self.inputs, self.requests, 400))
        events = [e for row in ref for e in row['events']]
        self.assertTrue(any(e['type'] == 'hold' for e in events), 'holds exercised')
        self.assertTrue(any(e['type'] == 'hold' and e['lane'] == -1 for e in events), 'phira holds exercised')
        self.assertGreater(len({len(r['tokens']) for r in ref}), 1, 'rows finish at different steps')

    def test_per_row_sampling_identical(self):
        for kw in (dict(temperature=1.), dict(temperature=.7, top_p=.9),
                   dict(temperature=1., field_temperature={'kind': .3, 'head_ms': 2.})):
            for seed in range(3):
                gens = [torch.Generator().manual_seed(100 * seed + i) for i in range(4)]
                self.assert_same(both(self.model, self.inputs, self.requests, 300, generator=gens, **kw))

    def test_per_row_reference_sampling_fallback_identical(self):
        gens = [torch.Generator().manual_seed(i) for i in range(4)]
        with mock.patch.object(gen, '_batched_sampling_ok', lambda g: False):
            self.assert_same(both(self.model, self.inputs, self.requests, 200, generator=gens, temperature=1.))

    def test_shared_and_global_rng_sampling_identical(self):
        self.assert_same(both(self.model, self.inputs, self.requests, 200, temperature=1.,
                              generator=torch.Generator().manual_seed(3)))
        self.assert_same(both(self.model, self.inputs, self.requests, 200, temperature=.9, top_p=.8,
                              field_temperature={'kind': .5}, generator=torch.Generator().manual_seed(4)))
        self.assert_same(both(self.model, self.inputs, self.requests, 200, temperature=1., global_seed=5))

    def test_multinomial_emulation_self_check_passes_here(self):
        self.assertTrue(gen._check_multinomial_emulation(torch.device('cpu')))
        if MPS:
            self.assertTrue(gen._check_multinomial_emulation(torch.device('mps')))

    def test_finished_rows_token_limit_and_deadline(self):
        short = [replace(r, end_us=r.start_us + 300_000) if i % 2 else r for i, r in enumerate(self.requests)]
        ref = self.assert_same(both(self.model, self.inputs, short, 400))
        self.assertTrue(ref[1]['complete'] and ref[3]['complete'])
        limited = self.assert_same(both(self.model, self.inputs, self.requests, 7))
        self.assertTrue(all(r['token_limit_reached'] and len(r['tokens']) == 7 for r in limited))
        stopped = self.assert_same(both(self.model, self.inputs, self.requests, 50, deadline=0.))
        self.assertTrue(all(r['deadline_reached'] and not r['tokens'] for r in stopped))

    def test_dead_end_row_under_substituted_state(self):
        class DeadEnd(EventState):
            def allowed_tokens(self):
                return () if self.request.max_events == 7 and len(self.tokens) >= 2 else super().allowed_tokens()
        requests = [replace(self.requests[0], max_events=7)] + self.requests[1:]
        gens = [torch.Generator().manual_seed(i) for i in range(4)]
        with mock.patch.object(gen, 'EventState', DeadEnd):
            ref = self.assert_same(both(self.model, self.inputs, requests, 200))
            self.assert_same(both(self.model, self.inputs, requests, 200, temperature=1., generator=gens))
        self.assertEqual(ref[0]['failure'], 'grammar_dead_end')
        self.assertEqual(len(ref[0]['tokens']), 2)

    def test_guidance_identical(self):
        uncond = dict(self.inputs, controls=torch.zeros_like(self.inputs['controls']))
        self.assert_same(both(self.model, self.inputs, self.requests, 150, guidance=2., unconditional_inputs=uncond))
        gens = [torch.Generator().manual_seed(i) for i in range(4)]
        self.assert_same(both(self.model, self.inputs, self.requests, 150, guidance=1.5, unconditional_inputs=uncond,
                              temperature=1., generator=gens))

    def test_unsupported_model_uses_reference(self):
        with mock.patch.object(gen, '_generate_fast', side_effect=AssertionError('fast path used')):
            with mock.patch.object(gen, 'fast_supported', lambda m: False):
                generate_batch(self.model, self.inputs, self.requests, 5)
            generate_batch(self.model, self.inputs, self.requests, 5, fast=False)

    @unittest.skipUnless(MPS, 'MPS unavailable')
    def test_mps_fast_matches_mps_reference(self):
        model = self.model.to('mps')
        inputs = {k: v.to('mps') for k, v in self.inputs.items()}
        self.assert_same(both(model, inputs, self.requests, 200))
        gens = [torch.Generator(device='mps').manual_seed(i) for i in range(4)]
        self.assert_same(both(model, inputs, self.requests, 200, temperature=1., generator=gens))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_graph_matches_eager_and_reference(self):
        model = self.model.to('cuda')
        inputs = {k: v.to('cuda') for k, v in self.inputs.items()}
        gens = lambda: [torch.Generator(device='cuda').manual_seed(i) for i in range(4)]  # noqa: E731
        for kw in (dict(), dict(temperature=1.)):
            runs = []
            for path in (dict(fast=False), dict(fast=True, cuda_graph=False), dict(fast=True, cuda_graph=True),
                         dict(fast=True, cuda_graph=True)):  # second graph call reuses the captured graphs
                extra = dict(generator=gens()) if kw else {}
                stats = {}
                runs.append(generate_batch(model, inputs, self.requests, 300, stats=stats, **path, **kw, **extra))
                if path.get('cuda_graph'):
                    self.assertTrue(stats['cuda_graph'])
            self.assertTrue(all(r == runs[0] for r in runs[1:]))
        self.assertFalse(gen._GRAPH_FAILURES)


class FakeFeatures:
    """Deterministic per-(audio, window) frontend tensors for a tiny model."""
    def __init__(self, c, durations):
        self.c, self.records = c, {a: dict(duration_us=d) for a, d in durations.items()}

    def window(self, audio, start_us, end_us):
        g = np.random.default_rng([ord(audio[0]), start_us // 1000])
        frames = 12
        return dict(local=g.standard_normal((self.c.taps, frames, self.c.bands, self.c.tap_dim)).astype(np.float32),
                    local_times=(np.arange(frames) * .75 - .5).astype(np.float32),
                    local_valid=np.ones(frames, bool),
                    song=g.standard_normal((6, self.c.tap_dim)).astype(np.float32),
                    song_times=(np.arange(6) * 4. - start_us / 1e6).astype(np.float32),
                    song_valid=np.ones(6, bool))


def row_seeded(seed, *, fast):
    """make_generator-shaped per-row sampler (seed, mode, duration, start)."""
    def generator(model, inputs, requests, max_tokens, *, deadline=None, row_seeds=None):
        seeds = row_seeds if row_seeds is not None else [seed] * len(requests)
        gens = [torch.Generator().manual_seed(s * 1_000_003 + r.start_us // 1000 + (r.mode == 'phira') * 7
                                              + r.duration_us % 9973) for s, r in zip(seeds, requests)]
        return generate_batch(model, inputs, requests, max_tokens, deadline=deadline, temperature=1.,
                              generator=gens, fast=fast)
    return generator


class RolloutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.c = tiny(taps=3)
        self.model = hold_heavy(FullEventModel(self.c).eval(), end_bias=-1.)
        self.songs = [dict(chart_id='m', audio_sha256='a' * 64, mode='mania_4k', duration_us=19_500_000),
                      dict(chart_id='p', audio_sha256='b' * 64, mode='phira', duration_us=13_000_000)]
        self.features = FakeFeatures(self.c, {s['audio_sha256']: s['duration_us'] for s in self.songs})

    @staticmethod
    def strip(result):
        return [{k: v for k, v in st.items() if k != 'cores'} | dict(cores=[
            {k: v for k, v in core.items() if k != 'batch_wall_s'} for core in st['cores']]) for st in result['songs']]

    def test_two_song_rollout_fast_equals_reference(self):
        from beatfcn.model.local_song_rollout import generate_complete_songs
        runs = {}
        for fast in (False, True):
            for name, g in (('greedy', lambda *a, fast=fast, **k: generate_batch(*a, fast=fast, **k)),
                            ('sampled', row_seeded(3, fast=fast))):
                runs[name, fast] = generate_complete_songs(self.model, self.features, self.songs, device='cpu',
                                                           batch_size=2, token_cap=300, generator=g)
        for name in ('greedy', 'sampled'):
            self.assertEqual(self.strip(runs[name, False]), self.strip(runs[name, True]))
        cores = [c for st in runs['sampled', True]['songs'] for c in st['cores']]
        self.assertEqual(len(cores), 3 + 2)
        self.assertTrue(any(c['known_carry'] for c in cores), 'a hold crossed a core boundary')

    def test_parallel_samples_equal_separate_rollouts(self):
        from beatfcn.model.local_song_rollout import generate_complete_song_samples, generate_complete_songs
        seeds = [4, 5, 6]
        pooled = generate_complete_song_samples(self.model, self.features, self.songs, seeds, device='cpu',
                                                batch_size=64, token_cap=300, generator=row_seeded(0, fast=True))
        for seed, result in zip(seeds, pooled):
            alone = generate_complete_songs(self.model, self.features, self.songs, device='cpu', batch_size=1,
                                            token_cap=300, generator=row_seeded(seed, fast=False))
            self.assertEqual(self.strip(result), self.strip(alone))
            self.assertEqual(result['complete'], alone['complete'])
        self.assertNotEqual(self.strip(pooled[0]), self.strip(pooled[1]))
        with self.assertRaises(ValueError):
            generate_complete_song_samples(self.model, self.features, self.songs, [1, 1], device='cpu')


if __name__ == '__main__':
    unittest.main()
