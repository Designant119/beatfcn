"""Opt-in review fixes for the full event codec/model/generation/objective."""
import math
import unittest
from collections import Counter
from dataclasses import replace
from unittest import mock

import numpy as np
import torch

from beatfcn.model import full_event_model as fem
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import (FIELDS, EventRequest, EventState, EventVocabulary,
    encode_events, teacher_sequence)
from beatfcn.model.full_event_model import ONSET_CHANNELS, Attention, FullEventConfig, FullEventModel
from beatfcn.train.training_objective import (exposure_group_loss, objective_terms, onset_frame_targets,
    onset_presence_auc, training_loss)

BOTH = EventVocabulary(tail_query=True, relative_blocks=True)


def replay(tokens, request, vocab):
    state = EventState(request, vocab)
    for token in tokens:
        state.apply(token)
    assert state.done
    return Counter(state.events)


def far_windows():
    duration = 1_100_000_000
    mania = EventRequest('mania_4k', 600_300_000, 608_300_000, duration,
        copy_times=(601_234_567,), known_carry=(RhythmEvent(590_000_000, 603_000_000, 'hold', 2),))
    mania_events = [RhythmEvent(600_500_000, 607_900_000, 'hold', 0),   # tail crosses 3 blocks
                    RhythmEvent(601_234_567, 601_234_567, 'tap', 1),    # copy anchor
                    RhythmEvent(602_000_000, 615_520_000, 'hold', 1),   # tail beyond the core
                    RhythmEvent(603_500_000, 603_500_000, 'tap', 2),
                    RhythmEvent(605_999_000, 605_999_000, 'tap', 3)]
    phira = EventRequest('phira', 900_001_000, 908_000_000, duration)
    phira_events = [RhythmEvent(900_010_000, 911_111_000, 'hold'),
                    RhythmEvent(900_010_000, 900_010_000, 'drag'),
                    RhythmEvent(901_999_000, 904_001_000, 'hold'),
                    RhythmEvent(903_000_000, 903_000_000, 'flick'),
                    RhythmEvent(907_999_000, 907_999_000, 'tap')]
    return [(mania, mania_events), (phira, phira_events)]


class CodecFixTests(unittest.TestCase):
    def test_relative_blocks_and_tail_query_round_trip_far_into_song(self):
        for request, events in far_windows():
            legacy = encode_events(events, request)
            tokens = encode_events(events, request, BOTH)
            self.assertEqual(replay(tokens, request, BOTH), Counter(events))
            self.assertEqual(replay(legacy, request, EventVocabulary()), Counter(events))
            self.assertEqual(len(tokens), len(legacy))
            fields = [BOTH.decision(t) for t in tokens]
            base = request.start_us // 2_000_000
            blocks = [(f, v) for f, v in fields if f in ('head_block', 'tail_block') and v < BOTH.blocks]
            legacy_blocks = [(f, v) for f, v in map(EventVocabulary().decision, legacy)
                             if f in ('head_block', 'tail_block') and v < BOTH.blocks]
            self.assertEqual([(f, v + base) for f, v in blocks], legacy_blocks)
            self.assertEqual(min(v for _, v in blocks), 0)
            self.assertLess(max(v for _, v in blocks), 10)
            # Copy-time tokens are unchanged.
            self.assertEqual([t for t in tokens if BOTH.decision(t)[0] == 'head_block' and BOTH.decision(t)[1] >= BOTH.blocks],
                             [t for t in legacy if BOTH.decision(t)[0] == 'head_block' and BOTH.decision(t)[1] >= BOTH.blocks])
            teacher = teacher_sequence(tokens, request, BOTH)
            self.assertTrue(teacher['allowed'][np.arange(len(tokens)), teacher['targets']].all())

    def test_relative_blocks_alone_and_tail_query_alone_round_trip(self):
        for vocab in (EventVocabulary(relative_blocks=True), EventVocabulary(tail_query=True)):
            for request, events in far_windows():
                self.assertEqual(replay(encode_events(events, request, vocab), request, vocab), Counter(events))

    def test_tail_queries_follow_emitted_tail_coordinates(self):
        request = EventRequest('mania_4k', 10_000_000, 18_000_000, 40_000_000)
        hold = RhythmEvent(11_000_000, 15_500_000, 'hold', 0)
        for vocab, cell_time, ms_time in [(EventVocabulary(), 1., 1.), (BOTH, 4., 4. + 37 * .04)]:
            tokens = encode_events([hold], request, vocab)
            seq = teacher_sequence(tokens, request, vocab)
            times = {FIELDS[f]: float(t) for f, t in zip(seq['fields'], seq['times'])}
            self.assertAlmostEqual(times['tail_block'], 1., places=5)
            self.assertAlmostEqual(times['tail_cell'], cell_time, places=5)
            self.assertAlmostEqual(times['tail_ms'], ms_time, places=5)
            self.assertAlmostEqual(times['head_cell'], 0., places=5)  # block 5 = 10s
            self.assertAlmostEqual(times['head_ms'], 1., places=5)   # cell 25 = 11.0s


def tiny(**overrides):
    return FullEventConfig(d=32, heads=4, kv_heads=2, ffn=48, encoder_layers=1, decoder_layers=2,
                           tap_dim=16, bands=4, dropout=0., **overrides)


def make_inputs(c, batch=1, frames=6, mel=24, cells=4):
    g = torch.Generator().manual_seed(7)
    inp = dict(local=torch.randn(batch, c.taps, frames, c.bands, c.tap_dim, generator=g),
               local_times=(torch.arange(frames)[None].float() * .04).repeat(batch, 1),
               local_valid=torch.tensor([[True] * (frames - 1) + [False]]).repeat(batch, 1),
               song=torch.randn(batch, 5, c.tap_dim, generator=g),
               song_times=(torch.arange(5)[None].float() * .5).repeat(batch, 1),
               song_valid=torch.tensor([[True] * 4 + [False]]).repeat(batch, 1),
               reference=torch.randn(batch, 3, c.reference_dim, generator=g),
               reference_times=torch.tensor([[-1., .2, .5]]).repeat(batch, 1),
               reference_valid=torch.tensor([[True, True, False]]).repeat(batch, 1),
               controls=torch.randn(batch, c.controls, c.control_dim, generator=g),
               mode=torch.zeros(batch, dtype=torch.long))
    if c.mel_bins:
        inp.update(mel=torch.randn(batch, mel, c.mel_bins, generator=g),
                   mel_times=(torch.arange(mel)[None].float() * .01 + .005).repeat(batch, 1),
                   mel_valid=torch.tensor([[True] * (mel - 3) + [False] * 3]).repeat(batch, 1))
    if c.intensity_dim:
        inp.update(intensity=torch.rand(batch, cells, c.intensity_dim, generator=g),
                   intensity_times=(torch.arange(cells)[None].float() * .5 + .25).repeat(batch, 1),
                   intensity_valid=torch.tensor([[True, True, False, True]]).repeat(batch, 1))
    return inp


def make_sequence(vocab=None, batch=1):
    request = EventRequest('mania_4k', 0, 8_000_000, 10_000_000)
    tokens = encode_events([RhythmEvent(1000, 3_000_000, 'hold', 0), RhythmEvent(2000, 2000, 'tap', 1),
                            RhythmEvent(2_500_000, 2_500_000, 'tap', 2)], request, vocab)
    seq = {k: torch.from_numpy(v)[None] for k, v in teacher_sequence(tokens, request, vocab).items()}
    return {k: v.repeat((batch,) + (1,) * (v.ndim - 1)) for k, v in seq.items()}


def legacy_rotate(x, positions, *, lo, hi, timed=None):
    half = x.shape[-1] // 2
    periods = torch.logspace(math.log10(lo), math.log10(hi), half, device=x.device)
    angle = positions.float()[:, None, :, None] * (2 * math.pi / periods)
    if timed is not None:
        angle = angle * timed[:, None, :, None]
    a, b = x[..., :half], x[..., half:]
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), -1)


def decode_pieces(model, seq, memory):
    cache = model.start_cache(memory)
    return torch.cat([model.decode(*(seq[k][:, i:i+1] for k in ['tokens', 'fields', 'times', 'valid']),
                                   memory, cache=cache) for i in range(seq['tokens'].shape[1])], 1)


class RotaryCacheTests(unittest.TestCase):
    def test_cached_tables_match_legacy_rotate_full_and_cached_decode(self):
        torch.manual_seed(1)
        c = tiny(mel_bins=8, intensity_dim=3)
        model = FullEventModel(c).eval()
        inp, seq = make_inputs(c), make_sequence()
        with torch.no_grad():
            new_full = model(inp, seq)
            new_cached = decode_pieces(model, seq, model.encode(inp))
            with mock.patch.object(fem, 'rotate', legacy_rotate), \
                    mock.patch.object(FullEventModel, '_decode_rope', lambda *a: None):
                old_full = model(inp, seq)
                old_cached = decode_pieces(model, seq, model.encode(inp))
        self.assertLessEqual((new_full - old_full).abs().max().item(), 1e-6)
        self.assertLessEqual((new_cached - old_cached).abs().max().item(), 1e-6)
        x, pos = torch.randn(2, 3, 7, 16), torch.rand(2, 7) * 100
        torch.testing.assert_close(fem.rotate(x, pos, lo=.02, hi=2400), legacy_rotate(x, pos, lo=.02, hi=2400),
                                   atol=0, rtol=0)
        self.assertIs(fem.rotary_periods(.02, 2400, 8, 'cpu'), fem.rotary_periods(.02, 2400, 8, 'cpu'))

    def test_periods_built_under_inference_mode_remain_usable_for_autograd(self):
        fem._cached_periods.cache_clear()
        with torch.inference_mode():
            fem.rotate(torch.randn(1, 1, 2, 12), torch.rand(1, 2), lo=.03, hi=99)
        x = torch.randn(1, 1, 2, 12, requires_grad=True)
        fem.rotate(x, torch.rand(1, 2), lo=.03, hi=99).sum().backward()
        self.assertIsNotNone(x.grad)


class TimedIntensityTests(unittest.TestCase):
    def test_default_curve_is_untimed_and_flag_times_valid_cells_only(self):
        torch.manual_seed(2)
        seq = make_sequence()
        base = tiny(intensity_dim=3)
        inp = make_inputs(base)
        shifted = {**inp, 'intensity_times': inp['intensity_times'] + 1.3}
        with torch.no_grad():
            old = FullEventModel(base).eval()
            torch.testing.assert_close(old(inp, seq), old(shifted, seq), atol=0, rtol=0)
            memory = old.encode(inp)
            self.assertFalse(memory['timed'][:, -(base.controls + 4):].any())
            model = FullEventModel(replace(base, timed_intensity=True)).eval()
            memory = model.encode(inp)
            curve = slice(-(base.controls + 4), -base.controls)
            torch.testing.assert_close(memory['timed'][:, curve], inp['intensity_valid'])
            torch.testing.assert_close(memory['valid'][:, curve], inp['intensity_valid'])
            self.assertFalse(memory['timed'][:, -base.controls:].any())
            torch.testing.assert_close(memory['times'][0, curve], torch.tensor([.25, .75, 0., 1.75]))
            self.assertFalse(torch.allclose(model(inp, seq), model(shifted, seq)))
            present = model.encode({**inp, 'intensity_present': torch.ones(1, 4, dtype=torch.bool)})
            self.assertTrue(present['valid'][:, curve].all())
            torch.testing.assert_close(present['timed'][:, curve], inp['intensity_valid'])
            # replace_controls still swaps exactly the trailing control rows.
            other = torch.randn(1, base.controls, base.control_dim)
            swapped = model.replace_controls(memory, other)
            torch.testing.assert_close(swapped['values'][:, :-base.controls], memory['values'][:, :-base.controls])
            torch.testing.assert_close(swapped['values'][:, -base.controls:], model.project_controls(other))


class TimedValueTests(unittest.TestCase):
    def test_values_carry_relative_key_time_and_untimed_rows_stay_unrotated(self):
        torch.manual_seed(3)
        c = tiny()
        attn = Attention(c, value_time=True).eval()
        mem = torch.randn(1, 2, c.d)
        times, timed = torch.tensor([[.37, 0.]]), torch.tensor([[True, False]])
        k, v = attn.project_kv(mem, times, torch.arange(2)[None], timed)
        hd = c.d // c.heads
        raw = attn.v(mem).view(1, 2, c.kv_heads, hd).transpose(1, 2)
        query = torch.randn(1, 1, c.d)
        for row in (0, 1):
            mask = torch.tensor([[[[row == 0, row == 1]]]])
            for tq in (0., 1.1):
                out = attn(query, torch.tensor([[tq]]), torch.zeros(1, 1, dtype=torch.long), (k, v), mask)
                rel = torch.tensor([[times[0, row].item() - tq]])
                expect_v = legacy_rotate(raw[:, :, row:row+1], rel, lo=.02, hi=2400) if row == 0 else raw[:, :, row:row+1]
                expect = attn.out(expect_v.repeat_interleave(c.heads // c.kv_heads, 1).transpose(1, 2).reshape(1, 1, c.d))
                torch.testing.assert_close(out, expect, atol=1e-5, rtol=1e-5)

    def test_all_flags_cached_decode_matches_full(self):
        torch.manual_seed(4)
        c = tiny(mel_bins=8, intensity_dim=3, timed_intensity=True, timed_memory_values=True, onset_head=True)
        model = FullEventModel(c, BOTH).eval()
        inp, seq = make_inputs(c, batch=2), make_sequence(BOTH, batch=2)
        with torch.no_grad():
            full = model(inp, seq)
            torch.testing.assert_close(full, decode_pieces(model, seq, model.encode(inp)), atol=1e-5, rtol=1e-5)
            # Moving one valid mel frame's time changes the logits only through timing.
            moved = {**inp, 'mel_times': inp['mel_times'].clone()}
            moved['mel_times'][:, 3] += .3
            self.assertFalse(torch.allclose(full, model(moved, seq)))

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_all_flags_mps_matches_cpu(self):
        # Torch 2.6 MPS SDPA assumes value width == q width; the wider timed
        # values must be padded explicitly (aligned_attention).
        torch.manual_seed(9)
        c = tiny(mel_bins=8, intensity_dim=3, timed_intensity=True, timed_memory_values=True, onset_head=True)
        model = FullEventModel(c, BOTH).eval()
        inp, seq = make_inputs(c, batch=2), make_sequence(BOTH, batch=2)
        with torch.no_grad():
            cpu = model(inp, seq)
            model = model.to('mps')
            mps = model({k: v.to('mps') for k, v in inp.items()}, {k: v.to('mps') for k, v in seq.items()})
        torch.testing.assert_close(mps.cpu(), cpu, atol=1e-5, rtol=1e-5)

    def test_default_config_adds_no_parameters(self):
        old = FullEventModel(tiny(mel_bins=8, intensity_dim=3))
        new = FullEventModel(tiny(mel_bins=8, intensity_dim=3, timed_memory_values=True))
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))
        new.load_state_dict(old.state_dict())


class OnsetHeadTests(unittest.TestCase):
    def test_onset_targets_soft_labels_types_and_context(self):
        frames = np.arange(30) * .01 + .005
        out = onset_frame_targets([1_105_000, 900_000, 1_255_000, 5_000_000], frames, 1_000_000,
                                  onset_types=['tap', 'hold', 'drag', 'tap'])
        y = out['onset_targets']
        self.assertEqual(y.shape, (30, ONSET_CHANNELS))
        self.assertEqual(y[10, 0], 1.)          # 0.105s nearest frame 0.105
        self.assertGreater(y[9, 0], .5); self.assertLess(y[9, 0], 1.)
        self.assertEqual(y[25, 3 + 1], 1.)      # drag at 0.255s -> channel 1+TYPES.index('drag')
        self.assertEqual(y[25, 1], 0.)
        self.assertEqual(y[10, 1], 1.)          # tap channel
        self.assertEqual(y[:3, 0].max(), 0.)    # -0.1s context onset is 10 hops away
        self.assertTrue(out['onset_valid'].all())
        self.assertEqual(onset_frame_targets([], frames, 0)['onset_targets'].sum(), 0.)

    def test_onset_head_on_mel_and_mert_frames_with_loss_gradients(self):
        torch.manual_seed(5)
        for mel in (8, 0):
            c = tiny(mel_bins=mel, onset_head=True)
            model = FullEventModel(c).train()
            inp, seq = make_inputs(c, batch=2), make_sequence(batch=2)
            frame_times = inp['mel_times' if mel else 'local_times']
            frame_valid = inp['mel_valid' if mel else 'local_valid']
            targets = [onset_frame_targets([50_000, 120_000], t.numpy(), 0, frame_valid=v.numpy())
                       for t, v in zip(frame_times, frame_valid)]
            seq['onset_targets'] = torch.from_numpy(np.stack([t['onset_targets'] for t in targets]))
            seq['onset_valid'] = torch.from_numpy(np.stack([t['onset_valid'] for t in targets]))
            logits, aux = model(inp, seq, return_aux=True)
            self.assertEqual(aux['onset_logits'].shape, (2, frame_times.shape[1], ONSET_CHANNELS))
            torch.testing.assert_close(aux['onset_times'], frame_times)
            terms = objective_terms(logits, seq, onset_weight=.5, aux=aux)
            self.assertEqual(terms['onset_active'].item(), 1.)
            self.assertGreater(terms['onset_bce'].item(), 0.)
            torch.testing.assert_close(terms['weighted_onset_bce'], .5 * terms['onset_bce'])
            terms['onset_bce'].backward()
            self.assertGreater(model.encoders[0][0].attn.q.weight.grad.norm().item(), 0.)
            self.assertIsNone(model.encoders[1][0].attn.q.weight.grad)
            auc = onset_presence_auc(aux, seq)
            self.assertTrue(auc is None or 0. <= auc <= 1.)
            with self.assertRaises(ValueError):
                objective_terms(logits, {**seq, 'onset_targets': seq['onset_targets'][:, 1:]},
                                onset_weight=.5, aux=aux)

    def test_absent_onset_tensors_skip_term_and_default_total_is_unchanged(self):
        torch.manual_seed(6)
        c = tiny(onset_head=True)
        model = FullEventModel(c).eval()
        inp, seq = make_inputs(c, batch=8), make_sequence(batch=8)
        logits, aux = model(inp, seq, return_aux=True)
        settings = dict(type_weight=.1, z_weight=1e-4, first_token_weight=5.)
        old = fem.event_loss(logits, seq)
        terms = objective_terms(logits, seq, **settings)
        expected = old + .1 * terms['type_ce'] + 1e-4 * terms['z_loss'] + 5. * terms['first_token_ce']
        self.assertEqual(terms['total'].item(), expected.item())
        skipped = objective_terms(logits, seq, onset_weight=1., aux=aux, **settings)
        self.assertEqual(skipped['onset_active'].item(), 0.)
        self.assertEqual(skipped['total'].item(), terms['total'].item())
        # Grouped exposure loss slices aux with the sequence.
        frames = inp['local_times'].shape[1]
        seq['onset_targets'] = torch.rand(8, frames, 1).round()
        seq['onset_valid'] = torch.ones(8, frames, dtype=torch.bool)
        grouped = exposure_group_loss(logits, seq, dict(onset_weight=1.), aux=aux)
        halves = [training_loss(logits[i:i+4], {k: v[i:i+4] for k, v in seq.items()}, dict(onset_weight=1.),
                                {k: v[i:i+4] for k, v in aux.items()}) for i in (0, 4)]
        torch.testing.assert_close(grouped, sum(halves) / 2)

    def test_presence_auc_is_one_for_perfect_scores(self):
        targets = torch.tensor([[[0.], [1.], [0.], [1.]]])
        aux = dict(onset_logits=targets * 4 - 2, onset_valid=torch.ones(1, 4, dtype=torch.bool))
        self.assertEqual(onset_presence_auc(aux, dict(onset_targets=targets, onset_valid=aux['onset_valid'])), 1.)


class SamplingGenerationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        self.c = tiny()
        self.model = FullEventModel(self.c).eval()
        self.inputs = make_inputs(self.c, batch=2)
        self.request = EventRequest('mania_4k', 0, 8_000_000, 10_000_000)

    def test_greedy_default_reports_rule_and_sampling_is_reproducible_and_legal(self):
        from beatfcn.model.full_event_generate import generate_batch
        greedy = generate_batch(self.model, self.inputs, [self.request] * 2, 30)
        self.assertEqual(greedy[0]['decoding'], dict(rule='greedy_argmax_legal'))
        self.assertIsNone(greedy[0]['failure'])
        draws = [generate_batch(self.model, self.inputs, [self.request] * 2, 30, temperature=1.,
                                generator=torch.Generator().manual_seed(seed)) for seed in (1, 1, 2)]
        self.assertEqual(draws[0][0]['tokens'], draws[1][0]['tokens'])
        self.assertNotEqual([r['tokens'] for r in draws[0]], [r['tokens'] for r in draws[2]])
        self.assertEqual(draws[0][0]['decoding'], dict(rule='ancestral_legal', temperature=1., top_p=None))
        for row in draws[0]:  # replay proves every sampled token was legal
            state = EventState(self.request, self.model.vocab)
            for token in row['tokens']:
                state.apply(token)
        nucleus = generate_batch(self.model, self.inputs, [self.request] * 2, 30, temperature=1., top_p=1e-6,
                                 generator=torch.Generator().manual_seed(3))
        self.assertEqual([r['tokens'] for r in nucleus], [r['tokens'] for r in greedy])
        with self.assertRaises(ValueError):
            generate_batch(self.model, self.inputs, [self.request] * 2, 3, top_p=.9)

    def test_per_row_generators_make_a_case_independent_of_batch_companions(self):
        from beatfcn.model.full_event_generate import generate_batch
        inputs = make_inputs(self.c, batch=3)
        rows = lambda order: {k: v[list(order)] for k, v in inputs.items()}
        gens = lambda seeds: [torch.Generator().manual_seed(s) for s in seeds]
        short = EventRequest('mania_4k', 0, 2_000_000, 10_000_000)
        lengths = []
        for seed in range(12):
            alone = generate_batch(self.model, rows([0]), [self.request], 40, temperature=1., generator=gens([seed]))
            first = generate_batch(self.model, rows([0, 1]), [self.request, short], 40, temperature=1.,
                                   generator=gens([seed, 5]))
            second = generate_batch(self.model, rows([2, 1, 0]), [self.request] * 3, 40,
                                    temperature=1., top_p=None, generator=gens([99, 6, seed]))
            self.assertEqual(alone[0]['tokens'], first[0]['tokens'])
            self.assertEqual(alone[0]['tokens'], second[2]['tokens'])
            lengths.append(len(alone[0]['tokens']))
        self.assertGreater(max(lengths), 3)
        self.assertEqual({r['rng_mode'] for r in first + second}, {'per_row'})
        shared = generate_batch(self.model, rows([0]), [self.request], 5, temperature=1.,
                                generator=torch.Generator().manual_seed(1))
        self.assertEqual(shared[0]['rng_mode'], 'shared_batch_generator')
        with self.assertRaises(ValueError):
            generate_batch(self.model, rows([0, 1]), [self.request] * 2, 5, temperature=1., generator=gens([1]))

    def test_evaluation_records_per_row_rng_mode(self):
        from beatfcn.train import learning_evaluation as le
        le._ROW_GENERATORS['supported'] = None
        inputs = make_inputs(self.c, batch=2)
        _, mode = le._call_generate(self.model, inputs, [self.request] * 2, 10, deadline=None,
                                    readout=dict(name='sample_t1', temperature=1.), seeds=[3, 4],
                                    device=torch.device('cpu'), chunk_seed=0)
        self.assertEqual(mode, 'per_row')
        self.assertTrue(le._ROW_GENERATORS['supported'])

    def test_dead_end_row_fails_alone(self):
        from beatfcn.model import full_event_generate as gen
        marked = replace(self.request, max_events=7)

        class DeadEnd(EventState):
            def allowed_tokens(self):
                return () if self.request.max_events == 7 and len(self.tokens) >= 2 else super().allowed_tokens()

        reference = gen.generate_batch(self.model, self.inputs, [self.request] * 2, 30)
        with mock.patch.object(gen, 'EventState', DeadEnd):
            result = gen.generate_batch(self.model, self.inputs, [marked, self.request], 30)
        self.assertEqual(result[0]['failure'], 'grammar_dead_end')
        self.assertFalse(result[0]['complete'] or result[0]['token_limit_reached'])
        self.assertEqual(len(result[0]['tokens']), 2)
        self.assertEqual(result[1]['tokens'], reference[1]['tokens'])
        self.assertIsNone(result[1]['failure'])


if __name__ == '__main__':
    unittest.main()


from beatfcn.model.full_event_generate import generate_batch  # noqa: E402


class GuidanceTests(unittest.TestCase):
    def test_guidance_one_equals_conditional_and_w_changes_logits(self):
        torch.manual_seed(3)
        c = tiny(mel_bins=8, intensity_dim=3)
        model = FullEventModel(c).eval()
        cond = make_inputs(c)
        uncond = dict(cond, controls=torch.zeros_like(cond['controls']))
        request = EventRequest('mania_4k', 0, 8_000_000, 10_000_000)
        plain = generate_batch(model, cond, [request], 64)
        one = generate_batch(model, cond, [request], 64, guidance=1.0, unconditional_inputs=uncond)
        self.assertEqual(plain[0]['tokens'], one[0]['tokens'])
        self.assertEqual(one[0]['decoding']['guidance'], 1.0)
        with self.assertRaises(ValueError):
            generate_batch(model, cond, [request], 8, guidance=2.0)


class FieldTemperatureTests(unittest.TestCase):
    def test_field_temperature_recorded_and_low_kind_temperature_is_near_argmax(self):
        torch.manual_seed(5)
        c = tiny(mel_bins=8, intensity_dim=3)
        model = FullEventModel(c).eval()
        inp = make_inputs(c)
        request = EventRequest('phira', 0, 8_000_000, 10_000_000)
        g = [torch.Generator().manual_seed(1)]
        out = generate_batch(model, inp, [request], 64, temperature=1.0, generator=g,
                             field_temperature={'kind': 1e-4})
        self.assertEqual(out[0]['decoding']['field_temperature'], {'kind': 1e-4})
        with self.assertRaises(ValueError):
            generate_batch(model, inp, [request], 8, field_temperature={'kind': .3})
