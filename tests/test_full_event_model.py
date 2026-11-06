import unittest
from dataclasses import replace

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import (BOS, FIELDS, EventRequest, EventState,
    EventVocabulary, encode_events, teacher_sequence)
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss


class EventCodecTests(unittest.TestCase):
    def test_hold_tail_does_not_skip_intermediate_heads_or_duplicate_drag(self):
        for mode, events in [
            ('mania_4k', [RhythmEvent(1000, 9_000_000, 'hold', 0),
                          RhythmEvent(2000, 2000, 'tap', 1), RhythmEvent(3000, 3000, 'tap', 2)]),
            ('phira', [RhythmEvent(1000, 9_000_000, 'hold'),
                       RhythmEvent(2000, 2000, 'drag'), RhythmEvent(2000, 2000, 'drag'),
                       RhythmEvent(3000, 3000, 'flick')])]:
            request = EventRequest(mode, 0, 8_000_000, 10_000_000)
            tokens = encode_events(events, request)
            state = EventState(request)
            for token in tokens:
                state.apply(token)
            self.assertEqual(tuple(sorted(state.events)), tuple(sorted(events)))
            self.assertTrue(state.done)

    def test_copy_time_and_incoming_holds_preserve_exact_boundaries(self):
        req = EventRequest('mania_4k', 1_000_000, 8_000_000, 10_000_000,
            copy_times=(1_500_123,), known_carry=(RhythmEvent(0, 2_000_000, 'hold', 0),))
        events = [RhythmEvent(1_500_123, 1_500_123, 'tap', 1),
                  RhythmEvent(2_000_000, 3_000_000, 'hold', 0)]
        self.assertTrue(encode_events(events, req))
        with self.assertRaises(ValueError):
            encode_events([RhythmEvent(1_500_123, 1_500_123, 'tap', 0)], req)
        with self.assertRaises(ValueError):
            encode_events([RhythmEvent(1_500_124, 1_500_124, 'tap', 1)], req)

    def test_all_occupied_keys_do_not_admit_a_dead_end_group(self):
        req = EventRequest('mania_4k', 1_000_000, 2_000_000, 5_000_000,
            known_carry=tuple(RhythmEvent(0, 3_000_000, 'hold', lane) for lane in range(4)))
        self.assertEqual(EventState(req).allowed_values(), (0,))

    def test_type_permissions_and_capacity_never_silently_remove_targets(self):
        req = EventRequest('phira', 0, 8_000_000, 10_000_000, allowed_types=('tap',))
        with self.assertRaises(ValueError):
            encode_events([RhythmEvent(1000, 1000, 'drag')], req)
        req = EventRequest('phira', 0, 8_000_000, 10_000_000, max_events=1)
        with self.assertRaises(ValueError):
            encode_events([RhythmEvent(1000, 1000, 'tap'), RhythmEvent(2000, 2000, 'tap')], req)


def inputs(c):
    return dict(local=torch.randn(1, c.taps, 6, c.bands, c.tap_dim),
                local_times=torch.arange(6)[None].float() * .04,
                local_valid=torch.tensor([[True] * 5 + [False]]),
                song=torch.randn(1, 5, c.tap_dim), song_times=torch.arange(5)[None].float() * .5,
                song_valid=torch.tensor([[True] * 4 + [False]]),
                reference=torch.randn(1, 3, c.reference_dim),
                reference_times=torch.tensor([[-1., .2, .5]]),
                reference_valid=torch.tensor([[True, True, False]]),
                controls=torch.randn(1, c.controls, c.control_dim), mode=torch.tensor([0]))


class EventModelTests(unittest.TestCase):
    def test_loss_weights_requests_equally_and_ignores_empty_padding(self):
        logits = torch.tensor([[[3., 0.]] * 3, [[0., 3.]] * 3, [[4., 2.]] * 3])
        seq = dict(allowed=torch.ones_like(logits, dtype=torch.bool),
                   targets=torch.zeros(3, 3, dtype=torch.long),
                   valid=torch.tensor([[True, False, False], [True] * 3, [False] * 3]))
        expected = torch.nn.functional.cross_entropy(logits[:2, 0], seq['targets'][:2, 0])
        torch.testing.assert_close(event_loss(logits, seq), expected)

    def test_default_mainline_has_quarter_billion_trainable_parameters(self):
        with torch.device('meta'):
            model = FullEventModel()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 250681347)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def setUp(self):
        torch.manual_seed(20261025)
        self.c = FullEventConfig(d=32, heads=4, kv_heads=2, ffn=48,
            encoder_layers=1, decoder_layers=2, tap_dim=16, bands=4, dropout=0.)
        self.model = FullEventModel(self.c).eval()
        self.inputs = inputs(self.c)
        req = EventRequest('mania_4k', 0, 8_000_000, 10_000_000)
        tokens = encode_events([RhythmEvent(1000, 3_000_000, 'hold', 0),
                               RhythmEvent(2000, 2000, 'tap', 1)], req)
        self.seq = {k: torch.from_numpy(v)[None] for k, v in teacher_sequence(tokens, req).items()}

    def test_microbatch_four_matches_four_equal_weight_requests(self):
        from beatfcn.train.training_objective import training_loss
        batch = {k: v.repeat((4,) + (1,) * (v.ndim-1)) for k, v in self.inputs.items()}
        sequence = {k: v.repeat((4,) + (1,) * (v.ndim-1)) for k, v in self.seq.items()}
        batch['local'] = batch['local'] + torch.arange(4)[:, None, None, None, None] * .1
        settings = dict(type_weight=.1, z_weight=.0001, first_token_weight=5.)
        batched_loss = training_loss(self.model(batch, sequence), sequence, settings)
        parameters = tuple(self.model.parameters())
        batched_grad = torch.autograd.grad(batched_loss, parameters)
        singles = []
        for index in range(4):
            inp = {k: v[index:index+1] for k, v in batch.items()}
            seq = {k: v[index:index+1] for k, v in sequence.items()}
            singles.append(training_loss(self.model(inp, seq), seq, settings) / 4)
        scalar_loss = sum(singles)
        scalar_grad = torch.autograd.grad(scalar_loss, parameters)
        torch.testing.assert_close(batched_loss, scalar_loss, atol=1e-5, rtol=1e-5)
        for a, b in zip(batched_grad, scalar_grad):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)

    def test_cached_and_full_forward_match_without_seeing_future(self):
        with torch.no_grad():
            memory = self.model.encode(self.inputs)
            full = self.model.decode(*(self.seq[k] for k in ['tokens', 'fields', 'times', 'valid']), memory)
            cache = self.model.start_cache(memory)
            pieces = [self.model.decode(*(self.seq[k][:, i:i+1] for k in ['tokens', 'fields', 'times', 'valid']),
                      memory, cache=cache) for i in range(full.shape[1])]
            self.assertLess((full - torch.cat(pieces, 1)).abs().max().item(), 1e-4)
            altered = {k: v.clone() for k, v in self.seq.items()}
            altered['tokens'][:, 7:] = BOS
            altered['times'][:, 7:] += 10
            changed = self.model(self.inputs, altered)
            torch.testing.assert_close(full[:, :7], changed[:, :7], atol=1e-5, rtol=1e-5)

    def test_masked_audio_and_reference_are_not_visible(self):
        modified = {k: v.clone() for k, v in self.inputs.items()}
        modified['local'][:, :, -1] = 10000
        modified['song'][:, -1] = 10000
        modified['reference'][:, -1] = 10000
        with torch.no_grad():
            torch.testing.assert_close(self.model(self.inputs, self.seq), self.model(modified, self.seq),
                                       atol=1e-5, rtol=1e-5)

    def test_every_encoder_and_output_receive_finite_gradients(self):
        self.model.train()
        loss = event_loss(self.model(self.inputs, self.seq), self.seq)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name, p in self.model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        for i in range(3):
            norm = sum(p.grad.square().sum().item() for p in self.model.encoders[i].parameters())
            self.assertGreater(norm, 0, f'encoder {i}')

    def test_mel_timing_mask_and_gradient(self):
        model = FullEventModel(replace(self.c, mel_bins=128)).eval()
        batch = {**self.inputs, 'mel': torch.randn(1, 24, 128),
                 'mel_times': torch.arange(24)[None].float() * .01 + .005,
                 'mel_valid': torch.tensor([[True] * 21 + [False] * 3])}
        before = model(batch, self.seq)
        changed = {k: v.clone() for k, v in batch.items()}
        changed['mel'][:, 21:] = 12345.
        torch.testing.assert_close(before, model(changed, self.seq), atol=1e-5, rtol=1e-5)
        event_loss(before, self.seq).backward()
        self.assertGreater(model.mel_projection.weight.grad.norm().item(), 0.)
        memory = model.encode(batch)
        self.assertEqual(int(memory['valid'].sum()), 10 + 21 + 4 + 3 + self.c.controls)

    def test_fixed_batch_cached_logits_match_full_independent_requests(self):
        batch={k:v.repeat((2,)+(1,)*(v.ndim-1)) for k,v in self.inputs.items()}
        sequence={k:v.repeat((2,)+(1,)*(v.ndim-1)) for k,v in self.seq.items()}
        batch['mode'][1]=1
        with torch.no_grad():
            memory=self.model.encode(batch)
            full=self.model(batch,sequence)
            cache=self.model.start_cache(memory)
            pieces=[self.model.decode(*(sequence[k][:,i:i+1] for k in ['tokens','fields','times','valid']),
                      memory,cache=cache) for i in range(full.shape[1])]
            torch.testing.assert_close(full,torch.cat(pieces,1),atol=1e-5,rtol=1e-5)
        with self.assertRaises(ValueError):
            self.model.decode(*(sequence[k][:1,:1] for k in ['tokens','fields','times','valid']),memory,cache=cache)

    def test_batched_generation_matches_scalar_and_deadline_is_explicit(self):
        from beatfcn.model.full_event_generate import generate_batch
        from scripts.probe_full_event_model import generate
        request=EventRequest('mania_4k',0,8_000_000,10_000_000)
        batch={k:v.repeat((2,)+(1,)*(v.ndim-1)) for k,v in self.inputs.items()}
        actual=generate_batch(self.model,batch,[request,request],20)
        reference=generate(self.model,self.inputs,request,20)
        for row in actual:
            self.assertEqual(row['events'],reference['events'])
            self.assertEqual(row['tokens'],reference['tokens'])
        stopped=generate_batch(self.model,batch,[request,request],20,deadline=0.)
        self.assertTrue(all(r['deadline_reached'] and not r['complete'] for r in stopped))


if __name__ == '__main__':
    unittest.main()
