import unittest
from unittest.mock import patch

import numpy as np
import torch

from beatfcn.events.canonical import Event, LaneLoc, LineLoc
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.model.full_event_codec import EventRequest, encode_events, teacher_sequence
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.train.event_conditions import (CONTROL_FIELDS, condition_match_loss,
    conditional_event_objective, encode_controls, in_band, paired_control_views, training_labels,
    validate_pair_metadata)
from tests.test_full_event_model import inputs


class ControlLabelTests(unittest.TestCase):
    def event(self, at, end=None, kind='tap', lane=None):
        loc = LaneLoc(lane) if lane is not None else LineLoc('1', 'above', 0.)
        return Event(str(at), at, at if end is None else end, kind, loc)

    def test_zero_unknown_and_inapplicable_are_distinct(self):
        scope = (1_000_000, 9_000_000)
        unknown = encode_controls('phira', {}, scope, origin_us=scope[0])
        zero = encode_controls('phira', {'hold_head_ratio': [0., .02]}, scope, origin_us=scope[0])
        hold, jack = map(CONTROL_FIELDS.index, ['hold_head_ratio', 'jack_ratio'])
        self.assertEqual(zero[hold, 0], 0.)
        self.assertEqual(zero[hold, 2], 1.)
        self.assertEqual(unknown[hold, 2], 0.)
        self.assertEqual(unknown[hold, 3], 1.)
        self.assertTrue((zero[jack] == 0).all())
        self.assertAlmostEqual(float(zero[hold, 5]), 8/1200)

    def test_incoming_hold_and_out_of_scope_heads(self):
        events = [self.event(0, 3_000_000, 'hold'), self.event(2_000_000), self.event(7_000_000)]
        bands, raw = training_labels('phira', events, (1_000_000, 5_000_000))
        self.assertEqual(raw['density_notes_per_second'], .25)
        self.assertEqual(raw['hold_occupancy_ratio'], .5)
        self.assertEqual(raw['hold_head_ratio'], 0.)
        self.assertEqual(bands['hold_occupancy_ratio'], [.5, .6])

    def test_artistic_drag_multiplicity_is_not_removed(self):
        e = self.event(1_000_000, kind='drag')
        _, raw = training_labels('phira', [e, e], (0, 8_000_000))
        self.assertEqual(raw['drag_ratio'], 1.)
        self.assertEqual(raw['density_notes_per_second'], .25)
        self.assertEqual(raw['chord_ratio'], 1.)

    def test_opposing_views_are_symmetric_and_single_field(self):
        a = [self.event(1_000_000)]
        b = [self.event(1_000_000, kind='drag')]
        v = paired_control_views('phira', a, b, (0, 8_000_000), 'drag_ratio', origin_us=0)
        np.testing.assert_array_equal(v['controls'][::-1], v['opposing_controls'])
        self.assertEqual(v['controls'][:, :, 2].sum(), 2)
        self.assertEqual(v['actual'], [0., 1.])
        with self.assertRaisesRegex(ValueError, 'does not contradict'):
            paired_control_views('phira', a, a, (0, 8_000_000), 'drag_ratio', origin_us=0)

    def test_invalid_conditions_and_empty_ratio_do_not_become_negatives(self):
        for bands in [{'jack_ratio': [0., .02]}, {'drag_ratio': [float('nan'), 1.]},
                      {'drag_ratio': [.21, .31]}]:
            with self.assertRaises(ValueError):
                encode_controls('phira', bands, (0, 8_000_000), origin_us=0)
        with self.assertRaisesRegex(ValueError, 'Undefined'):
            paired_control_views('phira', [], [self.event(0)], (0, 8_000_000),
                                 'hold_head_ratio', origin_us=0)
        with self.assertRaises(ValueError):
            encode_controls('phira', {}, (0, 0), origin_us=0)
        self.assertTrue(in_band('drag_ratio', 1., [.8, 1.]))
        self.assertFalse(in_band('density_notes_per_second', 64., [48., 64.]))


class ConditionObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261025)
        self.c = FullEventConfig(d=32, heads=4, kv_heads=2, ffn=48, encoder_layers=1,
                                decoder_layers=2, tap_dim=16, bands=4, dropout=0.)
        self.model = FullEventModel(self.c).eval()
        self.batch = {k: v.repeat((2,) + (1,) * (v.ndim - 1)) for k, v in inputs(self.c).items()}
        self.batch['controls'][1] += .5
        r = EventRequest('mania_4k', 0, 8_000_000, 10_000_000)
        t = teacher_sequence(encode_events([RhythmEvent(1000, 1000, 'tap', 0)], r), r)
        self.seq = {k: torch.from_numpy(v)[None].repeat((2,) + (1,) * v.ndim) for k, v in t.items()}

    def test_replacing_memory_matches_reencoding_without_mutation(self):
        memory = self.model.encode(self.batch)
        saved = memory['values'].clone()
        changed = self.batch['controls'].flip(0)
        replacement = self.model.replace_controls(memory, changed)
        actual = self.model.encode({**self.batch, 'controls': changed})
        torch.testing.assert_close(replacement['values'], actual['values'], atol=0, rtol=0)
        torch.testing.assert_close(memory['values'], saved, atol=0, rtol=0)
        keys = ('tokens', 'fields', 'times', 'valid')
        logits = self.model.decode(*(self.seq[k] for k in keys), replacement)
        torch.testing.assert_close(logits, self.model({**self.batch, 'controls': changed}, self.seq))

    def test_zero_weight_preserves_default_loss(self):
        expected = event_loss(self.model(self.batch, self.seq), self.seq)
        actual, stats = conditional_event_objective(self.model, self.batch, self.seq)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertIsNone(stats['condition_match'])

    def test_objective_encodes_once_and_gradients_match_naive_two_passes(self):
        import copy
        other = copy.deepcopy(self.model)
        altered = self.batch['controls'].flip(0)
        rows = torch.arange(2)
        with patch.object(self.model, 'encode', wraps=self.model.encode) as encode:
            loss, stats = conditional_event_objective(self.model, self.batch, self.seq,
                pair_rows=rows, opposing_controls=altered, weight=.2, margin=.1)
            loss.backward()
            self.assertEqual(encode.call_count, 1)
        pos = other(self.batch, self.seq)
        neg = other({**self.batch, 'controls': altered}, self.seq)
        rank, _ = condition_match_loss(pos, neg, self.seq, margin=.1)
        naive = event_loss(pos, self.seq) + .2 * rank
        naive.backward()
        torch.testing.assert_close(loss, naive)
        self.assertGreater(float(stats['condition_match']), 0.)
        for (name, p), (other_name, q) in zip(self.model.named_parameters(), other.named_parameters()):
            self.assertEqual(name, other_name)
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5, msg=name)

    def test_matching_loss_pushes_correct_up_and_wrong_down(self):
        pos = torch.zeros(2, 3, 2, requires_grad=True)
        neg = torch.zeros_like(pos, requires_grad=True)
        seq = dict(targets=torch.zeros(2, 3, dtype=torch.long),
                   allowed=torch.ones(2, 3, 2, dtype=torch.bool),
                   valid=torch.tensor([[True, False, False], [True, True, True]]))
        rank, gap = condition_match_loss(pos, neg, seq, margin=.1)
        rank.backward()
        self.assertTrue((pos.grad[:, 0, 0] < 0).all())
        self.assertTrue((neg.grad[:, 0, 0] > 0).all())
        self.assertTrue((pos.grad[0, 1:] == 0).all())
        self.assertAlmostEqual(float(pos.grad[0, :, 0].sum()), float(pos.grad[1, :, 0].sum()))
        self.assertTrue((gap == 0).all())

    def test_pair_metadata_uses_real_alternatives_and_equal_audio(self):
        a = encode_controls('mania_4k', {'hold_head_ratio':[0., .02]}, (0, 8_000_000), origin_us=0)
        b = encode_controls('mania_4k', {'hold_head_ratio':[.8, 1.]}, (0, 8_000_000), origin_us=0)
        self.batch['controls'] = torch.from_numpy(np.stack([a,b]))
        rows, opposing = torch.arange(2), self.batch['controls'].flip(0)
        validate_pair_metadata(self.batch, rows, opposing)
        with self.assertRaisesRegex(ValueError, 'Repeated'):
            validate_pair_metadata(self.batch, torch.tensor([0,0]), opposing)
        with self.assertRaisesRegex(ValueError, 'other real member'):
            validate_pair_metadata(self.batch, rows, self.batch['controls'])
        self.batch['local'][1,0,0,0,0] += 1
        with self.assertRaisesRegex(ValueError, 'non-condition input: local'):
            validate_pair_metadata(self.batch, rows, opposing)

    def test_contrast_metadata_survives_existing_packed_transport_format(self):
        from beatfcn.train.event_transport import pack_event_batch, split_event_batch
        from beatfcn.model.batch import unpack_batch
        a = encode_controls('mania_4k', {'hold_head_ratio':[0., .02]}, (0, 8_000_000), origin_us=0)
        b = encode_controls('mania_4k', {'hold_head_ratio':[.8, 1.]}, (0, 8_000_000), origin_us=0)
        self.batch['controls'] = torch.from_numpy(np.stack([a,b]))
        contrast = dict(pair_rows=torch.arange(2), opposing_controls=self.batch['controls'].flip(0))
        packed = pack_event_batch(self.batch, self.seq, dict(keys=['same','same']), condition_contrast=contrast)
        restored, _ = split_event_batch(unpack_batch(packed))
        torch.testing.assert_close(restored['condition_pair_rows'], contrast['pair_rows'])
        torch.testing.assert_close(restored['opposing_controls'], contrast['opposing_controls'])
        direct, _ = conditional_event_objective(self.model, self.batch, self.seq, weight=.2, **contrast)
        carried, _ = conditional_event_objective(self.model, {**self.batch, **restored}, self.seq, weight=.2)
        torch.testing.assert_close(direct, carried)


if __name__ == '__main__':
    unittest.main()
