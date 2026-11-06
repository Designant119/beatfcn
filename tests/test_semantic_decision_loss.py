import unittest

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import FIELDS, EventRequest, encode_events, teacher_sequence
from beatfcn.train.semantic_decision_loss import semantic_decision_loss, semantic_decision_weights


class SemanticDecisionLossTests(unittest.TestCase):
    def sequence(self, events, mode='phira', copy_times=()):
        req = EventRequest(mode, 0, 8_000_000, 10_000_000, copy_times=copy_times)
        seq = teacher_sequence(encode_events(events, req), req)
        seq['semantic_weights'], counts = semantic_decision_weights(seq)
        return seq, counts

    def test_empty_and_dense_start_have_identical_weight(self):
        empty, _ = self.sequence([])
        dense, _ = self.sequence([RhythmEvent(i*100_000, i*100_000, 'tap') for i in range(40)])
        self.assertEqual(empty['semantic_weights'][0], dense['semantic_weights'][0])
        self.assertAlmostEqual(float(empty['semantic_weights'].sum()), 1/6)

    def test_type_family_cannot_be_diluted_by_time_digits_or_more_taps(self):
        for count in (1, 8):
            seq, units = self.sequence([RhythmEvent(i*100_000, i*100_000, 'tap') for i in range(count)])
            kind = seq['fields'] == FIELDS.index('kind')
            self.assertAlmostEqual(float(seq['semantic_weights'][kind].sum()), 1/6)
            self.assertEqual(units['type'], count)
            self.assertTrue((seq['semantic_weights'][seq['allowed'].sum(-1) == 1] == 0).all())

    def test_copy_onset_and_hold_tail_are_distinct_units(self):
        seq, units = self.sequence([RhythmEvent(123, 1_000_000, 'hold')], copy_times=(123,))
        self.assertEqual(units['onset'], 1)
        self.assertEqual(units['hold_end'], 1)
        tail = np.isin(seq['fields'], [FIELDS.index(k) for k in ('tail_block', 'tail_cell', 'tail_ms')])
        np.testing.assert_allclose(seq['semantic_weights'][tail], 1/6)

    def test_mania_pattern_is_supervised_without_phira_member_units(self):
        _, units = self.sequence([RhythmEvent(0, 0, 'tap', 0), RhythmEvent(0, 1_000_000, 'hold', 2)], 'mania_4k')
        self.assertEqual(units['type'], 1)
        self.assertEqual(units['members'], 0)
        self.assertEqual(units['hold_end'], 1)

    def test_padding_has_zero_gradient_and_batch_is_request_mean(self):
        seq, _ = self.sequence([RhythmEvent(0, 0, 'tap')])
        ts = {k: torch.from_numpy(v)[None] for k, v in seq.items()}
        torch.manual_seed(7)
        logits = torch.randn(1, len(seq['targets']), seq['allowed'].shape[1], requires_grad=True)
        value = semantic_decision_loss(logits, ts)
        gradient = torch.autograd.grad(value, logits)[0]
        padded = {}
        for k, v in ts.items():
            out = v.new_zeros((1, v.shape[1]+3, *v.shape[2:]));out[:, :v.shape[1]] = v;padded[k] = out
        padded['allowed'][:, -3:, 0] = True
        larger = torch.cat([logits.detach(), torch.randn(1, 3, logits.shape[-1])], 1).requires_grad_(True)
        result = semantic_decision_loss(larger, padded)
        grad = torch.autograd.grad(result, larger)[0]
        torch.testing.assert_close(value, result)
        torch.testing.assert_close(gradient, grad[:, :-3])
        self.assertEqual(float(grad[:, -3:].abs().sum()), 0.)
        doubled = {k: v.repeat(2, *([1]*(v.ndim-1))) for k,v in ts.items()}
        torch.testing.assert_close(value, semantic_decision_loss(logits.repeat(2, 1, 1), doubled))

    def test_orphan_time_digit_rejected(self):
        seq, _ = self.sequence([RhythmEvent(0, 0, 'tap')])
        seq['fields'][1] = FIELDS.index('head_cell')
        with self.assertRaisesRegex(ValueError, 'Orphan'):
            semantic_decision_weights(seq)


if __name__ == '__main__':
    unittest.main()
