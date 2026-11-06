import unittest

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest, encode_events, teacher_sequence
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.train.condition_scst import (prefix_sequence, replay_log_probabilities,
    sample_trajectory, scst_loss)
from tests.test_full_event_model import inputs


class ConditionSCSTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(97)
        c = FullEventConfig(d=32, heads=4, kv_heads=2, ffn=48,
            encoder_layers=1, decoder_layers=1, tap_dim=16, bands=4, dropout=0.)
        self.model = FullEventModel(c).eval()
        self.inputs = inputs(c)
        self.inputs['mode'].fill_(1)
        self.request = EventRequest('phira', 0, 1_000_000, 2_000_000)

    def test_own_prefix_matches_complete_teacher_and_retains_capped_tail(self):
        tokens = encode_events([RhythmEvent(1000, 1_500_000, 'hold')], self.request)
        seq, state = prefix_sequence(tokens, self.request, self.model.vocab)
        self.assertTrue(state.done)
        for k, value in teacher_sequence(tokens, self.request).items():
            np.testing.assert_array_equal(seq[k], value)
        partial, state = prefix_sequence(tokens[:-2], self.request, self.model.vocab)
        self.assertFalse(state.done)
        np.testing.assert_array_equal(partial['targets'], tokens[:-2])
        with self.assertRaises(ValueError):
            prefix_sequence(list(tokens) + [tokens[-1]], self.request, self.model.vocab)

    def test_cached_sample_matches_gradient_replay_and_detects_policy_change(self):
        trajectory = sample_trajectory(self.model, self.inputs, self.request, 40)
        logp = replay_log_probabilities(self.model, self.inputs, self.request, trajectory)
        torch.testing.assert_close(logp.detach(), torch.tensor(trajectory['online_log_probabilities']), atol=2e-5, rtol=2e-5)
        self.assertTrue(logp.requires_grad)
        scst_loss(logp, 0., -1., normalizer=100.).backward()
        self.assertGreater(float(self.model.control_projection.weight.grad.norm()), 0.)
        with torch.no_grad():
            next(self.model.parameters()).add_(.01)
        with self.assertRaisesRegex(ValueError, 'Policy changed'):
            replay_log_probabilities(self.model, self.inputs, self.request, trajectory)

    def test_rewards_stop_gradient_and_normalizer_is_not_sample_length(self):
        logp = torch.tensor([-.2, -.3], requires_grad=True)
        sampled = torch.tensor(-.5, requires_grad=True)
        greedy = torch.tensor(-1., requires_grad=True)
        loss = scst_loss(logp, sampled, greedy, normalizer=10.)
        loss.backward()
        torch.testing.assert_close(logp.grad, torch.full((2,), -.05))
        self.assertIsNone(sampled.grad)
        self.assertIsNone(greedy.grad)
        long = scst_loss(logp.detach().repeat(2), sampled, greedy, normalizer=10.)
        torch.testing.assert_close(long, loss.detach()*2)

    def test_zero_advantage_has_exactly_zero_policy_gradient(self):
        logp = torch.tensor([-.2, -.8], requires_grad=True)
        scst_loss(logp, -2., -2., normalizer=10.).backward()
        torch.testing.assert_close(logp.grad, torch.zeros(2))
        with self.assertRaises(ValueError):
            scst_loss(logp, float('nan'), 0., normalizer=10.)


if __name__ == '__main__':
    unittest.main()
