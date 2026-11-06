import unittest

import torch

from beatfcn.model.candidate_time import head_ms_accuracy, head_ms_logits
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import FIELDS, EventRequest, encode_events, teacher_sequence
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from tests.test_full_event_model import inputs


class CandidateTimeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261025)
        self.c = FullEventConfig(d=32, heads=4, kv_heads=2, ffn=48,
            encoder_layers=1, decoder_layers=2, tap_dim=16, bands=4, dropout=0., mel_bins=8)
        self.model = FullEventModel(self.c).eval()
        self.inputs = inputs(self.c)
        self.inputs['mel'] = torch.randn(1, 12, 8)
        self.inputs['mel_times'] = torch.arange(12)[None].float() * .01
        self.inputs['mel_valid'] = torch.ones(1, 12, dtype=torch.bool)
        req = EventRequest('phira', 0, 8_000_000, 10_000_000)
        tokens = encode_events([RhythmEvent(40_000, 40_000, 'tap'),
                                 RhythmEvent(80_000, 120_000, 'hold'),
                                 RhythmEvent(80_000, 80_000, 'flick')], req)
        self.seq = {k: torch.from_numpy(v)[None] for k, v in teacher_sequence(tokens, req).items()}

    def test_shared_query_matches_replay_and_candidate_time_moves_only_head_ms(self):
        with torch.no_grad():
            ordinary = self.model(self.inputs, self.seq)
            shared = head_ms_logits(self.model, self.inputs, self.seq, per_candidate_time=False)
            moved = head_ms_logits(self.model, self.inputs, self.seq, per_candidate_time=True)
        field = FIELDS.index('head_ms')
        head = self.seq['fields'][0] == field
        self.assertTrue(bool(head.any()))
        self.assertLess((ordinary[0, ~head] - shared[0, ~head]).abs().max().item(), 1e-4)
        self.assertGreater((shared[0, head] - moved[0, head]).abs().max().item(), 1e-5)
        self.assertLess((shared[0, ~head] - moved[0, ~head]).abs().max().item(), 1e-5)
        report = head_ms_accuracy(shared, self.seq, self.model.vocab)
        self.assertGreater(report['count'], 0)
        self.assertTrue(torch.isfinite(shared).all())

    def test_candidate_branch_receives_audio_gradient(self):
        self.model.train()
        loss = event_loss(head_ms_logits(self.model, self.inputs, self.seq, per_candidate_time=True), self.seq)
        loss.backward()
        grad = self.model.mel_projection.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0.)
