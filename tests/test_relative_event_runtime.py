import unittest

import torch

from beatfcn.model.event_canvas import RhythmEvent as E
from beatfcn.model.full_event_codec import EventRequest
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.model.relative_event_codec import encode_events, teacher_sequence
from beatfcn.model.relative_event_generate import generate_batch
from beatfcn.train.local_panel import collate
from scripts.probe_relative_head_runtime import cache_proof
from tests.test_full_event_model import inputs


class RelativeRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(28)
        torch.set_num_threads(2)
        c = FullEventConfig(d=32, heads=4, kv_heads=2, ffn=48,
            encoder_layers=1, decoder_layers=2, tap_dim=16, bands=4, dropout=0.)
        self.model = FullEventModel(c).eval()
        self.inputs = inputs(c)
        self.request = EventRequest('mania_4k', 8_000_000, 16_000_000, 20_000_000)

    def test_relative_teacher_cached_full_causal_logits(self):
        values = encode_events([E(8_004_000, 18_000_000, 'hold', 0),
            E(8_181_000, 8_181_000, 'tap', 1), E(13_996_000, 13_996_000, 'tap', 2)], self.request)
        seq = {k: torch.from_numpy(v)[None] for k, v in teacher_sequence(values, self.request).items()}
        with torch.no_grad():
            memory = self.model.encode(self.inputs)
            full = self.model.decode(*(seq[k] for k in ['tokens', 'fields', 'times', 'valid']), memory)
            cache = self.model.start_cache(memory)
            cached = torch.cat([self.model.decode(*(seq[k][:, i:i+1]
                for k in ['tokens', 'fields', 'times', 'valid']), memory, cache=cache)
                for i in range(len(values))], 1)
            torch.testing.assert_close(full, cached, atol=1e-4, rtol=1e-5)
            changed = {k: v.clone() for k, v in seq.items()}
            changed['tokens'][:, 7:] = 1
            changed['times'][:, 7:] += 10
            altered = self.model(self.inputs, changed)
            torch.testing.assert_close(full[:, :7], altered[:, :7], atol=1e-4, rtol=1e-5)

    def test_batched_generation_matches_scalar_with_finished_row(self):
        blocked = EventRequest('mania_4k', 8_000_000, 16_000_000, 20_000_000,
            known_carry=tuple(E(0, 18_000_000, 'hold', lane) for lane in range(4)))
        inp = {k: v.repeat(2, *([1]*(v.ndim-1))) for k, v in self.inputs.items()}
        batch = generate_batch(self.model, inp, [blocked, self.request], 32)
        singles = [generate_batch(self.model, self.inputs, [req], 32)[0]
                   for req in [blocked, self.request]]
        self.assertEqual(batch, singles)
        self.assertTrue(batch[0]['complete'])
        self.assertEqual(batch[0]['events'], [])

    def test_real_collator_padding_is_not_an_incremental_query(self):
        for events in [[], [E(8_181_000, 18_000_000, 'hold', 0)]]:
            with self.subTest(events=events):
                values = encode_events(events, self.request)
                inp, seq = collate([dict(inputs={k:v[0].numpy() for k,v in self.inputs.items()},
                    sequence=teacher_sequence(values, self.request))])
                self.assertGreater(seq['valid'].shape[1], len(values))
                proof = cache_proof(self.model, inp, seq, lambda: None)
                self.assertLessEqual(proof['maximum_logit_error'], 1e-4)
                self.assertEqual(proof['tokens'], len(values))


if __name__ == '__main__':
    unittest.main()
