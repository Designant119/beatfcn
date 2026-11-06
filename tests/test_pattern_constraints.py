"""Decode-time mania pattern constraints (full_event_generate pattern_constraints)."""
from collections import defaultdict
import unittest

import torch

from beatfcn.model.full_event_codec import pattern_digits
from beatfcn.model.full_event_generate import _pattern_rule, generate_batch
from beatfcn.model.full_event_model import FullEventModel
from test_fast_generate import both, hold_heavy, make_inputs, requests_mixed, tiny


def chord_heavy(model):
    """Bias toward big chords and short gaps so both constraints have work to do."""
    v = model.vocab
    bias = torch.zeros(v.size)
    for code in range(1, 81):
        bias[v.token('mania', code)] += .9 * sum(d > 0 for d in pattern_digits(code))
    bias[v.token('group', 0)] = -3.
    model.output.register_forward_hook(lambda m, a, out: out + bias.to(out.device))
    return model


class PatternConstraintTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.c = tiny(qk_norm=True, mel_bins=8, intensity_dim=3, timed_intensity=True)
        self.model = chord_heavy(hold_heavy(FullEventModel(self.c).eval()))
        self.requests = requests_mixed()
        self.inputs = make_inputs(self.c, batch=4)

    def gens(self, seed=0):
        return [torch.Generator().manual_seed(100 * seed + i) for i in range(4)]

    def test_rule_validation(self):
        self.assertIsNone(_pattern_rule(None))
        self.assertIsNone(_pattern_rule({}))
        self.assertEqual(_pattern_rule(dict(max_chord=2)), dict(max_chord=2))
        for bad in (dict(max_chord=0), dict(max_chord=5), dict(min_same_lane_ms=0), dict(oops=1)):
            with self.assertRaises(ValueError):
                _pattern_rule(bad)

    def test_off_is_identical(self):
        for fast in (False, True):
            a = generate_batch(self.model, self.inputs, self.requests, 300, fast=fast, temperature=1., generator=self.gens())
            b = generate_batch(self.model, self.inputs, self.requests, 300, fast=fast, temperature=1., generator=self.gens(),
                               pattern_constraints=None)
            self.assertEqual(a, b)

    def test_constraints_hold_and_fast_equals_reference(self):
        rule = dict(max_chord=2, min_same_lane_ms=200)
        free = generate_batch(self.model, self.inputs, self.requests, 400, temperature=1., generator=self.gens(1))
        groups = defaultdict(int)
        for row in free:
            for e in row['events']:
                if e['lane'] >= 0:
                    groups[(id(row), e['onset_us'])] += 1
        self.assertTrue(any(n > 2 for n in groups.values()), 'fixture produces chords > 2 without constraints')
        ref, fast = both(self.model, self.inputs, self.requests, 400, generator=self.gens(1), temperature=1.,
                         pattern_constraints=rule)
        self.assertEqual(ref, fast)
        for request, row in zip(self.requests, ref):
            if request.mode != 'mania_4k':
                continue
            counts = row['decoding']['pattern_constraints']['counts'] if 'decoding' in row else {}
            by_onset, last = defaultdict(int), {}
            for e in sorted(row['events'], key=lambda e: (e['onset_us'], e['lane'])):
                by_onset[e['onset_us']] += 1
                carry = any(c.lane == e['lane'] and c.onset_us == e['onset_us'] for c in request.known_carry)
                if e['lane'] in last and not carry and not counts.get('relaxed'):
                    self.assertGreaterEqual(e['onset_us'] - last[e['lane']], 200_000)
                last[e['lane']] = e['onset_us']
            self.assertTrue(all(n <= 2 for n in by_onset.values()))


if __name__ == '__main__':
    unittest.main()


class ChunkedEncodeTest(unittest.TestCase):
    def test_chunked_encode_matches_full_batch(self):
        from beatfcn.model import full_event_generate as gen
        torch.manual_seed(3)
        c = tiny(qk_norm=True, mel_bins=8, intensity_dim=3, timed_intensity=True)
        model = FullEventModel(c).eval()
        inputs = make_inputs(c, batch=10)
        with torch.no_grad():
            full = model.encode(inputs)
            old = gen.ENCODE_CHUNK_ROWS
            try:
                gen.ENCODE_CHUNK_ROWS = 3
                chunked = gen._encode(model, inputs)
            finally:
                gen.ENCODE_CHUNK_ROWS = old
        self.assertEqual(set(full), set(chunked))
        for k, v in full.items():
            if torch.is_tensor(v):
                torch.testing.assert_close(chunked[k], v, rtol=1e-5, atol=1e-5)
