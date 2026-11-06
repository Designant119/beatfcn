import unittest

import torch

from beatfcn.model.fullstack_mamba import FullStackMamba, FullStackMambaConfig
from beatfcn.model.full_event_codec import EventRequest, encode_events, teacher_sequence
from beatfcn.model.event_canvas import RhythmEvent


class FullStackMambaTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.model = FullStackMamba(FullStackMambaConfig(d=16, ffn=32, encoder_layers=1,
            decoder_layers=2, bands=2, tap_dim=8, mel_bins=4, d_state=8, headdim=8, chunk_size=8)).eval()
        self.inputs = dict(local=torch.randn(1, 3, 7, 2, 8), local_valid=torch.tensor([[1] * 5 + [0] * 2], dtype=torch.bool),
            local_times=torch.arange(7)[None] * .04, mel=torch.randn(1, 28, 4),
            mel_valid=torch.tensor([[1] * 20 + [0] * 8], dtype=torch.bool), mel_times=torch.arange(28)[None] * .01,
            song=torch.randn(1, 11, 8), song_valid=torch.tensor([[1] * 8 + [0] * 3], dtype=torch.bool),
            song_times=torch.arange(11)[None] * .5, reference=torch.randn(1, 3, 16),
            reference_valid=torch.tensor([[1, 1, 0]], dtype=torch.bool), reference_times=torch.arange(3)[None].float(),
            controls=torch.zeros(1, 8, 6), mode=torch.ones(1, dtype=torch.long))
        req = EventRequest('phira', 0, 1_000_000, 2_000_000)
        values = teacher_sequence(encode_events([RhythmEvent(100_000, 300_000, 'hold', -1),
                                                RhythmEvent(600_000, 600_000, 'flick', -1)], req), req)
        self.seq = {k: torch.from_numpy(v)[None] for k, v in values.items()}

    def test_full_stack_cache_causality_and_request_reset(self):
        with torch.no_grad():
            memory = self.model.encode(self.inputs)
            full = self.model(self.inputs, self.seq)
            cache = self.model.start_cache(memory)
            parts = [self.model.decode(*(self.seq[k][:, i:i + 1] for k in ('tokens', 'fields', 'times', 'valid')),
                     memory, cache=cache) for i in range(self.seq['tokens'].shape[1])]
            torch.testing.assert_close(full, torch.cat(parts, 1), atol=2e-5, rtol=2e-5)
            altered = {k: v.clone() for k, v in self.seq.items()}
            altered['tokens'][:, 8:] = 1
            altered['times'][:, 8:] += 100
            torch.testing.assert_close(full[:, :8], self.model(self.inputs, altered)[:, :8], atol=1e-6, rtol=1e-6)
            fresh = self.model.start_cache(memory)
            one = self.model.decode(*(self.seq[k][:, :1] for k in ('tokens', 'fields', 'times', 'valid')), memory, cache=fresh)
            torch.testing.assert_close(parts[0], one)

    def test_masked_context_padding_is_never_scanned_or_exposed(self):
        changed = {k: v.clone() for k, v in self.inputs.items()}
        for key in ('mel', 'song', 'reference'):
            changed[key][~changed[key + '_valid']] = 1e5
            changed[key + '_times'][~changed[key + '_valid']] = 1e5
        changed['local'][:, :, 5:] = 1e5
        changed['local_times'][:, 5:] = 1e5
        with torch.no_grad():
            torch.testing.assert_close(self.model(self.inputs, self.seq), self.model(changed, self.seq), atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
