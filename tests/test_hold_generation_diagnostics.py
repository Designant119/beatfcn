from dataclasses import replace
from collections import Counter
from unittest.mock import patch
import unittest

import numpy as np
import torch

from beatfcn.model import features as F
from beatfcn.model import sample
from beatfcn.model import p1
from beatfcn.model.p1 import P1Model, endpoint_interval_mask
from beatfcn.events.requests import build_request
from events_helpers import synthetic_mania, synthetic_phira, DURATION
from test_p1_model import CFG, song


class HoldEndpointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = P1Model(replace(CFG, dropout=0.)).eval()
        with torch.no_grad():
            self.model.blk_q.weight.zero_()
            if self.model.blk_q.bias is not None:
                self.model.blk_q.bias.zero_()
            self.model.blk_off.weight.zero_()

    def endpoint(self, onset, hi):
        with torch.no_grad():
            return sample._hold_end(self.model, torch.zeros(1, CFG.gru), song(0), onset,
                                    {'lane': 0}, [], [(0, hi)],
                                    torch.Generator().manual_seed(0), 1., 1., True, 'cpu')

    def test_block_without_a_strictly_later_ms_endpoint_is_not_selected(self):
        # Argmax prefers block 0 if admitted. Its last ms equals the hold onset;
        # block 1 contains legal ends, so a failed proposal is a masking bug.
        end = self.endpoint(F.BLOCK_US - F.MS, 2 * F.BLOCK_US)
        self.assertIsNotNone(end, 'valid hold silently becomes tap at the block boundary')
        self.assertGreaterEqual(end, F.BLOCK_US)

    def test_sub_ms_copy_onset_can_end_on_next_ms_grid_point(self):
        end = self.endpoint(F.BLOCK_US - F.MS // 2, F.BLOCK_US)
        self.assertEqual(end, F.BLOCK_US)

    def test_truly_empty_endpoint_interval_returns_none(self):
        self.assertIsNone(self.endpoint(F.BLOCK_US - F.MS // 2, F.BLOCK_US - 1))

    def test_interval_mask_matches_enumerated_grid(self):
        rng = np.random.default_rng(5)
        for width in (F.CELL_US, F.BLOCK_US):
            for _ in range(100):
                start = int(rng.integers(0, 5)) * width
                lo = start + int(rng.integers(-2000, width + 2000))
                hi = lo + int(rng.integers(-1, 2000))
                expected = any(lo <= t <= hi for t in range(start, start + width, F.MS))
                self.assertEqual(bool(endpoint_interval_mask(torch.tensor(start), width, lo, hi)), expected)

    def test_training_uses_the_same_representable_block_mask(self):
        bt = {'mode': torch.tensor([0]), 'hold_onset_us': torch.tensor([F.BLOCK_US-F.MS]),
              'hold_ok_lo': torch.tensor([F.BLOCK_US-F.MS+1]), 'hold_ok_hi': torch.tensor([2*F.BLOCK_US]),
              'hold_block': torch.tensor([1]), 'hold_cell': torch.tensor([0]), 'hold_off': torch.tensor([0]),
              'hold_blocks': torch.zeros(1, F.MAX_END_BLOCKS, 8, CFG.summary_dim),
              'hold_detail': torch.zeros(1, F.BLOCK_US//F.CELL_US, 8, CFG.tap_dim)}
        masks = {}
        original = p1.masked_ce
        def checked(logits, target, valid, mask, name, **kw):
            masks[name] = mask.clone()
            return original(logits, target, valid, mask, name, **kw)
        with patch.object(p1, 'masked_ce', side_effect=checked):
            self.model._hold(bt, torch.tensor([0]), torch.zeros(1, CFG.gru), {})
        self.assertFalse(bool(masks['end_block'][0, 0]))
        self.assertTrue(bool(masks['end_block'][0, 1]))

    def test_decision_diagnostics_preserve_outputs_and_account_for_holds(self):
        for chart in (synthetic_mania(3), synthetic_phira(4)):
            request = build_request(chart, kind='from_scratch', seed=9, audio_id='a', audio_duration_us=DURATION)
            ex = F.build_core(request, chart.events, 0, DURATION)
            holds, runs, heads = sample.core_context(request, ex, F.line_index_of(chart.events))
            with torch.no_grad():
                self.model.type_h.weight.zero_()
                self.model.type_h.bias.fill_(-5.)
                self.model.type_h.bias[1] = 5.
            original = self.model.time_logits
            original_end = sample._hold_end
            for fail_ends in (False, True):
                outputs = []
                for with_diagnostics in (False, True):
                    calls = []
                    def pointer(*args, **kwargs):
                        logits, mask, ok = original(*args, **kwargs)
                        logits = torch.full_like(logits, -1e9)
                        available = torch.where(mask[0, 0, 1:201])[0]
                        choice = int(available[min(10, len(available)-1)])+1 if len(calls) < 3 and len(available) else 0
                        calls.append(choice)
                        logits[0, 0, choice] = 0
                        return logits, mask, ok
                    def ending(*args, **kwargs):
                        if fail_ends:
                            args = list(args)
                            args[6] = [(args[3], args[3]+1)]
                        return original_end(*args, **kwargs)
                    diagnostics = {} if with_diagnostics else None
                    with patch.object(self.model, 'time_logits', side_effect=pointer), patch.object(sample, '_hold_end', side_effect=ending):
                        groups = sample.generate_core(self.model, ex, song(0), holds, runs, heads,
                                                      greedy=True, diagnostics=diagnostics)
                    outputs.append([(g['t_rel_us'], g['notes']) for g in groups])
                self.assertEqual(outputs[0], outputs[1])
                d = diagnostics['note_decisions']
                emitted = Counter(n['type'] for g in groups for n in g['notes'])
                self.assertGreater(d['type_decisions'], 0)
                self.assertEqual(d['emitted_by_type'], [emitted[i] for i in range(4)])
                self.assertEqual(d['proposed_by_type'][1], d['hold_end_success']+sum(d['hold_end_failures'].values()))
                self.assertEqual(d['hold_to_tap']+d['hold_stopped_group'], sum(d['hold_end_failures'].values()))
                self.assertAlmostEqual(sum(d['legal_probability_sum']), d['type_decisions'], places=5)
                if fail_ends:
                    self.assertGreater(d['hold_to_tap'], 0)
                    self.assertEqual(d['hold_end_success'], 0)


if __name__ == '__main__':
    unittest.main()
