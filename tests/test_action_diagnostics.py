import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from beatfcn.events.canonical import Event, LaneLoc
from beatfcn.model import features as F, sample
from beatfcn.model.action_diagnostics import oracle_onsets, actions_with_true_outer_history
from beatfcn.model.p1 import P1Model
from beatfcn.train.condition_pairs import blank_context_excerpt, without_conditions
from events_helpers import _chart, DURATION
from test_p1_model import CFG, song


class ActionDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        chart = _chart('mania_4k', [Event(str(i), 8_000_000+t, 8_000_000+t, 'tap', LaneLoc(i % 4), 'test')
                                   for i, t in enumerate([123000, 356000, 777000, 1001000])])
        request = blank_context_excerpt(chart, 8*F.SEC, DURATION, 'audio', 2)
        self.ex = without_conditions(F.build_core(request, chart.events, 8*F.SEC, DURATION))
        torch.manual_seed(13)
        self.model = P1Model(CFG).eval()
        self.song = song(0)
        with torch.no_grad():
            self.model.type_h.weight.zero_(); self.model.type_h.bias.copy_(torch.tensor([20., -20., -20., -20.]))
            self.model.cont.weight.zero_(); self.model.cont.bias.copy_(torch.tensor([20., -20.]))

    def test_real_generation_forces_only_requested_times_and_restores_methods(self):
        before = copy.deepcopy(self.model.state_dict()); rng = torch.get_rng_state().clone()
        with oracle_onsets(self.model, self.ex.groups) as state:
            groups = sample.generate_core(self.model, self.ex, self.song, [], [(8*F.SEC, 16*F.SEC)], [], greedy=True)
        self.assertEqual([g['t_rel_us'] for g in groups], [g['t_rel_us'] for g in self.ex.groups])
        self.assertTrue(state['ended']); self.assertTrue(state['diagnostic_only'])
        self.assertNotIn('time_logits', self.model.__dict__)
        self.assertNotIn('offset_logits', self.model.__dict__)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(before[key], value, rtol=0, atol=0)

    def test_masks_are_not_overridden_and_exception_restores(self):
        original = self.model.time_logits
        def blocked(*args, **kwargs):
            logits, mask, ok = original(*args, **kwargs)
            return logits, mask, torch.zeros_like(ok)
        with patch.object(self.model, 'time_logits', blocked):
            with self.assertRaisesRegex(ValueError, 'masked out'):
                with oracle_onsets(self.model, self.ex.groups):
                    sample.generate_core(self.model, self.ex, self.song, [], [(8*F.SEC, 16*F.SEC)], [], greedy=True)
            self.assertIs(self.model.time_logits, blocked)
        self.assertNotIn('time_logits', self.model.__dict__)

    def test_true_outer_history_is_causal_and_does_not_force_action_types(self):
        groups, info = actions_with_true_outer_history(self.model, self.ex, self.song)
        changed = copy.deepcopy(self.ex)
        changed.groups[1]['notes'][0]['lane'] = 3
        alternate, _ = actions_with_true_outer_history(self.model, changed, self.song)
        np.testing.assert_equal(groups[0], alternate[0])
        self.assertEqual(len(groups), 4)
        self.assertEqual(info['truth_supplied'], ['onsets', 'earlier_outer_groups'])
        with torch.no_grad():
            self.model.lane_h.weight.zero_(); self.model.lane_h.bias.copy_(torch.tensor([-20., -20., 20., -20.]))
        predicted, _ = actions_with_true_outer_history(self.model, self.ex, self.song)
        self.assertTrue(all(g['notes'][0]['lane'] == 2 for g in predicted))
        self.assertNotEqual(predicted[0]['notes'][0]['lane'], self.ex.groups[0]['notes'][0]['lane'])


if __name__ == '__main__':
    unittest.main()
