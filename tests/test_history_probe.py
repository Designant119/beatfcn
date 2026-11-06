import copy
from dataclasses import replace
import unittest

import torch

from beatfcn.model.history_probe import history_intervention, history_probe
from beatfcn.model.p1 import P1Model
from test_p1_model import CFG, batch


class HistoryProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(13)
        self.bt = batch(1, seed=3)
        self.i = int(self.bt['grp_valid'][0].sum()) // 2
        self.model = P1Model(replace(CFG, lane_history_groups=8)).eval()
        with torch.no_grad():
            self.model.lane_history_y.weight.normal_(std=.02)
            self.model.lane_history_gru.weight.normal_(std=.02)

    def test_intervention_preserves_labels_audio_and_constraints(self):
        before = copy.deepcopy(self.bt)
        with history_intervention(self.model, self.bt, self.i, 'past_lane_memory_unknown') as view:
            for name in self.bt:
                if torch.is_tensor(self.bt[name]) and name not in ('known', 'blk_grid'):
                    torch.testing.assert_close(view[name], self.bt[name], atol=0, rtol=0)
            self.assertTrue(torch.equal(self.model.lane_history_y(torch.ones(1, 48)), torch.zeros(1, CFG.d)))
        for name in before:
            if torch.is_tensor(before[name]):
                torch.testing.assert_close(before[name], self.bt[name], atol=0, rtol=0)

    def test_split_path_probe_and_model_restoration(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        result = history_probe(self.model, self.bt, self.i)['conditions']
        self.assertEqual(result['k8_gru_off']['outer_state_rms_change'], 0)
        self.assertGreater(result['k8_gru_off']['gru_state_rms_change'], 0)
        self.assertGreater(result['outer_group_lanes_unknown']['outer_state_rms_change'], 0)
        self.assertEqual(result['baseline']['lane_probability_total_variation'], 0)
        for key, value in before.items():
            torch.testing.assert_close(value, self.model.state_dict()[key], atol=0, rtol=0)

    def test_outer_intervention_does_not_change_input_target_embedding(self):
        original = self.model.group_embedding(self.bt)
        with history_intervention(self.model, self.bt, self.i, 'outer_group_lanes_unknown') as view:
            changed = self.model.group_embedding(view)
            self.assertGreater(float((changed[:, :self.i] - original[:, :self.i]).abs().max()), 0)
            torch.testing.assert_close(changed[:, self.i:], original[:, self.i:])
        torch.testing.assert_close(self.model.group_embedding(self.bt), original)

    def test_failure_removes_hooks_and_method_overrides(self):
        original = self.model.group_embedding(self.bt)
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with history_intervention(self.model, self.bt, self.i, 'past_lane_memory_unknown'):
                raise RuntimeError('injected')
        torch.testing.assert_close(self.model.group_embedding(self.bt), original)
        self.assertEqual(len(self.model.lane_history_y._forward_hooks), 0)
        self.assertEqual(len(self.model.lane_history_gru._forward_hooks), 0)

    def test_compacted_decoder_scores_same_targets(self):
        first = history_probe(self.model, self.bt, self.i)
        self.model.c = replace(self.model.c, compact_inner=True)
        second = history_probe(self.model, self.bt, self.i)
        for name in first['conditions']:
            self.assertAlmostEqual(first['conditions'][name]['target_nll'], second['conditions'][name]['target_nll'], places=5)


if __name__ == '__main__':
    unittest.main()
