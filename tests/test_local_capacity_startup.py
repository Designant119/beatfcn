import unittest
import torch

from scripts.prepare_local_capacity import state_difference, width_only


class CapacityStartupTests(unittest.TestCase):
    def test_capacity_change_cannot_hide_objective_input_change(self):
        small = dict(d=128, ffn=384, mel_bins=128, dropout=0)
        large = dict(small, d=256, ffn=768)
        width_only(small, large)
        with self.assertRaises(ValueError):
            width_only(small, dict(large, mel_bins=0))

    def test_missing_adam_moment_is_rejected(self):
        old = dict(state={0: dict(step=torch.tensor(2.), exp_avg=torch.ones(3))})
        with self.assertRaises(ValueError):
            state_difference(dict(state={0: dict(step=torch.tensor(2.))}), old)

    def test_mask_and_target_differences_cannot_disappear(self):
        old = dict(valid=torch.tensor([True, False]), target=torch.tensor([2, 0]))
        self.assertTrue(state_difference(old, old)['exact'])
        self.assertEqual(state_difference(dict(old, valid=torch.tensor([True, True])), old)['max_abs'], 1.)
        with self.assertRaises(ValueError):
            state_difference(dict(old, target=torch.tensor([2])), old)


if __name__ == '__main__':
    unittest.main()
