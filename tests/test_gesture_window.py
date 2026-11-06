import unittest

import numpy as np

from beatfcn.train.gesture_window import fit_linear, onset_and_window, recall_of, roll_mel


class GestureWindowTests(unittest.TestCase):
    def test_window_reads_frames_after_the_onset_and_ignores_a_longer_label(self):
        mel = np.zeros((8, 2), np.float32)
        mel[1:4] = (4, 0)
        times = np.arange(8) * 0.01
        frame, window = onset_and_window(mel, times, 0.0, 4)
        np.testing.assert_allclose(frame, [0, 0])
        np.testing.assert_allclose(window, [3, 0])
        longer = mel.copy()
        longer[4:] = (100, 100)
        _, capped = onset_and_window(longer, times, 0.0, 4)
        np.testing.assert_allclose(capped, window)

    def test_time_shift_changes_the_window_and_recall_uses_that_difference(self):
        mel = np.zeros((6, 1), np.float32)
        mel[0, 0] = 1
        shifted = roll_mel(mel, 0.5)
        self.assertEqual(int(shifted.shape[0] * 0.5), 3)
        self.assertNotEqual(float(shifted[0, 0]), float(mel[0, 0]))
        values = np.array([0, 0, 0, 1, 1, 1, 1, 0], np.float32)[:, None]
        labels = (values[:, 0] > 0).astype(np.int64)
        train = np.arange(8) < 4
        test = ~train
        onset = np.zeros_like(values)
        weight, bias, _ = fit_linear(values, labels, train, 2, 80, 20261025, 0.2)
        held = recall_of(values, labels, test, weight, bias, 1)
        flat_w, flat_b, _ = fit_linear(onset, labels, train, 2, 80, 20261025, 0.2)
        missed = recall_of(onset, labels, test, flat_w, flat_b, 1)
        self.assertEqual(held['support'], 3)
        self.assertEqual(held['recall'], 1.0)
        self.assertEqual(missed['recall'], 0.0)
        moved = recall_of(roll_mel(values, 0.5), labels, test, weight, bias, 1)
        self.assertLess(moved['recall'], held['recall'])
