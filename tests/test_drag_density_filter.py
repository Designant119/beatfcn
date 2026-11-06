import unittest
from collections import Counter

import numpy as np

from beatfcn.train.drag_density_filter import dense_drag_mask as current_dense_drag_mask


def dense_drag_mask(times, kinds):
    # These fixtures exercise the historical 250ms/8-drag recipe explicitly;
    # production defaults have changed and must not change their test meaning.
    return current_dense_drag_mask(times, kinds, window_us=250_000, minimum_drag_objects=8)


class DragDensityFilterTest(unittest.TestCase):
    def test_half_open_boundary(self):
        times = np.array([0, 10, 20, 30, 40, 50, 60, 250_000])
        removed, _ = dense_drag_mask(times, np.full(8, 3))
        self.assertFalse(removed.any())
        times[-1] = 249_999
        removed, _ = dense_drag_mask(times, np.full(8, 3))
        self.assertTrue(removed.all())

    def test_only_drag_in_union_and_no_dedup(self):
        times = np.array([0]*8 + [1, 2, 3, 1_000_000])
        kinds = np.array([3]*8 + [0, 1, 2, 3])
        removed, _ = dense_drag_mask(times, kinds)
        self.assertEqual(np.flatnonzero(removed).tolist(), list(range(8)))
        self.assertEqual(Counter(kinds[~removed]), Counter([0, 1, 2, 3]))

    def test_detection_before_slicing_and_order_invariant(self):
        times = np.arange(8)*20_000+900_000
        kinds = np.full(8, 3)
        removed, _ = dense_drag_mask(times, kinds)
        self.assertTrue(removed[:5].all())
        local, _ = dense_drag_mask(times[:5], kinds[:5])
        self.assertFalse(local.any())
        order = np.array([7, 0, 4, 1, 2, 5, 6, 3])
        shuffled, _ = dense_drag_mask(times[order], kinds[order])
        self.assertTrue(shuffled.all())


if __name__ == '__main__':
    unittest.main()
