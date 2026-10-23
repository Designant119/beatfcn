import unittest
import numpy as np
from beatfcn.model.control_eval import bands_around, summarize, spearman
from beatfcn.events import controls as C


class ControlEvalTests(unittest.TestCase):
    def test_bands_around_are_distinct_clamped_edges(self):
        b = bands_around('density_notes_per_second', 3.5)
        self.assertEqual(len(b), 3)
        self.assertEqual(b[1], (3.0, 4.0))
        self.assertTrue(all(lo < hi for lo, hi in b))
        self.assertEqual(bands_around('hold_head_ratio', 0.0)[0], (0.0, 0.02))  # clamped at the bottom edge

    def test_summary_separates_responsive_from_flat_controls(self):
        rng = np.random.default_rng(0)
        recs = []
        for core in range(20):
            for band in bands_around('density_notes_per_second', 4.5):
                mid = (band[0] + band[1]) / 2
                recs.append({'field': 'density_notes_per_second', 'core': core, 'requested': band,
                             'achieved': mid + rng.normal(0, 0.2), 'legal': True})
                recs.append({'field': 'chord_ratio', 'core': core, 'requested': (0.1, 0.2) if band[0] < 4 else (0.3, 0.4),
                             'achieved': 0.25 + rng.normal(0, 0.01), 'legal': core % 7 != 0})
        s = summarize(recs)
        self.assertTrue(s['density_notes_per_second']['usable'])
        self.assertGreater(s['density_notes_per_second']['spearman_requested_vs_achieved'], 0.8)
        self.assertFalse(s['chord_ratio']['usable'])
        self.assertEqual(s['chord_ratio']['illegal_cores'], 3 * 3)  # cores 0, 7, 14 x 3 bands
        self.assertIsNone(spearman([1, 1, 1], [1, 2, 3]))


if __name__ == '__main__':
    unittest.main()
