import unittest

import numpy as np

from scripts.audit_parallel_candidates import ordered_chord_candidates, ranks, tail_candidates


class CandidateAuditTest(unittest.TestCase):
    def test_ordered_chord_and_single_limit(self):
        logits = np.array([[0., 9., 8., 0.], [0., 10., 7., 6.]])
        choices, _ = ordered_chord_candidates(logits)
        self.assertEqual(choices[0].tolist(), [1, 2])
        self.assertTrue(np.all(np.diff(choices, axis=-1) > 0))
        self.assertEqual(len(choices), 6)
        single, _ = ordered_chord_candidates(logits[:1])
        self.assertEqual(single[0, 0], logits[0].argmax())
        self.assertEqual(ranks(np.array([[1., 1., 0., -1.]]), [1]).tolist(), [2])

    def test_tail_cartesian_search_and_legal_clock(self):
        block = np.array([-10., 6., 8.])
        cell = np.full(50, -30.); cell[15] = 10.
        ms = np.full(40, -30.); ms[8] = 10.
        result = tail_candidates(block, cell, ms, head_ms=2308, target_ms=2608)
        self.assertEqual(result['independent_argmax_ms'], 4608)
        self.assertEqual(result['exact_rank'], 2)
        self.assertEqual(result['within20ms_rank'], 2)
        self.assertEqual(result['top5_tail_ms'][:2], [4608, 2608])
        self.assertTrue(all(t > 2308 for t in result['top5_tail_ms']))
        with self.assertRaises(ValueError):
            tail_candidates(block, cell, ms, head_ms=2608, target_ms=2608)


if __name__ == '__main__':
    unittest.main()
