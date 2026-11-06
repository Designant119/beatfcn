import unittest

import numpy as np

from beatfcn.model.full_event_codec import EventRequest
from beatfcn.model.rerank import score_candidate, select_best, short_gap_fraction


def events(onsets):
    return [dict(onset_us=int(t), end_us=int(t), type='tap', lane=-1) for t in onsets]


class RerankTests(unittest.TestCase):
    def setUp(self):
        self.request = EventRequest('phira', 0, 8_000_000, 60_000_000)
        self.times = np.arange(0, 8, .01)
        self.probs = np.zeros_like(self.times)
        self.beats = np.arange(0, 8_000_000, 500_000)
        for t in self.beats:
            self.probs[int(t // 10_000)] = .9

    def test_template_cluster_loses_to_onset_aligned_output(self):
        template = dict(complete=True, events=events([0, 12_000, 15_000, 18_000, 32_000, 37_000, 38_000, 39_000]))
        aligned = dict(complete=True, events=events(self.beats))
        best, scores, parts = select_best([template, aligned], self.request,
            onset_probs=self.probs, onset_times_s=self.times)
        self.assertEqual(best, 1)
        self.assertGreater(parts[0]['short_gap'], .8)

    def test_incomplete_and_empty_are_penalised(self):
        aligned = dict(complete=True, events=events(self.beats))
        empty = dict(complete=True, events=[])
        truncated = dict(complete=False, events=events(self.beats))
        best, scores, _ = select_best([empty, truncated, aligned], self.request,
            onset_probs=self.probs, onset_times_s=self.times)
        self.assertEqual(best, 2)
        self.assertLess(scores[1], scores[2] - 5)

    def test_density_request_is_followed_without_onset_head(self):
        curve = [dict(start_us=0, end_us=8_000_000, density=4.)]
        sparse = dict(complete=True, events=events(np.arange(0, 8_000_000, 1_000_000)))
        matched = dict(complete=True, events=events(np.arange(0, 8_000_000, 250_000)))
        best, _, parts = select_best([sparse, matched], self.request, curve=curve)
        self.assertEqual(best, 1)
        self.assertAlmostEqual(parts[1]['density'], 0.)

    def test_never_reads_targets(self):
        score, parts = score_candidate(dict(complete=True, events=events([0])), self.request)
        self.assertEqual(set(parts), {'incomplete', 'short_gap', 'density', 'onset', 'count'})
        self.assertEqual(short_gap_fraction(events([0, 10_000])), 1.)


if __name__ == '__main__':
    unittest.main()
