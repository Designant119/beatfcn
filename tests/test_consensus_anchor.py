import unittest

from scripts.run_workflow_benchmark import consensus_draft


def roll(lanes, step=100_000):
    return [(i * step, i * step, 'tap', l) for i, l in enumerate(lanes)]


class AnchorConsensusTest(unittest.TestCase):
    def test_anchor_keeps_one_hand_flow(self):
        # three samples, same onsets; per position the lanes disagree so majority stitches a different flow
        a = roll([0, 1, 2, 3, 0, 1, 2, 3])
        b = roll([0, 1, 2, 3, 0, 1, 2, 3])
        c = roll([3, 2, 1, 0, 3, 2, 1, 0])
        notes, info = consensus_draft([c, a, b], 'mania_4k', 10_000_000, pattern_source='anchor')
        self.assertEqual([n[3] for n in notes], [0, 1, 2, 3, 0, 1, 2, 3])
        self.assertEqual(info['pattern_source'], 'anchor')
        self.assertEqual(info['patterns_from_anchor'], 8)

    def test_anchor_keeps_holds_the_vote_splits(self):
        # anchor writes an LN; the other two disagree with each other (tap on two different lanes)
        anchor = [(0, 400_000, 'hold', 1), (500_000, 500_000, 'tap', 2)]
        x = [(0, 0, 'tap', 1), (500_000, 500_000, 'tap', 2)]
        y = [(0, 0, 'tap', 3), (500_000, 500_000, 'tap', 2)]
        notes, info = consensus_draft([anchor, x, y], 'mania_4k', 10_000_000, pattern_source='anchor')
        self.assertIn('hold', [n[2] for n in notes])
        maj, _ = consensus_draft([anchor, x, y], 'mania_4k', 10_000_000)
        self.assertEqual(len(notes), len(maj))

    def test_majority_default_unchanged(self):
        a = roll([0, 1, 2, 3]); b = roll([0, 1, 2, 3]); c = roll([3, 2, 1, 0])
        n1, _ = consensus_draft([a, b, c], 'mania_4k', 10_000_000)
        n2, _ = consensus_draft([a, b, c], 'mania_4k', 10_000_000, pattern_source='majority')
        self.assertEqual(n1, n2)

    def test_anchor_onsets_keep_one_rhythm(self):
        a = roll([0, 1, 2, 3, 0, 1]); b = roll([0, 1, 2, 3, 0, 1]); c = [(50_000+n[0], 50_000+n[1], n[2], n[3]) for n in a]
        pooled, _ = consensus_draft([a, b, c], 'mania_4k', 10_000_000, pattern_source='anchor', min_votes=1)
        kept, _ = consensus_draft([a, b, c], 'mania_4k', 10_000_000, pattern_source='anchor', min_votes=1, onset_source='anchor')
        self.assertLess(len(kept), len(pooled))
        self.assertEqual([n[0] for n in kept], [n[0] for n in a])

    def test_fragment_guard_drops_a_lone_weak_squeeze(self):
        grid = roll([0, 1, 2, 3]*8, step=400_000)
        extra = [(3_200_000+60_000, 3_200_000+60_000, 'tap', 0)]
        samples = [grid+extra, grid+extra, grid+extra, grid, grid, grid, grid, grid]
        loose, _ = consensus_draft(samples, 'mania_4k', 20_000_000, min_votes=3, window_us=30_000)
        guarded, _ = consensus_draft(samples, 'mania_4k', 20_000_000, min_votes=3, window_us=30_000, fragment_guard=True)
        self.assertEqual(len(loose), len(guarded)+1)

    def test_fragment_guard_keeps_a_fast_run(self):
        run = [(k*100_000, k*100_000, 'tap', k % 4) for k in range(40)]
        guarded, _ = consensus_draft([run]*3, 'mania_4k', 20_000_000, min_votes=2, window_us=30_000, fragment_guard=True)
        self.assertEqual(len(guarded), 40)

    def test_lone_key_repeat_under_40ms_is_dropped(self):
        # every sample repeats one key after 30 ms with no free pattern to re-choose: the second press cannot be played
        a = [(0, 0, 'tap', 1), (30_000, 30_000, 'tap', 1)]
        notes, info = consensus_draft([a, a, a], 'mania_4k', 10_000_000, min_jack_us=100_000, window_us=10_000)
        repeats = [y[0] - x[0] for x, y in zip(notes, notes[1:]) if x[3] == y[3]]
        self.assertTrue(all(gap >= 40_000 for gap in repeats))

    def test_repeat_rule_needs_min_jack(self):
        a = [(0, 0, 'tap', 1), (30_000, 30_000, 'tap', 1)]
        notes, info = consensus_draft([a, a, a], 'mania_4k', 10_000_000, window_us=10_000)
        self.assertEqual(len(notes), 2)
        self.assertEqual(info['unplayable_repeats_dropped'], 0)


if __name__ == '__main__':
    unittest.main()
