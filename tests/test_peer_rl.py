import tempfile
import unittest
from unittest.mock import patch

import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest, EventVocabulary, encode_events
from beatfcn.train.peer_rl import PeerRL, peer_faults, rule_faults, score_window, walk_groups
from tests.test_evolve_finetune import Run

MS = 1000
REQUEST = EventRequest('phira', 0, 8_000*MS, 10_000*MS)
VOCAB = EventVocabulary()


def tokens(*events):
    return encode_events([RhythmEvent(*e) for e in events], REQUEST, vocab=VOCAB)


class PeerRuleTest(unittest.TestCase):
    def rules(self, *events):
        return [rule for _, rule in rule_faults(walk_groups(tokens(*events), REQUEST, VOCAB))]

    def test_clean_chart_has_no_fault(self):
        self.assertEqual(self.rules((100*MS, 100*MS, 'tap', -1), (500*MS, 900*MS, 'hold', -1), (500*MS, 500*MS, 'tap', -1)), [])

    def test_each_rule_fires_on_its_own_action(self):
        self.assertEqual(self.rules((100*MS, 100*MS, 'tap', -1), (130*MS, 130*MS, 'tap', -1)), ['close_onset'])
        self.assertEqual(self.rules(*[(100*MS, 100*MS, 'tap', -1)]*3), ['third_finger'])
        self.assertEqual(self.rules((100*MS, 100*MS, 'flick', -1), (100*MS, 100*MS, 'drag', -1)), ['mixed_special'])
        self.assertEqual(self.rules((100*MS, 900*MS, 'hold', -1), (150*MS, 900*MS, 'hold', -1)), ['close_onset', 'stacked_hold'])

    def test_positions_point_at_the_deciding_token(self):
        t = tokens((100*MS, 100*MS, 'flick', -1), (100*MS, 100*MS, 'drag', -1))
        position, _ = rule_faults(walk_groups(t, REQUEST, VOCAB))[0]
        self.assertEqual(VOCAB.decision(t[position])[0], 'kind')
        t = tokens((100*MS, 100*MS, 'tap', -1), (130*MS, 130*MS, 'tap', -1))
        position, _ = rule_faults(walk_groups(t, REQUEST, VOCAB))[0]
        self.assertEqual(VOCAB.decision(t[position])[0], 'head_cell')

    def test_support_uses_the_other_samples_only(self):
        a = walk_groups(tokens((100*MS, 100*MS, 'tap', -1), (900*MS, 900*MS, 'tap', -1)), REQUEST, VOCAB)
        b = walk_groups(tokens((110*MS, 110*MS, 'tap', -1)), REQUEST, VOCAB)
        faults = peer_faults([a, b])
        self.assertEqual(len(faults[0]), 1)
        self.assertEqual(faults[1], [])
        self.assertEqual(VOCAB.decision(tokens((100*MS, 100*MS, 'tap', -1), (900*MS, 900*MS, 'tap', -1))[faults[0][0][0]])[0], 'head_cell')

    def test_min_support_matches_the_product_vote_rule(self):
        one = walk_groups(tokens((100*MS, 100*MS, 'tap', -1)), REQUEST, VOCAB)
        none = walk_groups(tokens((900*MS, 900*MS, 'tap', -1)), REQUEST, VOCAB)
        self.assertEqual(peer_faults([one, one, none], 1)[0], [])
        self.assertEqual(len(peer_faults([one, one, none], 2)[0]), 1)
        self.assertEqual(peer_faults([one, one, one], 2)[0], [])

    def test_score_window_counts_and_merges(self):
        t = tokens((100*MS, 100*MS, 'tap', -1), (130*MS, 130*MS, 'tap', -1))
        punished, counts = score_window([t, t], REQUEST, VOCAB)
        self.assertEqual(counts['close_onset'], 2)
        self.assertEqual(counts['no_support'], 0)
        self.assertEqual(len(punished[0]), 1)
        alone, _ = score_window([t], REQUEST, VOCAB, rules=False)
        self.assertEqual(alone, [[p for p in range(len(t)) if VOCAB.decision(t[p])[0] == 'head_cell']])


class GroupRewardTest(unittest.TestCase):
    def test_majority_onsets_are_rewarded_and_only_their_time_action(self):
        a = tokens((100*MS, 100*MS, 'flick', -1), (900*MS, 900*MS, 'tap', -1))
        b = tokens((110*MS, 110*MS, 'tap', -1))
        c = tokens((90*MS, 90*MS, 'tap', -1), (900*MS, 900*MS, 'tap', -1))
        punished, rewarded, counts = score_window([a, b, c], REQUEST, VOCAB, reward_majority=.75)
        self.assertEqual([VOCAB.decision(a[p])[0] for p in rewarded[0]], ['head_cell'])
        self.assertEqual(len(rewarded[0]), 1)
        self.assertEqual(punished[0], [])
        self.assertEqual(counts['majority'], 3)

    def test_a_punished_action_is_not_also_rewarded(self):
        t = tokens((100*MS, 100*MS, 'tap', -1), (130*MS, 130*MS, 'tap', -1))
        punished, rewarded, _ = score_window([t, t, t], REQUEST, VOCAB, reward_majority=.5)
        self.assertEqual(set(punished[0]) & set(rewarded[0]), set())
        self.assertEqual(len(punished[0]), 1)
        self.assertEqual(len(rewarded[0]), 1)


class AnchorTest(unittest.TestCase):
    def test_zero_at_the_start_positive_after_moving_and_skips_masked_actions(self):
        from beatfcn.train.peer_rl import anchor_kl
        start = torch.tensor([[[0., 1., -torch.inf], [2., 0., -torch.inf]]]).log_softmax(-1)
        mask = torch.tensor([[True, True]])
        self.assertAlmostEqual(float(anchor_kl(start, start, mask)), 0., places=6)
        moved = torch.tensor([[[3., 0., -torch.inf], [2., 0., -torch.inf]]]).log_softmax(-1)
        self.assertGreater(float(anchor_kl(moved, start, mask)), 0.)
        self.assertAlmostEqual(float(anchor_kl(moved, start, torch.tensor([[False, True]]))), 0., places=6)


class ReuseTest(unittest.TestCase):
    def test_one_sampling_round_serves_reuse_updates_then_samples_again(self):
        rl = PeerRL(dict(samples=2, reuse=3), VOCAB)
        calls = []
        t = tokens((100*MS, 100*MS, 'tap', -1))

        def fake(model, inputs, requests, lengths, seed):
            calls.append(seed)
            return [[dict(tokens=t, failure=None), dict(tokens=t, failure=None)] for _ in requests]
        rl.sample = fake
        rounds = [rl.accumulate(None, dict(), [REQUEST], [10], k, None)['rounds'] for k in range(4)]
        self.assertEqual(calls, [0, 3])
        self.assertEqual(rounds, [1, 0, 0, 1])


class PeerTrainingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.quiet = patch('builtins.print')
        self.quiet.start()
        self.addCleanup(self.quiet.stop)

    def test_real_run_samples_replays_and_changes_more_than_supervised(self):
        supervised = Run(self.tmp.name)
        self.assertEqual(supervised(2), 'done')
        saved = supervised.checkpoint(2)
        base = saved['raw'] if 'raw' in saved else saved['model']
        with tempfile.TemporaryDirectory() as other:
            run = Run(other, rl=dict(samples=3, windows=2, max_tokens=256, replay_chunk=2))
            self.assertEqual(run(2, compile_regions=False), 'done')
            saved = run.checkpoint(2)
            moved = saved['raw'] if 'raw' in saved else saved['model']
            self.assertTrue(any(not torch.equal(base[k], moved[k]) for k in base))
        with tempfile.TemporaryDirectory() as other:
            run = Run(other, rl=dict(samples=3, windows=2, max_tokens=256, replay_chunk=2, reward_majority=.5, reward_weight=.2,
                                     min_support=2, reuse=2, anchor_weight=.5))
            self.assertEqual(run(2, compile_regions=False), 'done')

    def test_one_step_makes_the_punished_draws_less_likely(self):
        from beatfcn.model.full_event_model import FullEventModel
        from beatfcn.train.condition_scst import prefix_sequence
        from beatfcn.train.peer_rl import collate_prefixes, replay_punished
        from tests.test_evolve_finetune import tiny_config
        from tests.test_full_event_model import inputs
        torch.manual_seed(3)
        config = tiny_config(mel_bins=0, intensity_dim=0)
        model = FullEventModel(config).eval()
        batch = inputs(config)
        batch['mode'] = torch.ones_like(batch['mode'])
        request = EventRequest('phira', 0, 8_000*MS, 10_000*MS)
        rl = PeerRL(dict(samples=4, weight=1., max_tokens=256, field_temperature={}), VOCAB)
        amp = lambda: torch.autocast('cpu', enabled=False)  # noqa: E731
        drawn = rl.sample(model, batch, [request], [100], 1)[0]
        token_lists = [list(o['tokens']) for o in drawn]
        punished, _ = score_window(token_lists, request, VOCAB)
        pairs = [(t, p) for t, p in zip(token_lists, punished) if p]
        self.assertTrue(pairs, 'an untrained tiny model must make faults')
        def measure():
            model.zero_grad()
            arrays = [prefix_sequence(t[:p[-1]+1], request, VOCAB)[0] for t, p in pairs]
            sequence = collate_prefixes(arrays, VOCAB, 'cpu')
            rows = torch.zeros(len(pairs), dtype=torch.long)
            return replay_punished(model, batch, rows, sequence, [p for _, p in pairs], amp)
        before = measure()
        before.backward()
        torch.optim.SGD(model.parameters(), lr=.01).step()
        self.assertLess(float(measure()), float(before))


if __name__ == '__main__':
    unittest.main()
