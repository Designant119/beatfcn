import copy
from dataclasses import replace
import unittest

import torch

from beatfcn.model.batch import collate
from beatfcn.model.p1 import P1Model, masked_ce
from beatfcn.model import features as F
from beatfcn.events.requests import build_request
from beatfcn.train.evolution import own_history_core, reserve_stage
from beatfcn.train.loop import CoreSampler
from events_helpers import synthetic_mania, DURATION
from test_p1_model import CFG, batch, batch_parts
from test_p1_model import song
import test_sample


class HistoryTests(unittest.TestCase):
    def test_history_has_no_current_or_future_hidden_groups(self):
        examples, songs = batch_parts(2, seed=3)
        ex = examples[0]
        cut = max(1, len(ex.groups) // 2)
        changed = copy.deepcopy(ex)
        for group in changed.groups[cut:]:
            for note in group['notes']:
                note['lane'] = (note['lane'] + 1) % 4
        a = collate([ex], [songs[0]], CFG.taps, CFG.top_tap)
        b = collate([changed], [songs[0]], CFG.taps, CFG.top_tap)
        torch.testing.assert_close(a['lane_history'][:, :cut + 1], b['lane_history'][:, :cut + 1])
        self.assertGreater((a['lane_history'][:, cut + 1:] - b['lane_history'][:, cut + 1:]).abs().sum(), 0)
        phira = collate([examples[1]], [songs[1]], CFG.taps, CFG.top_tap)
        self.assertEqual(phira['lane_history'].count_nonzero(), 0)

    def test_zero_adapter_warm_start_and_nonzero_gradients(self):
        bt = batch(2, seed=3)
        old = P1Model(CFG).eval()
        new = P1Model(replace(CFG, lane_history_groups=8)).eval()
        missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
        self.assertEqual(set(missing), {'lane_history_y.weight', 'lane_history_gru.weight'})
        self.assertEqual(unexpected, [])
        old_loss, _ = old(bt)
        new_loss, metrics = new(bt)
        torch.testing.assert_close(new_loss, old_loss, atol=1e-6, rtol=1e-6)
        new_loss.backward()
        for adapter in (new.lane_history_y, new.lane_history_gru):
            self.assertTrue(torch.isfinite(adapter.weight.grad).all())
            self.assertGreater(adapter.weight.grad.abs().sum(), 0)
        self.assertLessEqual(metrics['lane_choice'][1], metrics['lane'][1])

    def test_trained_adapter_incremental_matches_full(self):
        ex, song, _ = test_sample.IncrementalDecodingTests()._example()
        model = P1Model(replace(CFG, lane_history_groups=8)).eval()
        torch.nn.init.normal_(model.lane_history_y.weight, std=.03)
        torch.nn.init.normal_(model.lane_history_gru.weight, std=.03)
        with torch.no_grad():
            bt = collate([ex], [song], CFG.taps, CFG.top_tap)
            full, _, _ = model.encode(bt)
            cache = {}
            for n in range(4):
                prefix = copy.copy(ex)
                prefix.groups = ex.groups[:n]
                prefix.right = ex.right[:n + 1]
                partial = collate([prefix], [song], CFG.taps, CFG.top_tap)
                if n == 0:
                    row = model.encode(partial, cache)[0][:, :1]
                else:
                    row = model.encode_step(partial, cache)[0]
                torch.testing.assert_close(row, full[:, n:n + 1], atol=2e-5, rtol=2e-5)
                t = torch.tensor([[ex.groups[n]['t_rel_us'] / 1e6]])
                hist_partial = model.history_features(partial, t, start=n)
                hist_full = model.history_features(bt, t, start=n)
                torch.testing.assert_close(hist_partial, hist_full)


class SoftTimeTests(unittest.TestCase):
    def test_gaussian_mask_boundary_and_hard_metric(self):
        logits = torch.zeros(3, 40, requires_grad=True)
        target = torch.tensor([0, 39, 20])
        valid = torch.tensor([True, True, False])
        mask = torch.ones(3, 40, dtype=torch.bool)
        mask[0, 1] = False
        loss, count, correct, illegal = masked_ce(logits, target, valid, mask, sigma=1.5)
        self.assertEqual((int(count), int(correct), int(illegal)), (2, 1, 0))
        loss.backward()
        self.assertEqual(float(logits.grad[0, 1]), 0)
        self.assertEqual(float(logits.grad[2].abs().sum()), 0)
        # Recover target mass from the CE gradient: no wrapping across the cell edge.
        p = logits.detach().masked_fill(~mask, -1e9).softmax(-1)
        q = p - logits.grad * count
        self.assertGreater(float(q[0, 0]), float(q[0, 2]))
        self.assertAlmostEqual(float(q[0, 39]), 0, places=7)
        self.assertGreater(float(q[1, 39]), float(q[1, 37]))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_real_forward_with_soft_onset_and_hold_offsets(self):
        model = P1Model(replace(CFG, lane_history_groups=8, offset_sigma_ms=1.5))
        loss, metrics = model(batch(2, seed=3))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics['offset'][1], 0)
        self.assertGreater(metrics['end_off'][1], 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


class PolicyTests(unittest.TestCase):
    def test_own_history_keeps_future_labels_and_uses_generated_past(self):
        chart = synthetic_mania(3)
        sample = build_request(chart, kind='from_scratch', seed=1, audio_id='song', audio_duration_us=DURATION)
        def generate(*args, **kwargs):
            return [{'t_rel_us': 7_321_000, 'notes': [{'type': 0, 'lane': 3, 'line': -1, 'side': -1, 'x': 0}]}]
        actual = own_history_core(P1Model(CFG), sample, chart.events, song(0), 0, seed=3, generator=generate)
        expected = F.build_core(sample, chart.events, F.CORE_US, DURATION)
        self.assertEqual([g['t_rel_us'] for g in actual.groups], [g['t_rel_us'] for g in expected.groups])
        self.assertEqual([g['notes'] for g in actual.groups], [g['notes'] for g in expected.groups])
        past = actual.known[actual.known[:, 7] == 1]
        self.assertEqual(len(past), 1)
        self.assertAlmostEqual(float(past[0, 0]), -.679, places=5)
        self.assertEqual(int(past[0, 2]), 3)

    def test_conflicting_generated_hold_is_rejected_without_target_deletion(self):
        chart = synthetic_mania(3)
        sample = build_request(chart, kind='from_scratch', seed=1, audio_id='song', audio_duration_us=DURATION)
        target = next(e for e in sample.hidden if F.CORE_US <= e.onset_us < 2 * F.CORE_US)
        def generate(*args, **kwargs):
            return [{'t_rel_us': 7_321_000, 'notes': [{'type': 1, 'lane': target.location.lane,
                     'line': -1, 'side': -1, 'x': 0, 'end_rel_us': target.onset_us + 100_000 - 7_321_000}]}]
        with self.assertRaisesRegex(F.SkipExample, 'conflicts with unchanged future'):
            own_history_core(P1Model(CFG), sample, chart.events, song(0), 0, seed=3, generator=generate)

    def test_complete_control_dropout_and_same_audio_reference_pairs(self):
        charts = {'a': synthetic_mania(3), 'b': synthetic_mania(5)}
        sample = build_request(charts['a'], kind='from_scratch', seed=2, audio_id='song',
                               audio_duration_us=DURATION, control_drop_all_p=1)
        self.assertEqual(set(sample.request['controls']), {'allowed_types'})
        rows = [{'chart_id': cid, 'song': 'same', 'mode': 'mania_4k', 'duration_us': DURATION} for cid in charts]
        sampler = CoreSampler(rows, charts.__getitem__, lambda _: object(), seed=3,
                              request_mix={'reference': 1.0}, paired_references=True)
        pairs = [sampler.draw(2) for _ in range(10)]
        matched = [(a, b) for items in pairs if len(items) == 2 for (a, _), (b, _) in [items]
                   if a.provenance.get('pair_id')]
        self.assertTrue(matched)
        for a, b in matched:
            self.assertEqual(a.c0_us, b.c0_us)
            self.assertEqual(a.provenance['pair_id'], b.provenance['pair_id'])
            self.assertNotEqual(a.provenance['chart_id'], b.provenance['chart_id'])
        four = sampler.draw(4)
        self.assertEqual(len(four), 4)
        self.assertTrue(all(ex.provenance.get('pair_id') for ex, _ in four))
        self.assertEqual(four[0][0].provenance['pair_id'], four[1][0].provenance['pair_id'])
        self.assertEqual(four[2][0].provenance['pair_id'], four[3][0].provenance['pair_id'])

    def test_budget_reserves_failed_attempts_and_rejects_duplicates(self):
        ledger = {'approved_usd': 1, 'stages': {}}
        resources = {'gpu': 'L40S', 'cpu': 4, 'memory_gib': 32, 'timeout': 600}
        reserve_stage(ledger, 'one', resources)['status'] = 'failed'
        with self.assertRaises(ValueError):
            reserve_stage(ledger, 'one', resources)
        reserve_stage(ledger, 'two', resources)
        with self.assertRaisesRegex(ValueError, 'budget exhausted'):
            reserve_stage(ledger, 'three', resources)


if __name__ == '__main__':
    unittest.main()
