import itertools
from fractions import Fraction
import random
import unittest

import numpy as np

from beatfcn.events.canonical import TimingMap
from beatfcn.train.workflow_metrics import (DEFAULT_COSTS, as_notes, bootstrap_mean, chart_edit_rate,
    cost_sensitivity, density_profile, grid_from_timing, hungarian, optimal_matching, pair_ops,
    playability, snap_notes, snap_times)

MS = 1000


def tap(t_ms, lane=0, typ='tap'):
    return dict(onset_us=int(t_ms*MS), end_us=int(t_ms*MS), type=typ, lane=lane)


def hold(t_ms, end_ms, lane=0):
    return dict(onset_us=int(t_ms*MS), end_us=int(end_ms*MS), type='hold', lane=lane)


def brute_force_cost(draft, ref, mode, costs=None, compound=False):
    costs = dict(DEFAULT_COSTS, **(costs or {}))
    d, r = as_notes(draft, mode), as_notes(ref, mode)
    best = [costs['insert']*len(r)+costs['delete']*len(d)]

    def go(i, used, acc):
        if i == len(d):
            best[0] = min(best[0], acc+costs['insert']*(len(r)-len(used)))
            return
        go(i+1, used, acc+costs['delete'])
        for j in range(len(r)):
            if j in used:
                continue
            ops = pair_ops(d[i], r[j], compound=compound)
            if ops is not None:
                go(i+1, used | {j}, acc+sum(costs[o] for o in ops))
    go(0, frozenset(), 0.)
    return best[0]


def random_chart(rng, mode, n):
    types = ('tap', 'hold') if mode == 'mania_4k' else ('tap', 'hold', 'flick', 'drag')
    out = []
    for _ in range(n):
        t = rng.choice([0, 0, 10, 30, 60, 100, 140, 170, 200, 260])+rng.randint(-15, 15)
        typ = rng.choice(types)
        lane = rng.randrange(4) if mode == 'mania_4k' else -1
        out.append(hold(t, t+rng.choice([30, 80, 200]), lane) if typ == 'hold' else tap(t, lane, typ))
    return out


class EditRate(unittest.TestCase):
    def setUp(self):
        self.ref = [tap(0, 0), tap(0, 2), hold(500, 900, 1), tap(750, 3), tap(1000, 0)]

    def test_identical_is_zero(self):
        r = chart_edit_rate(self.ref, self.ref, 'mania_4k')
        self.assertEqual(r['cer'], 0.)
        self.assertEqual(r['ops']['keep'], 5)
        self.assertEqual(r['savings'], 1.)

    def test_empty_draft_is_one(self):
        r = chart_edit_rate([], self.ref, 'mania_4k')
        self.assertEqual(r['cer'], 1.)
        self.assertEqual(r['ops']['insert'], 5)
        self.assertIsNone(chart_edit_rate([], [], 'phira')['cer'])

    def test_known_edit_scripts(self):
        n = len(self.ref)
        cases = [  # (mutated draft, expected total cost, op)
            ([tap(50, 0), tap(0, 2), hold(500, 900, 1), tap(750, 3), tap(1000, 0)], .4, 'retime'),
            ([tap(0, 0), tap(0, 2), hold(500, 1000, 1), tap(750, 3), tap(1000, 0)], .4, 'tail'),
            ([tap(0, 0), tap(0, 2), tap(500, 1), tap(750, 3), tap(1000, 0)], .8, None),  # retype + place tail
            ([tap(0, 0), tap(0, 2), hold(500, 900, 1), tap(750, 2), tap(1000, 0)], .4, 'relane'),
            (self.ref+[tap(300, 1)], .3, 'delete'),
            (self.ref[:-1], 1., 'insert'),
            ([tap(0, 0), tap(0, 2), hold(500, 900, 1), tap(750, 3), tap(1300, 0)], 1.3, None),  # 300 ms: del+ins
            ([tap(0, 0), tap(0, 2), hold(560, 960, 1), tap(750, 3), tap(1000, 0)], .4, 'retime'),  # hold moved whole
            ([tap(0, 0), tap(0, 2), hold(560, 900, 1), tap(750, 3), tap(1000, 0)], .8, None),  # moved + tail
        ]
        for draft, cost, op in cases:
            r = chart_edit_rate(draft, self.ref, 'mania_4k')
            self.assertAlmostEqual(r['total_cost'], cost, msg=(draft, r))
            self.assertAlmostEqual(r['cer'], cost/n)
            if op:
                self.assertEqual(r['ops'][op], 1, r['ops'])

    def test_strict_versus_compound(self):
        ref = [tap(0, 0)]
        draft = [tap(60, 1)]  # late AND wrong column
        self.assertAlmostEqual(chart_edit_rate(draft, ref, 'mania_4k')['total_cost'], 1.3)
        self.assertAlmostEqual(chart_edit_rate(draft, ref, 'mania_4k', compound=True)['total_cost'], .8)

    def test_phira_ignores_position_and_scores_types(self):
        ref = [tap(0, -1, 'tap'), tap(0, -1, 'flick'), tap(200, -1, 'drag')]
        draft = [tap(5, -1, 'flick'), tap(5, -1, 'tap'), tap(200, -1, 'tap')]
        r = chart_edit_rate(draft, ref, 'phira')
        self.assertAlmostEqual(r['total_cost'], .4)  # chord types swap freely; one retype
        self.assertEqual(r['ops']['retype'], 1)

    def test_cost_override_and_sensitivity(self):
        draft = [tap(50, 0), tap(0, 2), hold(500, 900, 1), tap(750, 3), tap(1000, 0), tap(300, 1)]
        base = chart_edit_rate(draft, self.ref, 'mania_4k')
        self.assertAlmostEqual(base['total_cost'], .7)
        s = cost_sensitivity(draft, self.ref, 'mania_4k', per_op=True)
        self.assertAlmostEqual(s['edits_x2'], 1.4/5)
        self.assertAlmostEqual(s['edits_x0.5'], .35/5)
        self.assertAlmostEqual(s['delete_x2'], 1./5)
        self.assertAlmostEqual(s['retime_x0.5'], .5/5)
        with self.assertRaises(ValueError):
            chart_edit_rate(draft, self.ref, 'mania_4k', costs=dict(bogus=1))

    def test_matcher_is_optimal_against_brute_force(self):
        rng = random.Random(7)
        for trial in range(400):
            mode = rng.choice(['mania_4k', 'phira'])
            d, r = random_chart(rng, mode, rng.randint(0, 5)), random_chart(rng, mode, rng.randint(0, 5))
            costs = None if trial % 2 else {k: rng.choice([.1, .3, .5, .9]) for k in ('delete', 'retime', 'retype', 'tail')}
            compound = trial % 3 == 0
            got = chart_edit_rate(d, r, mode, costs=costs, compound=compound)['total_cost']
            want = brute_force_cost(d, r, mode, costs=costs, compound=compound)
            self.assertAlmostEqual(got, want, places=9, msg=(trial, d, r, costs))

    def test_matching_is_one_to_one(self):
        rng = random.Random(3)
        d, r = random_chart(rng, 'mania_4k', 40), random_chart(rng, 'mania_4k', 40)
        m = optimal_matching(as_notes(d, 'mania_4k'), as_notes(r, 'mania_4k'))
        self.assertEqual(len({x[0] for x in m}), len(m))
        self.assertEqual(len({x[1] for x in m}), len(m))

    def test_hungarian_against_permutations(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            n, m = int(rng.integers(1, 5)), int(rng.integers(1, 6))
            n, m = min(n, m), max(n, m)
            c = rng.normal(size=(n, m)).round(2)
            a = hungarian(c)
            got = c[np.arange(n), a].sum()
            want = min(sum(c[i, p[i]] for i in range(n)) for p in itertools.permutations(range(m), n))
            self.assertAlmostEqual(got, want)

    def test_large_song_is_fast_enough(self):
        import time
        rng = np.random.default_rng(1)
        ref = [tap(float(t), int(l)) for t, l in zip(np.sort(rng.uniform(0, 180_000, 2000)), rng.integers(0, 4, 2000))]
        shifts = rng.integers(-60, 60, len(ref))*MS
        draft = [tap((x['onset_us']+int(s))/MS, x['lane']) for x, s in zip(ref, shifts)]
        start = time.monotonic()
        r = chart_edit_rate(draft, ref, 'mania_4k')
        self.assertLess(time.monotonic()-start, 20)
        self.assertLess(r['cer'], .5)


class Snapping(unittest.TestCase):
    def test_union_grid(self):
        grid = [(0., 500_000.)]  # 120 BPM from 0
        got, moved = snap_times([510_000, 1_170_000, 1_000_000], grid)
        self.assertEqual(got.tolist(), [500_000, 1_166_667, 1_000_000])
        self.assertTrue(moved.all())
        slow = [(0., 2_000_000.)]  # 30 BPM: 1/16 = 125 ms, 1/6 = 333 ms
        got, moved = snap_times([62_000], slow)
        self.assertEqual(got.tolist(), [62_000])
        self.assertFalse(moved[0])

    def test_grid_from_timing_map_and_segments(self):
        tm = TimingMap.from_bpm_points([(0, 120), (4, 60)], offset_sec=Fraction(1, 10))
        grid = grid_from_timing(tm)
        self.assertEqual(grid, [(100_000., 500_000.), (2_100_000., 1_000_000.)])
        got, _ = snap_times([2_590_000, 90_000], grid)
        self.assertEqual(got.tolist(), [2_600_000, 100_000])

    def test_snap_notes_keeps_hold_valid(self):
        grid = [(0., 500_000.)]
        notes, stats = snap_notes([(510_000, 520_000, 'hold', 0), (995_000, 995_000, 'tap', 1)], grid)
        self.assertEqual(notes[0][0], 500_000)
        self.assertGreater(notes[0][1], notes[0][0])
        self.assertEqual(notes[1], (1_000_000, 1_000_000, 'tap', 1))
        self.assertEqual(stats['heads_snapped'], 2)

    def test_snapping_repairs_small_offsets_in_cer(self):
        grid = [(0., 1_000_000.)]  # 60 BPM: nearest union points 62.5 ms (1/16) apart
        ref = as_notes([tap(1000, 0), tap(2000, 1), tap(3000, 2)], 'mania_4k')
        draft = as_notes([tap(1025, 0), tap(1977, 1), tap(3024, 2)], 'mania_4k')
        self.assertAlmostEqual(chart_edit_rate(draft, ref, 'mania_4k')['cer'], .4)
        snapped, _ = snap_notes(draft, grid)
        self.assertEqual(chart_edit_rate(snapped, ref, 'mania_4k')['cer'], 0.)


class Playability(unittest.TestCase):
    def test_mania_detectors(self):
        p = playability([tap(0, 0), tap(0, 0), hold(100, 400, 1), tap(200, 1), tap(500, 2), tap(530, 2),
                         tap(1000, 3)], 'mania_4k')
        self.assertEqual((p['same_lane_overlap'], p['hold_covers_head'], p['jack_lt_40ms']), (1, 1, 1))
        clean = playability([tap(0, 0), hold(100, 400, 1), tap(400, 1), tap(450, 1)], 'mania_4k')
        self.assertEqual(clean['any_violation_rate'], 0.)

    def test_phira_detectors(self):
        ev = [tap(0, -1), tap(0, -1, 'flick'), tap(20, -1), tap(1000, -1, 'drag'), tap(1020, -1, 'drag')]
        ev += [tap(2000, -1)]*5+[dict(onset_us=3_000_000, end_us=3_000_000, type='hold', lane=-1)]
        p = playability(ev, 'phira')
        self.assertEqual(p['short_gap_lt_40ms'], 2)
        self.assertEqual(p['short_gap_nondrag'], 1)
        self.assertEqual(p['stack_gt4'], 1)
        self.assertEqual(p['zero_length_hold'], 1)
        self.assertEqual(playability([], 'phira')['short_gap_lt_40ms_rate'], None)


class Aggregates(unittest.TestCase):
    def test_bootstrap_is_cluster_level_and_seeded(self):
        a = bootstrap_mean([[.2, .4], [.5], [.9]], samples=500, seed=1)
        self.assertAlmostEqual(a['mean'], (0.3+.5+.9)/3)
        self.assertEqual(a, bootstrap_mean([[.2, .4], [.5], [.9]], samples=500, seed=1))
        lo, hi = a['ci95']
        self.assertLessEqual(lo, a['ci80'][0])
        self.assertGreaterEqual(hi, a['ci80'][1])
        self.assertIsNone(bootstrap_mean([[.3]])['ci80'])

    def test_density_profile(self):
        ref = as_notes([tap(100), tap(1100), tap(1200)], 'mania_4k')
        p = density_profile(ref, ref, 3_000_000)
        self.assertAlmostEqual(p['pearson'], 1.)
        self.assertEqual(p['count_ratio'], 1.)
        self.assertIsNone(density_profile([], ref, 3_000_000)['pearson'])


class UnusableTimingTest(unittest.TestCase):
    def test_gimmick_bpm_raises_unusable_timing(self):
        from fractions import Fraction
        from types import SimpleNamespace
        from beatfcn.train.workflow_metrics import UnusableTiming, grid_from_timing
        timing = lambda bpm: SimpleNamespace(segments=[SimpleNamespace(start_sec=Fraction(0), bpm=bpm)], offset_sec=Fraction(0))
        self.assertEqual(grid_from_timing(timing(Fraction(120))), [(0.0, 500000.0)])
        for bpm in (Fraction(1, 10**400), Fraction(10**9)):
            with self.assertRaises(UnusableTiming):
                grid_from_timing(timing(bpm))



class HoldReleaseTest(unittest.TestCase):
    def test_metric_counts_press_at_release_and_repair_fixes_it(self):
        from beatfcn.train.workflow_metrics import playability, repair_hold_release
        notes = [(0, 300_000, 'hold', 0), (300_000, 300_000, 'tap', 0),        # press exactly at release
                 (500_000, 800_000, 'hold', 1), (820_000, 820_000, 'tap', 1),  # 20 ms after release
                 (1_000_000, 1_050_000, 'hold', 2), (1_070_000, 1_070_000, 'tap', 2),  # too short after repair
                 (2_000_000, 2_200_000, 'hold', 3), (2_400_000, 2_400_000, 'tap', 3)]  # fine
        p = playability(notes, 'mania_4k')
        self.assertEqual(p['hold_release_lt_40ms'], 3)
        self.assertEqual(p['hold_covers_head'], 0)  # the old metric missed all three
        fixed, counts = repair_hold_release(notes, 'mania_4k')
        self.assertEqual(counts, dict(shortened=2, to_tap=1))
        self.assertEqual(playability(fixed, 'mania_4k')['hold_release_lt_40ms'], 0)
        self.assertIn((0, 260_000, 'hold', 0), fixed)
        self.assertIn((1_000_000, 1_000_000, 'tap', 2), fixed)
        self.assertEqual(len(fixed), len(notes))
        self.assertEqual(repair_hold_release(notes, 'phira')[1], dict(shortened=0, to_tap=0))


if __name__ == '__main__':
    unittest.main()
