"""Unit tests for scripts/benchmark_v3.py on synthetic data only (no tape, no model). python -m unittest tests.test_benchmark_v3"""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import benchmark_v3 as B  # noqa: E402

D_FEATS = Path('/Users/yukoval/yukoval-projects/beatfcn/reports/research-20261104/data/feats.py')


def mania_chart(seconds=120, step_ms=250, seed=0, lanes=4):
    rng = np.random.default_rng(seed)
    notes, last = [], -1
    for t in range(1_000_000, seconds*1_000_000, step_ms*1000):
        lane = int(rng.integers(0, lanes))
        while lane == last and lanes > 1:
            lane = int(rng.integers(0, lanes))
        last = lane
        notes.append((t, t, 'tap', lane))
    return notes


def phira_chart(seconds=120, step_ms=300, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for t in range(1_000_000, seconds*1_000_000, step_ms*1000):
        k = str(rng.choice(['tap', 'tap', 'hold', 'flick', 'drag']))
        out.append((t, t+200_000 if k == 'hold' else t, k, -1))
    return out


def write_chart(tape, chart_id, notes, split='val'):
    stem = chart_id.replace(':', '-')
    on, end, kind, lane = B.arrays(notes)
    np.savez(Path(tape)/f'{stem}-targets.npz', onset_us=on, end_us=end, kind=kind, lane=lane)
    (Path(tape)/f'{stem}-chart.json').write_text(json.dumps(dict(chart_id=chart_id, split=split)))
    return f'/data/full-database-v1/native-v2-tape/{stem}-targets.npz'


def human_population(mode, n, seed):
    """Feature rows of synthetic 'human' charts: human_reference arrays + log_nps."""
    names = B.core_features(mode)+B.GRID_FEATURES
    rows, lns = [], []
    for i in range(n):
        step = int(np.random.default_rng(seed+i).integers(150, 500))
        notes = mania_chart(step_ms=step, seed=seed+i) if mode == 'mania_4k' else phira_chart(step_ms=step, seed=seed+i)
        o, e, k, l = B.arrays(notes)
        f = B.features(o, e, k, l, mode, 120_000_000, [[0., 500_000., 4]])
        rows.append(B._vec(f, names))
        lns.append(f['log_nps'])
    return names, np.array(rows, np.float32), np.array(lns, np.float32)


def tiny_floor(folder):
    data = dict(schema=B.SCHEMA, H=dict(reference=dict(file='human_reference.npz', names={})), W={}, rules={})
    arrays = {}
    for mode in ('phira', 'mania_4k'):
        names, fit, ln = human_population(mode, 40, 100)
        _, ev, le = human_population(mode, 40, 500)
        arrays.update({f'{mode}_fit': fit, f'{mode}_fit_lognps': ln, f'{mode}_eval': ev, f'{mode}_eval_lognps': le})
        data['H']['reference']['names'][mode] = names
        data['H'][mode] = dict(auc_null=.99, rulers={k: .1 for k in B.core_features(mode)})
        table = dict(exact=dict(null=[.9]*7, human=[.5]*7), best=dict(null=[.85]*7, human=[.45]*7))
        minref = {c: dict(human={str(k): table[c]['human'][0] for k in range(1, 6)}, null={str(k): table[c]['null'][0] for k in range(1, 6)}) for c in table}
        data['W'][mode] = dict(rho_table=table, minref=minref)
        data['rules'][mode] = dict(per_1k={k: 0. for k in B.RATE_RULES[mode]})
    np.savez_compressed(Path(folder)/'human_reference.npz', **arrays)
    (Path(folder)/'floor.json').write_text(json.dumps(data))
    return Path(folder)/'floor.json'


def state(notes, complete=True):
    return dict(complete=complete, failure=None, legality=[], events=[dict(onset_us=a, end_us=b, type=t, lane=l) for a, b, t, l in notes])


class Basics(unittest.TestCase):
    def test_constants_match_source(self):
        src = (ROOT/'scripts/run_workflow_benchmark.py').read_text()
        self.assertIn(f'PANEL_SEED = {B.PANEL_SEED}', src)
        self.assertIn("TYPES", (ROOT/'beatfcn/events/canonical.py').read_text())
        self.assertRegex((ROOT/'beatfcn/events/canonical.py').read_text(), r"TYPES\s*=\s*\(\s*'tap',\s*'hold',\s*'flick',\s*'drag'")

    @unittest.skipUnless(D_FEATS.exists(), 'research D feats.py not local')
    def test_features_match_research_d(self):
        ns = {}
        exec(compile(D_FEATS.read_text(), str(D_FEATS), 'exec'), ns)
        for mode, notes in (('mania_4k', mania_chart(step_ms=180)), ('phira', phira_chart())):
            o, e, k, l = B.arrays(notes)
            for segs in (None, [[0., 400_000., 4]]):
                a = B.features(o, e, k, l, mode, 130_000_000, segs)
                b = ns['features'](o, e, k, l, mode, 130_000_000, segs)
                self.assertEqual(sorted(a), sorted(b))
                for key in a:
                    if not (math.isnan(a[key]) and math.isnan(b[key])):
                        self.assertEqual(a[key], b[key], key)

    def test_primary_reference_hash(self):
        import hashlib
        song = dict(references=[dict(chart_id=x) for x in ('osu:1', 'osu:2', 'osu:3')])
        rank = lambda c: hashlib.sha256(f'{B.PANEL_SEED}:primary:{c}'.encode()).hexdigest()
        self.assertEqual(B.primary_reference(song)['chart_id'], min(('osu:1', 'osu:2', 'osu:3'), key=rank))

    def test_load_drafts_formats(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'c.json'
            p.write_text(json.dumps({'mania_4k:a': state([(1, 1, 'tap', 0)])}))
            self.assertIn('mania_4k:a', B.load_drafts(p))
            with self.assertRaises(ValueError):
                B.load_drafts(p, 1)
            s = Path(d)/'s.json'
            body = {'0': {'x': state([(1, 1, 'tap', 0)])}, '1': {'x': state([(2, 2, 'tap', 0)])}}
            s.write_text(json.dumps(body))
            self.assertEqual(B.load_drafts(s, 1)['x']['events'][0]['onset_us'], 2)
            with self.assertRaises(ValueError):
                B.load_drafts(s)
            t = Path(d)/'t.json'
            t.write_text(json.dumps(body)[:-20])
            self.assertEqual(B.load_drafts(t, 0)['x']['events'][0]['onset_us'], 1)
            with self.assertRaises(ValueError):
                B.load_drafts(t, 1)

    def test_test_split_refused_before_reading_notes(self):
        with tempfile.TemporaryDirectory() as d:
            write_chart(d, 'osu:9', mania_chart(), split='test')
            with self.assertRaises(PermissionError):
                B.load_chart(d, 'osu-9-targets.npz', 'mania_4k')
            write_chart(d, 'osu:8', mania_chart(), split='train')
            self.assertTrue(B.load_chart(d, 'osu-8-targets.npz', 'mania_4k'))
            (Path(d)/'osu-8-chart.json').write_text('{}')
            with self.assertRaises(ValueError):  # no flag: fail closed
                B.load_chart(d, 'osu-8-targets.npz', 'mania_4k')

    def test_beat_clock_round_trip_and_warp(self):
        c = B.BeatClock([[1_000_000., 500_000., 4], [5_000_000., 250_000., 4]])
        t = np.array([0., 1_000_000., 2_500_000., 5_000_000., 7_000_000.])
        np.testing.assert_allclose(c.to_time(c.to_beats(t)), t)
        self.assertAlmostEqual(float(c.to_beats(5_000_000.)), 8.)
        notes = [(1_000_000+k*500_000, 1_000_000+k*500_000, 'tap', 0) for k in range(8)]
        w = B.warp_notes(notes, [[1_000_000., 500_000., 4]], [[0., 400_000., 4]], 100_000_000)
        self.assertEqual([n[0] for n in w], [k*400_000 for k in range(8)])  # beats kept, grid changed

    def test_count_lower_bound(self):
        self.assertAlmostEqual(B.count_lower_bound(.5), .5)
        self.assertAlmostEqual(B.count_lower_bound(2.), .3)
        self.assertEqual(B.count_lower_bound(1.), 0.)


class Rules(unittest.TestCase):
    def test_clean_mania_passes(self):
        r = B.rule_counts(mania_chart(), 'mania_4k', 130_000_000)
        for k in B.HARD_RULES['mania_4k']:
            self.assertEqual(r[k], 0, k)
        self.assertFalse(r['short_drain'])

    def test_each_violation_detected(self):
        base = mania_chart()
        t = 5_000_000
        cases = [
            ('same_lane_overlap', base+[(base[0][0], base[0][0], 'tap', base[0][3])]),
            ('hold_covers_head', [(t, t+1_000_000, 'hold', 0), (t+200_000, t+200_000, 'tap', 0)]+base[40:]),
            ('bad_lane', base+[(t, t, 'tap', 7)]),
            ('bad_hold_end', base+[(t+1, t+1, 'hold', 1)]),
            ('nonhold_with_tail', base+[(t+1, t+50, 'tap', 1)]),
            ('out_of_song', base+[(-5, -5, 'tap', 1)]),
            ('empty_lanes', mania_chart(lanes=3)),
            ('jack_lt_40ms', base+[(base[10][0]+10_000, base[10][0]+10_000, 'tap', base[10][3])])]
        for key, notes in cases:
            self.assertGreater(B.rule_counts(notes, 'mania_4k', 130_000_000)[key], 0, key)
        p = [(1_000_000, 1_000_000, 'hold', -1)]+phira_chart()
        self.assertEqual(B.rule_counts(p, 'phira', 130_000_000)['zero_length_hold'], 1)
        chord = [(2_000_000, 2_000_000, 'tap', -1)]*5+phira_chart()
        self.assertGreaterEqual(B.rule_counts(chord, 'phira', 130_000_000)['stack_gt4'], 1)
        self.assertTrue(B.rule_counts(mania_chart(seconds=10), 'mania_4k', 12_000_000)['short_drain'])


class Classifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.names = [f'f{i}' for i in range(6)]
        X = rng.normal(size=(6000, 6))
        ln = rng.normal(1.5, .3, 6000)
        cls.ref = B.HumanRef(cls.names, ln[:3000], X[:3000], ln[3000:], X[3000:])
        cls.solver = cls.ref.solver(list(range(6)))
        cls.rng = rng

    def test_same_distribution_auc_half_no_leak(self):
        aucs = []
        for _ in range(40):
            Z = self.ref.z(self.rng.normal(size=(20, 6)), np.full(20, 1.5))
            aucs.append(B.loso_auc(Z, self.solver)[0])
        self.assertLess(abs(np.mean(aucs)-.5), .03)

    def test_shift_detected_at_small_panel(self):
        aucs = [B.loso_auc(self.ref.z(self.rng.normal(1.5, 1, size=(15, 6)), np.full(15, 1.5)), self.solver)[0] for _ in range(20)]
        self.assertGreater(np.mean(aucs), .9)

    def test_density_is_conditioned_out(self):
        rng = np.random.default_rng(3)
        ln = rng.uniform(0, 3, 1000)
        x = (ln*2+rng.normal(size=1000))[:, None]
        ref = B.HumanRef(['a'], ln[:500], x[:500], ln[500:], x[500:])
        solver = ref.solver([0])
        dense = np.full(30, 2.8)
        draft = dense*2+rng.normal(size=30)  # humanlike at its own density, though far from the population mean
        self.assertLess(abs(B.loso_auc(ref.z(draft[:, None], dense), solver)[0]-.5), .2)
        self.assertLess(abs(float(ref.z(draft[:, None], dense).mean())), .5)

    def test_bootstrap_copies_do_not_leak(self):
        Z = self.ref.z(self.rng.normal(size=(10, 6)), np.full(10, 1.5))
        counts = np.array([5, 5, 0, 0, 0, 0, 0, 0, 0, 0])
        a, u = B.loso_auc(Z, self.solver, counts)
        mu, A, E = self.solver
        w0 = A@(Z[1]-mu)
        s = Z[0]@w0
        expect = (E@w0 < s).mean()
        self.assertAlmostEqual(u[0], expect)
        self.assertTrue(np.isnan(u[2]))
        self.assertTrue(math.isnan(B.loso_auc(Z, self.solver, np.array([9]+[0]*9))[0]))

    def test_h_scale(self):
        self.assertEqual(B.h_from_auc(.5, .99), 1.)
        self.assertEqual(B.h_from_auc(.99, .99), 0.)
        self.assertEqual(B.h_from_auc(.3, .99), 1.)
        self.assertAlmostEqual(B.h_from_auc(.745, .99), .5)
        self.assertEqual(B.gap_score([.05, .2], [.1, .1]), .75)


class Corruptions(unittest.TestCase):
    def test_corruptions_change_what_they_claim(self):
        n = mania_chart(step_ms=200)
        self.assertEqual(B.corrupt(n, 'control', 0, 'mania_4k', 'x'), n)
        s = B.corrupt(n, 'shift', 30, 'mania_4k', 'x')
        self.assertEqual(s[0][0], n[0][0]+30_000)
        self.assertLess(len(B.corrupt(n, 'drop', .5, 'mania_4k', 'x')), .6*len(n))
        self.assertGreater(len(B.corrupt(n, 'insert', .5, 'mania_4k', 'x')), 1.4*len(n))
        p = phira_chart()
        c = B.corrupt(p, 'type_collapse', 1., 'phira', 'x')
        self.assertTrue(all(x[2] == 'tap' and x[0] == x[1] for x in c))
        ts = B.corrupt(p, 'type_shuffle', 1., 'phira', 'x')
        self.assertEqual(sorted(x[2] for x in ts), sorted(x[2] for x in p))
        self.assertEqual(B.corrupt(n, 'lane_mirror', 1, 'mania_4k', 'x')[0][3], 3-n[0][3])
        m = B.corrupt(n, 'null_shuffle', 1, 'mania_4k', 'x')
        self.assertEqual(len(m), len(n))
        flat = B.corrupt(p, 'null_flat', 1, 'phira', 'x')
        self.assertEqual(len(flat), len(p))
        self.assertTrue(all(x[2] == 'tap' and x[0] == x[1] for x in flat))
        g = B.corrupt(p+[(p[0][0], p[0][0], 'flick', -1)], 'group_jitter', 20, 'phira', 'x')
        self.assertEqual(len(np.unique([x[0] for x in g])), len(np.unique([x[0] for x in p])))  # chords stay chords
        for name, dose in (('jitter', 20), ('bar_shuffle', 8), ('loop', 8), ('burst', .1), ('jack_inject', .5)):
            self.assertTrue(B.corrupt(n, name, dose, 'mania_4k', 'x', 130_000_000))

    def test_core_features_ignore_storage_order_of_notes(self):
        p = phira_chart()
        chord = [(t, t, k, -1) for t in range(3_000_000, 100_000_000, 900_000) for k in ('flick', 'tap')]
        a = p+chord
        names = B.core_features('phira')
        v = lambda x: B._vec(B.features(*B.arrays(x), 'phira', 130_000_000, None), names)
        np.testing.assert_allclose(v(a), v(sorted(a, key=lambda n: (n[0], n[2]))))

    def test_chart_features_invariant_to_time_origin_and_mirror(self):
        n = mania_chart(step_ms=180)
        names = B.core_features('mania_4k')
        v = lambda x: B._vec(B.features(*B.arrays(x), 'mania_4k', 130_000_000, None), names)
        np.testing.assert_allclose(v(B.corrupt(n, 'lane_mirror', 1, 'mania_4k', 'x')), v(n))
        np.testing.assert_allclose(v(B.corrupt(n, 'shift', 1000, 'mania_4k', 'x')), v(n))


class Music(unittest.TestCase):
    @staticmethod
    def mel_with_peaks(onsets_us, frames=12000, seed=0):
        rng = np.random.default_rng(seed)
        mel = rng.normal(0, .15, (frames, 16))
        for t in onsets_us:
            k = int(t//10_000)
            if 0 <= k < frames-4:
                mel[k:k+4] += 2.
        return mel

    def test_lift_nulls_and_silence(self):
        beat = 500_000
        segs = [[0., float(beat), 4]]
        rng = np.random.default_rng(1)
        # sounds on a random subset of beats, so grid regularity alone gives nothing
        hits = np.array([k*beat for k in range(4, 230) if rng.random() < .4])
        mel = self.mel_with_peaks(hits)
        cur = B.audio_curves(mel)
        real = B.onset_lift(cur['star'], hits, segs)
        self.assertTrue(real['grid_controlled'])
        self.assertGreater(real['lift'], 1.)
        unrelated = np.array([k*beat for k in range(4, 230) if rng.random() < .4])  # on the grid, not on the sounds
        self.assertLess(B.onset_lift(cur['star'], np.setdiff1d(unrelated, hits), segs)['lift'], real['lift']/2)
        rolled = np.roll(cur['star'], 317)
        self.assertLess(abs(B.onset_lift(rolled, hits, segs)['lift']), real['lift']/2)
        other = B.audio_curves(self.mel_with_peaks(np.array([k*beat for k in range(4, 230) if rng.random() < .4]), seed=2))
        self.assertLess(B.onset_lift(other['star'], hits, segs)['lift'], real['lift']/2)
        no_grid = B.onset_lift(cur['star'], hits, None)
        self.assertFalse(no_grid['grid_controlled'])
        # energy follows note density
        e = np.ones(12000)
        e[6000:] = 5.
        notes = np.r_[np.arange(0, 60_000_000, 2_000_000), np.arange(60_000_000, 120_000_000, 200_000)]
        self.assertGreater(B.density_energy(e, notes, 120_000_000), .5)
        # true silence
        song = np.full(12000, 5.)
        song[3000:4000] = -20.
        ref = [(1_000_000, 1_000_000, 'tap', 0), (29_000_000, 29_000_000, 'tap', 0), (41_000_000, 41_000_000, 'tap', 0), (118_000_000, 118_000_000, 'tap', 0)]
        got = B.silence_rate([35_000_000, 36_000_000, 10_000_000], ref, 120_000_000, song)
        self.assertEqual(got['true_silence_notes'], 2)
        self.assertGreater(got['true_silence_events_per_s'], 0)

    def test_find_mel_layouts(self):
        with tempfile.TemporaryDirectory() as d:
            for p in (Path(d)/'a', Path(d)/'features'/'b'):
                p.mkdir(parents=True)
                np.save(p/'mel10_complete.npy', np.zeros((2, 2)))
            np.save(Path(d)/'c.npy', np.zeros((2, 2)))
            for sha in 'abc':
                self.assertIsNotNone(B.find_mel(d, sha))
            self.assertIsNone(B.find_mel(d, 'zz'))


def make_panel(tape, n_per_mode=8, multi=True):
    songs, good, bad = [], {}, {}
    for mode in ('phira', 'mania_4k'):
        for i in range(n_per_mode):
            sid = f'{mode}:{i}'
            maker = mania_chart if mode == 'mania_4k' else phira_chart
            r1 = maker(seed=i*7+1)
            refs = [dict(chart_id=f'{mode[:2]}:{i}0', target_path=write_chart(tape, f'{mode[:2]}:{i}0', r1))]
            if mode == 'mania_4k' and multi and i % 2:
                refs.append(dict(chart_id=f'{mode[:2]}:{i}1', target_path=write_chart(tape, f'{mode[:2]}:{i}1', maker(seed=i*7+2))))
            songs.append(dict(song_id=sid, mode=mode, audio_sha256=f'{i:064x}', duration_us=130_000_000, references=refs))
            primary = B.load_chart(tape, B.primary_reference(songs[-1])['target_path'], mode)
            good[sid] = state(primary)
            rng = np.random.default_rng(i)
            kept = [n for n in primary if rng.random() < .25]+[(n[0]+70_000, n[1]+70_000, n[2], n[3]) for n in maker(seed=i*7+3)[:120]]
            bad[sid] = state(sorted(kept))
    return songs, good, bad


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.d = d
        (d/'tape').mkdir()
        cls.floor_path = tiny_floor(d)
        cls.songs, good, bad = make_panel(d/'tape')
        (d/'wb.json').write_text(json.dumps(dict(panel=dict(songs=cls.songs, split='val', panel_sha256='x'))))
        (d/'good.json').write_text(json.dumps(good))
        (d/'bad.json').write_text(json.dumps(bad))
        (d/'seeds.json').write_text(json.dumps({'0': good, '1': bad}))
        for name in ('good', 'bad'):
            B.main(['score', '--drafts', str(d/f'{name}.json'), '--panel', str(d/'wb.json'), '--tape', str(d/'tape'), '--floor', str(cls.floor_path),
                    '--timing', str(d/'absent.json'), '--out', str(d/f'{name}_v3.json'), '--workers', '1', '--label', name])
        cls.good = json.loads((d/'good_v3.json').read_text())
        cls.bad = json.loads((d/'bad_v3.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_identity_draft_scores_top_on_w(self):
        for mode, m in self.good['modes'].items():
            self.assertEqual(m['W']['single_exact']['cer']['mean'], 0.)
            self.assertEqual(m['W']['nearest_exact']['W'], 1.)  # capped
            self.assertEqual(m['W']['single_exact']['W'], 1.)
            self.assertGreater(m['W']['single_exact']['W_raw'], 1.)
            self.assertTrue(m['constraints']['pass'], m['constraints']['failed'])
            self.assertAlmostEqual(m['constraints']['note_ratio_median'], 1.)
            self.assertEqual(m['constraints']['f1_20_max']['mean'], 1.)

    def test_best_shift_separates_clocks(self):
        with tempfile.TemporaryDirectory() as t:
            d = {s['song_id']: state([(a+24_000, b+24_000, k, l) for a, b, k, l in B.load_chart(self.d/'tape', B.primary_reference(s)['target_path'], s['mode'])]) for s in self.songs}
            f = Path(t)/'shifted.json'
            f.write_text(json.dumps(d))
            B.main(['score', '--drafts', str(f), '--panel', str(self.d/'wb.json'), '--tape', str(self.d/'tape'), '--floor', str(self.floor_path), '--timing', str(self.d/'absent.json'), '--out', str(Path(t)/'s.json'), '--workers', '1'])
            doc = json.loads((Path(t)/'s.json').read_text())
            for m in doc['modes'].values():
                self.assertGreater(m['W']['single_exact']['cer']['mean'], .05)
                self.assertLess(m['W']['single_best']['cer']['mean'], .001)
                self.assertEqual(m['W']['median_shift_ms'], 4.)  # smallest shift inside the 20 ms keep tolerance

    def test_scaling_between_null_and_human(self):
        cer = np.array([.7, .7])
        self.assertAlmostEqual(B.ratio_score(cer, np.full(2, .9), np.full(2, .5)), .5)
        b = B.boot_ratio(np.full(5, .9), np.full(5, .9), np.full(5, .5))
        self.assertEqual(b['W'], 0.)
        self.assertEqual(B.boot_ratio(np.full(5, .5), np.full(5, .9), np.full(5, .5))['W'], 1.)

    def test_incomplete_and_note_ratio_fail_constraints(self):
        with tempfile.TemporaryDirectory() as t:
            drafts = json.loads((self.d/'good.json').read_text())
            for sid in drafts:
                drafts[sid]['complete'] = False
                drafts[sid]['events'] = drafts[sid]['events'][::2]
            f = Path(t)/'x.json'
            f.write_text(json.dumps(drafts))
            B.main(['score', '--drafts', str(f), '--panel', str(self.d/'wb.json'), '--tape', str(self.d/'tape'), '--floor', str(self.floor_path), '--timing', str(self.d/'absent.json'), '--out', str(Path(t)/'x_v3.json'), '--workers', '1'])
            doc = json.loads((Path(t)/'x_v3.json').read_text())
            for m in doc['modes'].values():
                self.assertFalse(m['constraints']['pass'])
                self.assertIn('complete', m['constraints']['failed'])
                self.assertIn('note_ratio', m['constraints']['failed'])

    def test_test_split_reference_stops_scoring(self):
        with tempfile.TemporaryDirectory() as t:
            tape = Path(t)/'tape'
            tape.mkdir()
            songs, good, _ = make_panel(tape, n_per_mode=2, multi=False)
            write_chart(tape, 'ph:00', phira_chart(seed=1), split='test')
            (Path(t)/'wb.json').write_text(json.dumps(dict(panel=dict(songs=songs, split='val'))))
            (Path(t)/'d.json').write_text(json.dumps(good))
            with self.assertRaises(PermissionError):
                B.main(['score', '--drafts', str(Path(t)/'d.json'), '--panel', str(Path(t)/'wb.json'), '--tape', str(tape), '--floor', str(self.floor_path), '--timing', 'none', '--out', str(Path(t)/'o.json'), '--workers', '1'])

    def test_seed_average_and_samples_file(self):
        with tempfile.TemporaryDirectory() as t:
            B.main(['score', '--drafts', str(self.d/'seeds.json'), '--seeds', '0,1', '--panel', str(self.d/'wb.json'), '--tape', str(self.d/'tape'), '--floor', str(self.floor_path), '--timing', 'none', '--out', str(Path(t)/'o.json'), '--workers', '1'])
            doc = json.loads((Path(t)/'o.json').read_text())
            for mode, m in doc['modes'].items():
                expected = (self.good['modes'][mode]['W']['single_exact']['cer']['mean']+self.bad['modes'][mode]['W']['single_exact']['cer']['mean'])/2
                self.assertAlmostEqual(m['W']['single_exact']['cer']['mean'], expected, places=6)
                self.assertEqual(m['H']['seeds'], 2)

    def test_compare_finds_better_arm(self):
        floor = B.Floor(self.floor_path)
        rep = B.compare_docs(self.good, self.bad, floor, samples=60)
        for mode, m in rep['modes'].items():
            self.assertEqual(m['axes']['W_nearest_exact']['call'], 'A')
            self.assertEqual(m['objective_calls']['W_nearest_exact'], 'A')
        # The admission gate rewrites `verdict` when an objective axis is unmeasured; the raw call stays in objective_verdict.
        self.assertIn(rep['objective_verdict'], ('A dominates B', 'no dominance (trade-off)'))
        self.assertFalse(rep['promotion_allowed'])
        self.assertIn('M', ' '.join(rep['promotion_blockers']))
        self.assertIn('VERDICT', B.compare_text(rep))

    def test_compare_self_ties_and_goes_to_cheaper(self):
        floor = B.Floor(self.floor_path)
        tie = B.compare_docs(self.good, self.good, floor, samples=30)
        self.assertIn('tie', tie['verdict'])
        cheap = B.compare_docs(self.good, self.good, floor, cost_a=1., cost_b=1.02, samples=30)
        self.assertEqual(cheap['cost']['call'], 'tie')  # under 5%
        self.assertIn('tie', cheap['objective_verdict'])  # costs that close are not a reason to switch
        self.assertEqual(cheap['cost']['a'], 1.)
        dom = B.compare_docs(self.good, self.good, floor, cost_a=2., cost_b=1., samples=30)
        self.assertEqual(dom['objective_verdict'], 'B dominates A')

    def test_compare_refuses_different_panels(self):
        floor = B.Floor(self.floor_path)
        other = dict(self.bad, songs_sha256='different')
        with self.assertRaises(ValueError):
            B.compare_docs(self.good, other, floor, samples=10)

    def test_noise_report_shape(self):
        floor = B.Floor(self.floor_path)
        rep = B.noise_report(self.bad, floor, [4, 8], other=self.good, reps=10, aa_docs=[self.bad, self.bad], inner=10, aa_reps=2)
        for mode in rep:
            self.assertGreaterEqual(rep[mode]['8']['H']['sd'], 0)
            self.assertIn('paired_diff', rep[mode]['8'])
            self.assertEqual(rep[mode]['aa_false_alarm']['8']['W_nearest_exact'], 0.)  # identical drafts cannot alarm

    def test_axis_call_rules(self):
        self.assertEqual(B.axis_call(dict(diff=.05, ci80=[.02, .08])), 'A')
        self.assertEqual(B.axis_call(dict(diff=-.05, ci80=[-.08, -.02])), 'B')
        self.assertEqual(B.axis_call(dict(diff=.05, ci80=[-.01, .1])), 'tie')  # CI touches 0
        self.assertEqual(B.axis_call(dict(diff=.005, ci80=[.001, .009])), 'tie')  # under epsilon
        self.assertEqual(B.axis_call(dict(diff=.05, ci80=[.02, .08]), -1), 'B')  # lower is better


class MusicEndToEnd(unittest.TestCase):
    def test_score_with_features_dir(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t/'tape').mkdir()
            (t/'mel').mkdir()
            floor = tiny_floor(t)
            rng = np.random.default_rng(5)
            songs, good, off = [], {}, {}
            timing = {}
            beat = 500_000
            for mode in ('phira', 'mania_4k'):
                for i in range(10):
                    hits = np.array([k*beat for k in range(4, 230) if rng.random() < .45], np.int64)
                    mel = rng.normal(0, .15, (12000, 16))
                    for h_ in hits:
                        mel[int(h_//10_000):int(h_//10_000)+4] += 2.
                    sha = hashlib_like(f'{mode}{i}')
                    (t/'mel'/sha).mkdir()
                    np.save(t/'mel'/sha/'mel10_complete.npy', mel.astype(np.float16))
                    lanes = (rng.permutation(len(hits)) % 4) if mode == 'mania_4k' else np.full(len(hits), -1)
                    ref = [(int(h_), int(h_), 'tap', int(l)) for h_, l in zip(hits, lanes)]
                    cid = f'{mode[:2]}:{i}'
                    timing[cid] = [[0., float(beat), 4]]
                    songs.append(dict(song_id=f'{mode}:{i}', mode=mode, audio_sha256=sha, duration_us=120_000_000, references=[dict(chart_id=cid, target_path=write_chart(t/'tape', cid, ref))]))
                    good[f'{mode}:{i}'] = state(ref)
                    other = np.array([k*beat for k in range(4, 230) if rng.random() < .45], np.int64)  # on grid, ignores the sounds
                    off[f'{mode}:{i}'] = state([(int(h_), int(h_), 'tap', int(l)) for h_, l in zip(other, (rng.permutation(len(other)) % 4) if mode == 'mania_4k' else np.full(len(other), -1))])
            (t/'wb.json').write_text(json.dumps(dict(panel=dict(songs=songs, split='val'))))
            (t/'timing.json').write_text(json.dumps(dict(charts={k: dict(valid=True, segments=v) for k, v in timing.items()})))
            (t/'g.json').write_text(json.dumps(good))
            (t/'o.json').write_text(json.dumps(off))
            common = ['--panel', str(t/'wb.json'), '--tape', str(t/'tape'), '--floor', str(floor), '--timing', str(t/'timing.json'), '--features-dir', str(t/'mel'), '--workers', '1']
            B.main(['score', '--drafts', str(t/'g.json'), '--out', str(t/'g_v3.json')]+common)
            B.main(['score', '--drafts', str(t/'o.json'), '--out', str(t/'o_v3.json')]+common)
            g = json.loads((t/'g_v3.json').read_text())
            o = json.loads((t/'o_v3.json').read_text())
            for mode in g['modes']:
                mg, mo = g['modes'][mode]['M'], o['modes'][mode]['M']
                self.assertTrue(mg['available'])
                self.assertEqual(mg['coverage'], 10)
                self.assertTrue(mg['grid_controlled'])
                self.assertTrue(mg['admitted_vs_nulls'])
                self.assertAlmostEqual(mg['M'], 1., places=3)  # the sounds' own notes equal the human reference
                self.assertFalse(mo['admitted_vs_nulls'])
                self.assertIsNone(mo['M'])
                self.assertIsNotNone(g['modes'][mode]['constraints']['silence'])
            rep = B.compare_docs(g, o, B.Floor(floor), samples=40)
            self.assertEqual(rep['modes']['mania_4k']['axes']['M']['call'], 'A')


def hashlib_like(s):
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()


if __name__ == '__main__':
    unittest.main()
