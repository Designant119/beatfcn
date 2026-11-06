"""Charter-timing grid metrics, charter_timing_snap, pattern-level human likeness, clock-shifted raw grids."""
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

from beatfcn.train import requirement_diagnostics as D
from beatfcn.train import workflow_metrics as W

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_workflow_benchmark as wb  # noqa: E402

MS = 1000
LN2 = math.log(2)


def t(ms, lane=0, typ='tap', end=None):
    return (int(ms*MS), int((ms if end is None else end)*MS), typ, lane)


class GridMetrics(unittest.TestCase):
    GRID = [(0., 500_000.)]  # 120 BPM from 0

    def test_subdivision_classes_take_the_coarsest_hit(self):
        beat = 500.
        times = [0, beat/2, beat/3, beat/4, beat/6, beat/8, beat/12, beat/16, 1000+beat/12+7]
        got = W.subdivision_classes([x*MS for x in times], self.GRID)
        self.assertEqual([W.SUBDIVISION_LABELS[k] for k in got],
                         ['1/1', '1/2', '1/3', '1/4', '1/6', '1/8', '1/12', '1/16', 'off'])
        # 4 ms is on-grid (tolerance 5 ms), 6 ms is not.
        self.assertEqual(W.subdivision_classes([1_004_000, 1_006_000], self.GRID).tolist(), [0, 8])

    def test_next_red_line_start_counts_as_on_beat(self):
        grid = [(0., 500_000.), (1_010_000., 400_000.)]  # BPM change 10 ms after the old 1/1 point at 1 s
        got = W.subdivision_classes([1_007_000, 1_410_000], grid)
        self.assertEqual([W.SUBDIVISION_LABELS[k] for k in got], ['1/1', '1/1'])

    def test_grid_comparison(self):
        ref = [t(0, 0), t(0, 1), t(250, 2), t(500, 3), t(666.667, 0)]
        same = W.grid_comparison(ref, ref, self.GRID)
        self.assertEqual(same['draft_on_grid_fraction'], 1.)
        self.assertEqual(same['subdivision_js_nats'], 0.)
        self.assertEqual(same['reference_onsets'], 4)  # the chord is one onset
        shifted = W.grid_comparison([(a+10_000, b+10_000, k, l) for a, b, k, l in ref], ref, self.GRID)
        self.assertEqual(shifted['draft_on_grid_fraction'], 0.)
        self.assertAlmostEqual(shifted['subdivision_js_nats'], LN2)  # all mass in 'off' vs none there
        self.assertEqual(shifted['on_grid_abs_diff'], 1.)

    def test_charter_timing_snap_repairs_offsets_and_keeps_far_notes(self):
        grid60 = [(0., 1_000_000.)]  # 60 BPM: union points around a beat are >= 62.5 ms apart
        ref = [t(1000, 0), t(2000, 1), t(3000, 2, 'hold', 3500)]
        draft = [t(1025, 0), t(1976, 1), t(3024, 2, 'hold', 3512)]  # 20-30 ms late/early: retimes
        raw = W.chart_edit_rate(draft, ref, 'mania_4k')['cer']
        r = W.charter_timing_snap(draft, ref, 'mania_4k', grid60)
        self.assertGreater(raw, 0)
        self.assertEqual(r['cer'], 0.)
        self.assertEqual(r['note_f1_50ms'], 1.)
        self.assertEqual(r['label'], 'charter_timing_snap')
        self.assertEqual(r['snap_stats']['heads_snapped'], 3)
        slow = [(0., 2_000_000.)]  # 30 BPM: 1/16 = 125 ms; 62 ms is > 30 ms from every point
        kept = W.charter_timing_snap([t(62)], [t(0)], 'mania_4k', slow)
        self.assertEqual(kept['snap_stats']['heads_snapped'], 0)
        self.assertEqual(kept['raw_note_f1_50ms'], kept['note_f1_50ms'])

    def test_chance_on_grid_matches_uniform_onsets(self):
        rng = np.random.default_rng(0)
        for bpm in (60, 180):
            beat = 60e6/bpm
            onsets = rng.uniform(0, 1e8, 50_000)
            empirical = float((W.subdivision_classes(onsets, [(0., beat)]) < len(W.CHARTER_DIVISORS)).mean())
            self.assertAlmostEqual(W.chance_on_grid(beat), empirical, delta=.01)
        self.assertAlmostEqual(W.chance_on_grid(1_000_000), .24, places=3)  # 24 distinct points (16 + 12 - 4) x 10 ms / 1 s

    def test_note_f1_counts_lane_for_mania_only(self):
        f = W.note_f1([t(0, 1)], [t(10, 0)], 'mania_4k')
        self.assertEqual((f['note_f1'], f['onset_f1']), (0., 1.))
        self.assertEqual(W.note_f1([t(0, -1)], [t(10, -1)], 'phira')['note_f1'], 1.)
        self.assertIsNone(W.note_f1([], [], 'phira')['note_f1'])


class ManiaProfile(unittest.TestCase):
    def test_constructed_chart(self):
        notes = [t(0, 0), t(0, 1), t(0, 2), t(150, 0), t(500, 1), t(1000, 2, 'hold', 2000), t(1000, 3)]
        p = W.pattern_profile(notes, 'mania_4k', duration_us=4_000_000)
        self.assertEqual(p['group_size'], [2, 1, 1, 0])  # onsets with 3, 1, 1, 2 notes
        v = W.profile_values(p)
        self.assertAlmostEqual(v['group_size_mean'], (3+1+1+2)/4)
        # lane 0: 0->150 ms is a jack; lane 1: 0->500 not; lane 2: 0->1000 not -> 1/3
        self.assertAlmostEqual(v['jack_rate_200ms'], 1/3)
        self.assertAlmostEqual(v['hold_occupancy'], 1_000_000/(4*4_000_000))
        self.assertEqual(p['lane_counts'], [2, 2, 2, 1])
        self.assertLess(v['lane_balance_entropy'], 1.)
        # density: 2 s bins; all 7 heads fall in [0, 2 s)
        self.assertEqual(p['density_rates'], [3.5, 0.])
        self.assertAlmostEqual(v['density_var_2s'], np.var([3.5, 0.]))
        # transitions: {0,1,2}->{0} (3 pairs, 1/3 each), {0}->{1}, {1}->{2,3} (1/2 each)
        tr = np.asarray(p['transition'])
        self.assertAlmostEqual(tr.sum(), 3.)
        self.assertAlmostEqual(tr[0, 0], 1/3)
        self.assertAlmostEqual(v['same_key_transition'], (1/3)/3)

    def test_balanced_lanes_and_identical_charts(self):
        notes = [t(i*250, i % 4) for i in range(16)]
        v = W.profile_values(W.pattern_profile(notes, 'mania_4k'))
        self.assertAlmostEqual(v['lane_balance_entropy'], 1.)
        self.assertEqual(v['jack_rate_200ms'], 0.)
        cmp = W.compare_patterns(notes, notes, 'mania_4k', duration_us=4_000_000)['metrics']
        for name, m in cmp.items():
            self.assertEqual(m['abs_diff'], 0., name)
            if 'js_nats' in m:
                self.assertEqual(m['js_nats'], 0., name)

    def test_empty_draft_is_maximally_unlike(self):
        ref = [t(i*250, i % 4) for i in range(8)]
        cmp = W.compare_patterns([], ref, 'mania_4k', duration_us=2_000_000)['metrics']
        self.assertEqual(cmp['chord_size']['js_status'], 'one_side_empty')
        self.assertAlmostEqual(cmp['chord_size']['js_nats'], LN2)
        self.assertIsNone(cmp['chord_size']['abs_diff'])


class PhiraProfile(unittest.TestCase):
    def test_runs_holds_simultaneous_and_transitions(self):
        notes = [t(0, -1, 'drag'), t(100, -1, 'drag'), t(200, -1, 'drag'), t(300, -1, 'tap'), t(300, -1, 'flick'),
                 t(400, -1, 'drag'), t(600, -1, 'hold', 900), t(700, -1, 'flick'), t(800, -1, 'flick')]
        p = W.pattern_profile(notes, 'phira')
        v = W.profile_values(p)
        self.assertEqual(p['run_lengths']['drag'], [3, 1])
        self.assertEqual(p['run_lengths']['flick'], [1, 2])
        self.assertEqual(p['runs']['drag'], [1, 0, 1, 0, 0, 0])
        self.assertAlmostEqual(v['drag_run_mean'], 2.)
        self.assertAlmostEqual(v['hold_fraction'], 1/9)
        self.assertAlmostEqual(v['multi_note_fraction'], 1/8)
        self.assertEqual(p['type_counts'], [1, 1, 3, 4])
        tr = np.asarray(p['transition'])
        drag, flick = W.PHIRA_KINDS.index('drag'), W.PHIRA_KINDS.index('flick')
        self.assertAlmostEqual(tr[drag, drag], 2.)
        self.assertAlmostEqual(tr[flick, flick], 1.)
        self.assertAlmostEqual(tr.sum(), 7.)

    def test_merge_equals_concatenation_for_counts(self):
        a = [t(0, -1, 'drag'), t(100, -1, 'tap'), t(100, -1, 'flick')]
        b = [t(5000, -1, 'drag'), t(5100, -1, 'drag'), t(5200, -1, 'hold', 5500)]
        m = W.merge_profiles([W.pattern_profile(a, 'phira'), W.pattern_profile(b, 'phira')])
        whole = W.pattern_profile(a+b, 'phira')
        for key in ('type_counts', 'group_size', 'notes', 'groups'):
            self.assertEqual(m[key], whole[key], key)
        self.assertEqual(sorted(m['run_lengths']['drag']), [1, 2])
        with self.assertRaises(ValueError):
            W.merge_profiles([W.pattern_profile(a, 'phira'), W.pattern_profile([t(0)], 'mania_4k')])


class IoiUnits(unittest.TestCase):
    def test_beat_units_need_charter_timing(self):
        notes = [t(i*250) for i in range(9)]
        fast, slow = [(0., 500_000.)], [(0., 1_000_000.)]
        v = W.profile_values(W.pattern_profile(notes, 'mania_4k', grid=fast))
        self.assertAlmostEqual(v['ioi_beat'], .5)
        self.assertAlmostEqual(v['ioi_ms'], 250.)
        self.assertAlmostEqual(W.profile_values(W.pattern_profile(notes, 'mania_4k', grid=slow))['ioi_beat'], .25)
        self.assertNotIn('ioi_beat', W.compare_patterns(notes, notes, 'mania_4k')['metrics'])
        cmp = W.compare_patterns(notes, [t(i*125) for i in range(17)], 'mania_4k', grid=fast)
        self.assertTrue(cmp['charter_timing'])
        self.assertAlmostEqual(cmp['metrics']['ioi_beat']['js_nats'], LN2)  # 1/2 vs 1/4 beat: disjoint bins
        # the bin edges put every canonical value in its own bin
        for value in W.IOI_BEAT_VALUES:
            self.assertEqual(np.histogram([value], W.IOI_BEAT_EDGES)[0].tolist().index(1),
                             W.IOI_BEAT_VALUES.index(value))


class BenchmarkV2PatternProfile(unittest.TestCase):
    def test_pooled_windows_per_mode(self):
        req = lambda mode, start: dict(mode=mode, start_us=start, end_us=start+8_000_000)  # noqa: E731
        ev = lambda ms, typ='tap', lane=0: dict(onset_us=ms*MS, end_us=ms*MS, type=typ, lane=lane)  # noqa: E731
        human = [ev(0), ev(250, lane=1), ev(500, lane=2)]
        pairs = [(human, human, req('mania_4k', 0)), ([ev(8000+k*30, lane=0) for k in range(5)],
                 [ev(8000, lane=0), ev(8500, lane=0)], req('mania_4k', 8_000_000)),
                 ([ev(0, 'drag', -1)], [ev(0, 'tap', -1)], req('phira', 0))]
        out = D.human_likeness(pairs)['pattern_profile']
        self.assertEqual(sorted(out), ['mania_4k', 'phira'])
        self.assertEqual(out['mania_4k']['windows'], 2)
        self.assertGreater(out['mania_4k']['metrics']['jack_rate_200ms']['model'], 0.)
        self.assertEqual(out['mania_4k']['metrics']['jack_rate_200ms']['human'], 0.)
        self.assertGreater(out['phira']['metrics']['type_mix']['js_nats'], .5)


def osu_bytes(notes, points):
    from beatfcn.events.chart_export import osu_mania_bytes
    return osu_mania_bytes([dict(onset_us=a, end_us=b, type=k, lane=l) for a, b, k, l in notes],
                           timing_points=points)[0]


class ClockShiftedGrid(unittest.TestCase):
    """Full-database targets = raw chart + a constant model-clock shift (e.g. +14 ms); the raw grid must follow."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.raw = self.tmp/'raw'
        (self.raw/'osu/charts').mkdir(parents=True)
        beat = 500_000
        self.native = [(26_000+k*beat//2, 26_000+k*beat//2, 'tap', k % 4) for k in range(40)]
        (self.raw/'osu/charts/77.osu').write_bytes(osu_bytes(self.native, [(26., 120.)]))
        self.target = [(a+14_000, b+14_000, k, l) for a, b, k, l in self.native]

    def test_unshifted_target_uses_shift_zero(self):
        grid, info = wb.reference_grid(self.raw, 'osu:77', self.native)
        self.assertTrue(info['available'], info)
        self.assertEqual((info['clock_shift_us'], info['clock_shift_source']), (0, 'none'))
        self.assertEqual(grid[0], (26_000., 500_000.))

    def test_shift_estimated_without_receipts(self):
        grid, info = wb.reference_grid(self.raw, 'osu:77', self.target)
        self.assertTrue(info['available'], info)
        self.assertEqual((info['clock_shift_us'], info['clock_shift_source']), (14_000, 'estimated'))
        self.assertEqual(grid[0][0], 40_000.)
        self.assertEqual(W.grid_profile(self.target, grid)['on_grid_fraction'], 1.)

    def test_shift_from_receipt_and_identity_checks(self):
        canonical = 'c'*64
        key = hashlib.sha256(f'osu:77:{canonical}'.encode()).hexdigest()
        folder = self.tmp/'receipts/targets'/key
        folder.mkdir(parents=True)
        (folder/'receipt.json').write_text(json.dumps(dict(chart_id='osu:77', canonical_sha256=canonical,
                                                           statistics=dict(model_shift_us=14_000))))
        shifts = wb.ClockShifts(self.tmp/'receipts', {'osu:77': canonical})
        self.assertEqual(shifts('osu:77')[0], 14_000)
        _, info = wb.reference_grid(self.raw, 'osu:77', self.target, clock_shifts=shifts)
        self.assertEqual((info['clock_shift_us'], info['clock_shift_source']), (14_000, 'receipt'))
        self.assertIsNone(wb.ClockShifts(self.tmp/'receipts', {'osu:77': 'd'*64})('osu:77')[0])
        self.assertIsNone(wb.ClockShifts(None, {})('osu:77')[0])

    def test_wrong_chart_stays_unavailable(self):
        rng = np.random.default_rng(0)
        jitter = rng.integers(-90_000, 90_000, size=len(self.native))
        other = [(int(a+d), int(a+d), k, l) for (a, _, k, l), d in zip(self.native, jitter)]
        _, info = wb.reference_grid(self.raw, 'osu:77', other)
        self.assertFalse(info['available'])
        self.assertIn('every tried shift', info['reason'])

    def test_estimate_clock_shift(self):
        self.assertEqual(wb.estimate_clock_shift([t[0] for t in self.target], [n[0] for n in self.native]), 14_000)
        self.assertIsNone(wb.estimate_clock_shift([], [1]))

    def test_coverage_summary(self):
        _, a = wb.reference_grid(self.raw, 'osu:77', self.target)
        _, b = wb.reference_grid(self.raw, 'osu:78', self.target)
        cov = wb.timing_coverage({'osu:77': a, 'osu:78': b})
        self.assertEqual((cov['available'], cov['charts']), (1, 2))
        self.assertEqual(cov['shift_source'], {'estimated:14000': 1})
        self.assertEqual(cov['reasons'], {'raw chart not local': 1})


class PhiraWithoutYaml(unittest.TestCase):
    def test_info_scan_fallback_matches_load_package(self):
        fix = ROOT/'tests/fixtures'
        rpe = next(iter(sorted(fix.rglob('rpe_multibpm_synthetic.json'))), None)
        if rpe is None:
            self.skipTest('fixture missing')
        tmp = Path(tempfile.mkdtemp())
        path = tmp/'5.zip'
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('info.yml', 'name: "Song"\nchart: chart.json\nmusic: song.ogg\noffset: 0.1\n')
            z.writestr('chart.json', rpe.read_bytes())
        with_yaml = wb.load_raw_chart(path, 'phira:5')
        with mock.patch.dict(sys.modules, {'yaml': None}):
            without = wb.load_raw_chart(path, 'phira:5')
        self.assertEqual([e.onset_us for e in with_yaml.events], [e.onset_us for e in without.events])
        self.assertEqual(with_yaml.timing, without.timing)


class WorkflowIntegration(unittest.TestCase):
    def test_rows_aggregate_tables_and_panel_floor(self):
        song = dict(song_id='mania_4k:x', mode='mania_4k', duration_us=10_000_000, references=[])
        grid = [(0., 500_000.)]
        ref_a = [(k*250_000, k*250_000, 'tap', k % 4) for k in range(36)]
        ref_b = [(k*500_000, k*500_000, 'tap', (k+1) % 4) for k in range(18)]
        refs = [dict(chart_id='osu:1', stratum='m', coarse_stratum='m', notes=ref_a),
                dict(chart_id='osu:2', stratum='m', coarse_stratum='m', notes=ref_b)]
        draft = [(a+12_000, b+12_000, k, l) for a, b, k, l in ref_a]
        rows, row = wb.evaluate(draft, dict(draft_id='model'), song, refs, grids={'osu:1': grid}, per_op=False,
                                split='val')
        self.assertIn('grid', rows[0])
        self.assertNotIn('grid', rows[1])
        self.assertEqual(rows[0]['charter_timing_snap']['cer'], 0.)
        self.assertEqual(row['min_ref_charter_snap_cer'], 0.)
        self.assertIn('ioi_beat', rows[0]['pattern']['metrics'])
        self.assertNotIn('ioi_beat', rows[1]['pattern']['metrics'])
        agg = wb.aggregate(rows, [row], samples=20, seed=0)
        ct = agg['charter_timing']['val/mania_4k']
        self.assertEqual((ct['rows'], ct['rows_with_charter_timing']), (2, 1))
        self.assertEqual(ct['draft_on_grid_fraction']['mean'], 0.)
        prof = agg['human_likeness_profile']['val/mania_4k']
        self.assertEqual(prof['metrics']['ioi_beat']['rows'], 1)
        self.assertEqual(prof['metrics']['chord_size']['rows'], 2)
        floor = wb.panel_human_floor(dict(songs=[song], split='val'), {song['song_id']: refs}, {'osu:1': grid},
                                     samples=20, seed=0)
        self.assertEqual((floor['songs'], floor['pairs']), (1, 2))
        lines = '\n'.join(wb.aggregate_tables(agg, floor=floor['aggregate']))
        self.assertIn('Human-likeness profile', lines)
        self.assertIn('floor abs diff', lines)
        self.assertIn('human-vs-human floor, same panel', lines)
        self.assertIn('NEVER capability', lines)
        empty_rows, empty = wb.evaluate([], dict(draft_id='empty'), song, refs, grids={}, per_op=False, split='val')
        self.assertNotIn('charter_timing', wb.aggregate(empty_rows, [empty], samples=10, seed=0))


if __name__ == '__main__':
    unittest.main()
