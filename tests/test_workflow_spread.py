import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import run_workflow_benchmark as W

DURATION = 30_000_000


def _write(root, name, step, lane_shift=0):
    onset = np.arange(0, DURATION, step, dtype=np.int64)
    arrays = dict(onset_us=onset, end_us=onset.copy(), kind=np.zeros(len(onset), np.int64),
                  lane=((np.arange(len(onset)) + lane_shift) % 4).astype(np.int64))
    path = Path(root)/f'{name}.npz'
    np.savez(path, **arrays)
    return path.name, hashlib.sha256(path.read_bytes()).hexdigest(), len(onset)


class SpreadPanelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        charts = []
        for song in range(6):
            audio = '%064x' % (song + 1)
            specs = [('easy', 1_000_000), ('normal', 500_000), ('hard', 250_000)] if song < 5 else [('only', 500_000)]
            for name, step in specs:
                path, sha, heads = _write(self.root, f's{song}-{name}', step, song)
                stratum = {'easy': 'density/low', 'normal': 'density/medium', 'hard': 'density/high'}.get(name, 'density/medium')
                charts.append(dict(chart_id=f'osu:{song}{name}', audio_sha256=audio, mode='mania_4k', split='val',
                                   duration_us=DURATION, component=f'k{song}', stratum=stratum, target_path=path,
                                   target_sha256=sha, canonical_sha256=f'c{song}{name}', requests=[dict(event_count=heads)]))
        self.manifest = dict(charts=charts)

    def tearDown(self):
        self.tmp.cleanup()

    def test_target_has_denser_nearest_sibling_and_only_target_is_scored(self):
        panel = W.freeze_spread_panel(self.manifest, 'x', max_songs=10, seed=3)
        self.assertEqual(len(panel['songs']), 5)                        # the single-chart song is not eligible
        order = {'easy': 0, 'normal': 1, 'hard': 2}
        for song in panel['songs']:
            (target,) = song['references']
            t, s = target['chart_id'].split(':')[1], song['sibling']['chart_id'].split(':')[1]
            self.assertEqual(t[0], s[0])                                # same song
            self.assertEqual(order[s[1:]], order[t[1:]] + 1)            # nearest denser difficulty
            self.assertGreater(song['sibling']['heads'], song['target_heads'])
            self.assertIs(W.primary_reference(song), target)
        self.assertEqual(panel, W.freeze_spread_panel(self.manifest, 'x', max_songs=10, seed=3))

    def test_thinning_and_baselines(self):
        notes = [(t, t, 'tap', lane) for t in range(0, 4_000_000, 250_000) for lane in (0, 1)]
        thin = W.thin_to_count(notes, 10)
        self.assertEqual(len(thin), 10)
        self.assertTrue(all(sum(1 for x in thin if x[0] == t) == 1 for t in {x[0] for x in thin}))  # chords dropped first
        self.assertEqual(W.thin_to_count(notes, 100), notes)
        panel = W.freeze_spread_panel(self.manifest, 'x', max_songs=10, seed=3)
        base = W.spread_baselines(self.root, panel, samples=200, seed=0)
        self.assertEqual(len(base['rows']), 5)
        for row in base['rows']:
            self.assertGreater(row['notes_ratio'], 1)
            self.assertLessEqual(row['oracle_thinned_cer'], row['sibling_cer'])
        self.assertIsNotNone(base['thinned_cer']['mean'])


if __name__ == '__main__':
    unittest.main()


class ShapedPresetTest(unittest.TestCase):
    def test_shape_from_first_pass_level_from_tier(self):
        from beatfcn.model.full_event_codec import EventRequest
        panel = dict(songs=[dict(song_id='mania_4k:a', mode='mania_4k', duration_us=8_000_000,
                                 references=[dict(chart_id='osu:1', stratum='density/high', coarse_stratum='high')]),
                            dict(song_id='mania_4k:b', mode='mania_4k', duration_us=8_000_000,
                                 references=[dict(chart_id='osu:2', stratum='density/low', coarse_stratum='low')])])
        presets = {('mania_4k', 'high'): 12.0, ('mania_4k', 'low'): 4.0}
        # first pass of song a: 2 notes/s in the first 4 s, 6 notes/s in the last 4 s (mean 4/s); song b empty
        events = [dict(onset_us=t, end_us=t, type='tap', lane=0) for t in range(0, 4_000_000, 500_000)]
        events += [dict(onset_us=t, end_us=t, type='tap', lane=1) for t in range(4_000_000, 8_000_000, 166_667)]
        provider = W.shaped_preset_provider(panel, presets, {'mania_4k:a': events, 'mania_4k:b': []}, cell_us=100_000)
        request = EventRequest('mania_4k', 0, 8_000_000, 8_000_000)
        rec = provider(dict(chart_id='mania_4k:a', mode='mania_4k'), request)['record']
        mean = len(events) / 8.0
        self.assertAlmostEqual(rec['shape_factor'], 12.0 / mean)
        self.assertEqual(rec['label'], 'charter_difficulty_preset_shaped')
        flat = provider(dict(chart_id='mania_4k:b', mode='mania_4k'), request)['record']
        self.assertIsNone(flat['shape_factor'])
        self.assertEqual(flat['level'], 4.0)


class JackAwareConsensusTest(unittest.TestCase):
    def test_vote_stitched_jacks_are_removed_and_off_is_unchanged(self):
        # 3 samples of a 90 ms stream; two alternate keys 0/1 with opposite parity, one has a timing wobble
        times = list(range(0, 1_800_000, 90_000))
        a = [(t, t, 'tap', (i % 2)) for i, t in enumerate(times)]
        b = [(t, t, 'tap', ((i + 1) % 2)) for i, t in enumerate(times)]
        c = [(t + 5_000, t + 5_000, 'tap', (i % 2) if i % 3 else ((i + 1) % 2)) for i, t in enumerate(times)]
        plain, info0 = W.consensus_draft([a, b, c], 'mania_4k', 10_000_000)
        fixed, info = W.consensus_draft([a, b, c], 'mania_4k', 10_000_000, min_jack_us=100_000)
        def jacks(notes):
            last, n = {}, 0
            for t, _, _, lane in sorted(notes):
                n += lane in last and t - last[lane] < 100_000
                last[lane] = t
            return n
        self.assertGreater(jacks(plain), 0)                      # per-cluster voting stitched repeats together
        self.assertEqual(jacks(fixed), 0)
        self.assertEqual(len(fixed), len(plain))                 # same onsets, only lanes re-chosen
        self.assertGreater(info['jack_rechosen'], 0)
        self.assertNotIn('min_jack_us', info0)
        again, _ = W.consensus_draft([a, b, c], 'mania_4k', 10_000_000, min_jack_us=None)
        self.assertEqual(again, plain)

    def test_playability_reports_jacks_under_100ms(self):
        from beatfcn.train.workflow_metrics import playability
        notes = [(0, 0, 'tap', 0), (80_000, 80_000, 'tap', 0), (300_000, 300_000, 'tap', 0)]
        p = playability(notes, 'mania_4k')
        self.assertEqual((p['jack_lt_100ms'], p['jack_lt_40ms']), (1, 0))


class JackFallbackTest(unittest.TestCase):
    def test_single_candidate_repeats_are_relaned_with_their_hold_tails(self):
        # two identical samples -> one candidate per cluster; key 0 hit every 60 ms, last one a hold
        times = list(range(0, 600_000, 60_000))
        notes = [(t, t, 'tap', 0) for t in times[:-1]] + [(times[-1], times[-1] + 400_000, 'hold', 0)]
        fixed, info = W.consensus_draft([notes, list(notes)], 'mania_4k', 5_000_000, min_jack_us=100_000)
        last = {}
        for t, _, _, lane in sorted(fixed):
            self.assertTrue(lane not in last or t - last[lane] >= 100_000)
            last[lane] = t
        self.assertEqual(len(fixed), len(notes))
        self.assertGreater(info['jack_relaned'], 0)
        hold = [n for n in fixed if n[2] == 'hold']
        self.assertEqual(len(hold), 1)
        self.assertEqual(hold[0][1] - hold[0][0], 400_000)          # tail carried from the voted key
