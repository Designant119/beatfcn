"""Adversarial checks: counters must detect timing, density and empty-output shortcuts."""
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from beatfcn.events.canonical import Event, LaneLoc, LineLoc
from beatfcn.events.requests import build_request
from beatfcn.model.benchmark import (audio_features, music_metrics, product_metrics, compare_vocabulary,
                                     capability_metrics, scorecard, beat_grid_metrics)
from beatfcn.model.control_eval import spearman
from beatfcn.model.evaluation import evaluate_songs
from events_helpers import synthetic_mania, synthetic_phira, DURATION
from test_p1_model import CFG, song
from beatfcn.model.p1 import P1Model


def ev(t, lane=0, hold=0):
    return Event(f'{t}-{lane}', t, t+hold, 'hold' if hold else 'tap', LaneLoc(lane))


def audio(duration=30_000_000):
    return {'onsets_us': np.arange(500_000, duration, 500_000),
            'strong_onsets_us': np.arange(500_000, duration, 500_000),
            'duration_us': duration, 'flux': np.arange(duration // 10_000, dtype=float),
            'energy_proxy': np.arange(duration // 10_000, dtype=float)}


class CapabilityBenchmarkTests(unittest.TestCase):
    def test_tied_rank_correlation_and_constant_arrays(self):
        x, y = [1, 1, 2, 3], [3, 2, 1, 1]
        expected = np.corrcoef([.5, .5, 2, 3], [3, 2, .5, .5])[0, 1]
        self.assertAlmostEqual(spearman(x, y), expected)
        self.assertAlmostEqual(spearman(x[::-1], y[::-1]), expected)
        self.assertIsNone(spearman([0, 0, 0], [1, 2, 3]))
        self.assertIsNone(spearman([1, 2, float('nan')], [1, 2, 3]))

    def test_alignment_detects_shift_and_chords_do_not_multiply_recall(self):
        a = audio()
        aligned = [ev(int(t), lane) for t in a['onsets_us'] for lane in range(4)]
        runs = [(0, a['duration_us'])]
        correct, _ = music_metrics(aligned, runs, a)
        wrong, _ = music_metrics([replace(e, onset_us=e.onset_us+100_000) for e in aligned], runs, a)
        self.assertEqual(correct['anchor_20ms_precision'], 1)
        self.assertEqual(correct['anchor_20ms_strong_recall'], 1)
        self.assertEqual(correct['anchor_20ms_shift_excess'], 1)
        self.assertEqual(wrong['anchor_20ms_precision'], 0)
        partial, _ = music_metrics(aligned[:4], runs, a)
        self.assertAlmostEqual(partial['anchor_20ms_strong_recall'], 1 / len(a['onsets_us']))
        a['strong_onsets_us'] = [490_000, 510_000]
        partial, _ = music_metrics([ev(500_000)], runs, a)
        self.assertEqual(partial['anchor_20ms_strong_recall'], .5)

    def test_dense_spam_has_little_chance_corrected_alignment(self):
        a = audio()
        spam = [ev(t) for t in range(0, 30_000_000, 10_000)]
        m, _ = music_metrics(spam, [(0, 30_000_000)], a)
        self.assertEqual(m['anchor_20ms_strong_recall'], 1)
        self.assertLess(m['anchor_20ms_precision'], .15)
        self.assertLess(abs(m['anchor_20ms_shift_excess']), .03)

    def test_empty_silence_missing_and_partial_blocks(self):
        a = audio()
        m, _ = music_metrics([], [(0, 30_000_000)], a)
        self.assertIsNone(m['anchor_20ms_precision'])
        self.assertEqual(m['anchor_20ms_strong_recall'], 0)
        self.assertIsNone(m['density_flux_spearman'])
        m, coverage = music_metrics([ev(500_000)], [(100_000, 1_900_000)], a)
        self.assertEqual(coverage['full_blocks'], 0)
        self.assertIsNone(m['density_flux_spearman'])
        self.assertEqual(music_metrics([], [], None)[1]['status'], 'missing_audio')
        quiet = audio_features(np.zeros((3000, 128), np.float32), 30_000_000)
        self.assertEqual(len(quiet['onsets_us']), 0)
        m, c = music_metrics([ev(500_000)], [(0, 30_000_000)], quiet)
        self.assertIsNone(m['anchor_20ms_precision'])
        self.assertEqual(c['status'], 'no_detected_onsets')

    def test_audio_features_are_deterministic_and_validate_clock(self):
        mel = np.zeros((3000, 128), np.float32)
        mel[50:55] = 3
        a, b = audio_features(mel, 30_000_000), audio_features(mel, 30_000_000)
        self.assertEqual(a['mel_sha256'], b['mel_sha256'])
        self.assertEqual(len(a['onsets_us']), 1)
        self.assertLessEqual(abs(a['onsets_us'][0]-500_000), 10_000)
        truncated = audio_features(mel, 30_039_800)
        self.assertEqual(truncated['uncovered_tail_us'], 39_800)
        _, counts = music_metrics([ev(30_020_000)], [(0, 30_039_800)], truncated)
        self.assertEqual(counts['notes_outside_audio_coverage'], 1)
        for bad, duration in ((mel, 31_000_000), (mel[:10], 100_000), (mel*np.nan, 30_000_000)):
            with self.assertRaises(ValueError): audio_features(bad, duration)

    def test_playability_includes_hold_release_and_phira_is_not_mania(self):
        events = [ev(0, hold=490_000), ev(500_000), ev(550_000)]
        p, c = product_metrics(events, [(0, 1_000_000)], 'mania_4k')
        self.assertEqual(p['same_lane_min_gap_ms'], 50)
        self.assertEqual(p['release_to_repress_min_ms'], 10)
        self.assertEqual(p['rapid_same_lane_fraction'], .5)
        phira = [Event('p', 0, 0, 'tap', LineLoc('0', 'above', 0.))]
        p, c = product_metrics(phira, [(0, 1_000_000)], 'phira')
        self.assertIsNone(p['hand_onset_imbalance'])
        self.assertIsNone(p['lane_bigram_entropy'])
        self.assertFalse(c['lane_applicable'])

    def test_ngrams_are_chord_order_invariant_and_do_not_bridge_holes(self):
        notes = [ev(10, 0), ev(10, 1), ev(500_000, 3), ev(900_000, 1)]
        p = product_metrics(notes, [(0, 1_000_000)], 'mania_4k')[0]
        q = product_metrics(notes[::-1], [(0, 1_000_000)], 'mania_4k')[0]
        self.assertEqual(p['lane_bigram_entropy'], q['lane_bigram_entropy'])
        d = compare_vocabulary(notes, notes[::-1], [(0, 1_000_000)], 'mania_4k')
        self.assertEqual(d['lane_bigram_js_bits'], 0)
        p, c = product_metrics([ev(10), ev(900_000)], [(0, 100), (800_000, 1_000_000)], 'mania_4k')
        self.assertEqual(c['iois'], 0)
        self.assertEqual(c['same_lane_pairs'], 0)
        self.assertIsNone(p['lane_bigram_entropy'])

    def test_local_beat_grid_handles_tempo_change_without_phase_fitting(self):
        beats = [0, 500_000, 1_000_000, 1_250_000, 1_500_000]
        g = beat_grid_metrics([ev(250_000), ev(1_125_000)], [(0, 1_500_000)], beats)
        self.assertEqual(g['subdivision_2']['within_20ms'], 1)
        self.assertEqual(g['subdivision_1']['within_20ms'], 0)

    def test_scorecard_keeps_failures_empty_and_duplicate_song_denominators(self):
        req = {'regions': [{'start_us':0, 'end_us':30_000_000, 'state':'editable'}], 'mode':'mania_4k'}
        truth = [ev(500_000), ev(1_000_000)]
        r = {'song':'s', 'mode':'mania_4k', 'complete':True, 'legality':[], 'request_kind':'from_scratch',
             'benchmark':capability_metrics(truth, truth, req, audio())}
        empty = {**r, 'song':'empty', 'benchmark':capability_metrics([], truth, req, audio())}
        failed = {**r, 'song':'failed', 'complete':False}
        card = scorecard([r, r, empty, failed])
        cell = next(c for c in card['cells'] if c['mode']=='mania_4k' and c['request_kind']=='from_scratch')
        self.assertEqual(cell['songs'], 3)
        self.assertEqual(cell['requests'], 4)
        self.assertEqual(cell['eligible_nonempty'], 2)
        self.assertEqual(cell['metrics']['anchor_20ms_precision']['generated_songs'], 1)
        self.assertIsNone(cell['metrics']['anchor_20ms_precision']['human_p05_p95'])
        self.assertEqual(card['status'], 'not_calibrated')
        self.assertTrue(any(c['status']=='not_run' for c in card['cells']))
        json.dumps(card, allow_nan=False)

    def test_whole_song_evaluator_scores_all_request_types_and_retains_inputs(self):
        charts = {'mania': synthetic_mania(), 'phira': synthetic_phira()}
        rows = [{'chart_id':key, 'audio_sha256':key, 'mode':chart.mode, 'split':'val', 'audio_sec':DURATION/1e6}
                for key, chart in charts.items()]
        corpus = SimpleNamespace(train_rows=lambda rows:rows, chart=charts.__getitem__, song=lambda key,in_memory:song(0))
        with patch('beatfcn.model.sample._pick', return_value=0):
            out = evaluate_songs(P1Model(CFG).eval(), corpus, rows, n_songs=2,
                                 request_kinds=('from_scratch', 'middle_holes', 'reference'))
        self.assertEqual(out['requested'], 6)
        self.assertEqual(len(out['songs']), 6)
        for r in out['songs']:
            self.assertIn('request', r)
            self.assertIn('truth_events', r)
            self.assertEqual(r['benchmark']['audio']['status'], 'missing_audio')
        json.dumps(out, allow_nan=False)

    def test_frozen_manifest_integrity_and_sample_seeds(self):
        from beatfcn.model.evaluation import evaluate_benchmark
        from scripts.capability_benchmark import freeze
        from tempfile import TemporaryDirectory
        from pathlib import Path
        charts = {'mania':synthetic_mania(), 'phira':synthetic_phira()}
        rows = [{'chart_id':k, 'audio_sha256':k, 'mode':c.mode, 'split':'val', 'audio_sec':DURATION/1e6,
                 'kept':True, 'source':'osu', 'approved':1, 'n_events':len(c.events),
                 'offset_decision':{'decision_version':2}} for k,c in charts.items()]
        with TemporaryDirectory() as folder:
            file = Path(folder)/'rows.jsonl'
            file.write_text('\n'.join(json.dumps(r) for r in rows))
            manifest = freeze(SimpleNamespace(rows=str(file), songs=2, seed=1234))
        corpus = SimpleNamespace(train_rows=lambda rows:rows, chart=charts.__getitem__, song=lambda key,in_memory:song(0))
        with patch('beatfcn.model.sample._pick', return_value=0):
            out = evaluate_benchmark(P1Model(CFG).eval(), corpus, manifest)
        expected = {(r['chart_id'],r['kind']):r['seed'] for r in manifest['requests']}
        self.assertEqual({(r['chart_id'],r['request_kind']):r['seed'] for r in out['songs']}, expected)
        manifest['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            evaluate_benchmark(None, None, manifest)


if __name__ == '__main__':
    unittest.main()
