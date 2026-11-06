import unittest

import numpy as np

from beatfcn.train.self_timing import fit_constant, fit_grid


def phases(t, bpm, offset, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    th = 2 * np.pi * (t - offset) / beat + rng.normal(0, noise, len(t))
    bar = 2 * np.pi * (t - offset) / (4 * beat) + rng.normal(0, noise, len(t))
    return np.stack([np.sin(th), np.cos(th), np.sin(bar), np.cos(bar)], 1)


class SelfTimingTest(unittest.TestCase):
    def test_constant_tempo_recovered(self):
        t = np.arange(0, 120, 0.01)
        for bpm, off in ((128.0, 0.137), (174.5, 0.05), (90.0, 0.4)):
            beat, offset, coh = fit_constant(t, np.arctan2(*phases(t, bpm, off, 0.3)[:, :2].T))
            self.assertAlmostEqual(60 / beat, bpm, delta=0.02)
            d = (offset - off) % beat
            self.assertLess(min(d, beat - d), 0.004)
            self.assertGreater(coh, 0.9)

    def test_grid_segments_and_downbeat(self):
        t = np.arange(0, 90, 0.01)
        segs, diag = fit_grid(t, phases(t, 150.0, 0.3, 0.25))
        self.assertIsNotNone(segs)
        self.assertEqual(len(segs), 1)
        start_us, beat_us, meter = segs[0]
        self.assertAlmostEqual(beat_us, 400_000, delta=100)
        self.assertEqual(meter, 4)
        # downbeat: the bar starts at offset 0.3 s (+ k bars)
        bar = 4 * beat_us
        d = (start_us - 300_000) % bar
        self.assertLess(min(d, bar - d), 5_000)

    def test_noise_gives_no_grid(self):
        t = np.arange(0, 60, 0.01)
        rng = np.random.default_rng(1)
        pred = rng.normal(0, 1, (len(t), 4))
        segs, diag = fit_grid(t, pred)
        self.assertIsNone(segs)
        self.assertEqual(diag['reason'], 'low_coherence')

    def test_tempo_change_piecewise(self):
        t1 = np.arange(0, 60, 0.01); t2 = np.arange(60, 120, 0.01)
        p1 = phases(t1, 120.0, 0.0, 0.2); p2 = phases(t2, 160.0, 60.0, 0.2, seed=2)
        segs, diag = fit_grid(np.concatenate([t1, t2]), np.concatenate([p1, p2]))
        self.assertIsNotNone(segs)
        bpms = sorted(round(60e6 / s[1]) for s in segs)
        self.assertIn(120, bpms)
        self.assertIn(160, bpms)

    def test_short_early_tempo_section_is_split(self):
        t1 = np.arange(0, 26, 0.01); t2 = np.arange(26, 120, 0.01)
        p1 = phases(t1, 250.0, 0.0, 0.2); p2 = phases(t2, 260.0, 26.0, 0.2, seed=3)
        segs, diag = fit_grid(np.concatenate([t1, t2]), np.concatenate([p1, p2]))
        self.assertIsNotNone(segs)
        bpms = sorted(round(60e6 / s[1]) for s in segs)
        self.assertIn(250, bpms)
        self.assertIn(260, bpms)

    def test_octave_windows_are_not_a_tempo_change(self):
        t1 = np.arange(0, 40, 0.01); t2 = np.arange(40, 60, 0.01); t3 = np.arange(60, 120, 0.01)
        pred = np.concatenate([phases(t1, 150.0, 0.0, 0.2), phases(t2, 300.0, 0.0, 0.2, seed=4),
                               phases(t3, 150.0, 0.0, 0.2, seed=5)])
        segs, diag = fit_grid(np.concatenate([t1, t2, t3]), pred)
        self.assertEqual(len(segs), 1)
        self.assertAlmostEqual(60e6 / segs[0][1], 150.0, delta=0.05)


def _notes(bpm, offset_ms, seconds, start_s=0.0, jitter_ms=2.0, seed=0, div=2):
    rng = np.random.default_rng(seed)
    beat = 60e6 / bpm
    t = np.arange(start_s * 1e6 + offset_ms * 1e3, (start_s + seconds) * 1e6, beat / div)
    return t + rng.normal(0, jitter_ms * 1e3, len(t))


class RefineGridTest(unittest.TestCase):
    def test_refine_snaps_tempo_drift_to_the_round_bpm(self):
        from beatfcn.train.self_timing import grid_score, refine_grid
        on = _notes(222.0, 10.0, 180.0)
        est = [[0.0, 60e6 / 222.112, 4]]
        out, d = refine_grid(est, on, 180e6)
        assert d['changed'] and abs(60e6 / out[0][1] - 222.0) < 1e-6
        assert grid_score(out, on, 180e6) > 0.95 > grid_score(est, on, 180e6)


    def test_refine_fixes_a_constant_phase_bias(self):
        from beatfcn.train.self_timing import grid_score, refine_grid
        on = _notes(200.0, 16.0, 120.0)
        out, d = refine_grid([[0.0, 300e3, 4]], on, 120e6)
        assert d['changed'] and grid_score(out, on, 120e6) > 0.95
        assert abs(60e6 / out[0][1] - 200.0) < 1e-6


    def test_refine_finds_a_tempo_change(self):
        from beatfcn.train.self_timing import refine_grid
        on = np.concatenate([_notes(250.0, 4.0, 26.6), _notes(260.0, 9.0, 120.0, start_s=26.6)])
        out, d = refine_grid([[0.0, 60e6 / 260.0, 4]], on, 146.6e6)
        assert d['changed'] and d['bpm'] == [250.0, 260.0]
        assert abs(out[1][0] - 26.6e6) < 1.5e6


    def test_refine_keeps_a_good_grid_and_a_true_non_round_tempo(self):
        from beatfcn.train.self_timing import refine_grid
        good = [[4e3, 60e6 / 180.0, 4]]
        on = _notes(180.0, 4.0, 120.0)
        assert refine_grid(good, on, 120e6)[1]['changed'] is False
        est = [[4e3, 60e6 / 174.27, 4]]
        on = _notes(174.27, 4.0, 180.0)
        out, _ = refine_grid(est, on, 180e6)
        assert abs(60e6 / out[0][1] - 174.27) < 0.01


    def test_refine_handles_no_input(self):
        from beatfcn.train.self_timing import refine_grid
        assert refine_grid(None, [], 1e6)[1]['changed'] is False
        est = [[0.0, 3e5, 4]]
        assert refine_grid(est, [1.0, 2.0], 1e6)[0] is est


if __name__ == '__main__':
    unittest.main()
