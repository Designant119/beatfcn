import unittest
from types import SimpleNamespace

import numpy as np

from beatfcn.train import exposure_loader as E


class SongStyleTest(unittest.TestCase):
    def setUp(self):
        # 100 notes, the last 50 are holds; the first window holds none of them
        n = 100
        self.arrays = dict(onset_us=np.arange(n)*1_000_000, end_us=np.arange(n)*1_000_000+(np.arange(n) >= 50)*500_000,
                           kind=(np.arange(n) >= 50).astype(int), lane=np.zeros(n, int))
        self.orig = E.chart_arrays
        E.chart_arrays = lambda panel, chart: self.arrays
        self.ds = SimpleNamespace(history_seed=1, song_style=1., _song_bands_cache={}, panel=None)

    def tearDown(self):
        E.chart_arrays = self.orig

    def call(self, audit):
        cls = next(v for v in vars(E).values() if isinstance(v, type) and hasattr(v, '_song_style_shown'))
        return cls._song_style_shown(self.ds, 0, dict(chart_index=7), SimpleNamespace(mode='phira', duration_us=100_000_000), 0, audit)

    def test_shown_style_field_takes_the_whole_chart_band(self):
        shown = self.call(dict(shown_fields=['hold_head_ratio'], bands=dict(hold_head_ratio=[0., .02])))
        self.assertEqual(shown['hold_head_ratio'], [.5, .6])

    def test_density_fields_stay_with_the_window(self):
        window = [4., 5.]
        shown = self.call(dict(shown_fields=['density_notes_per_second'], bands=dict(density_notes_per_second=window)))
        self.assertEqual(shown['density_notes_per_second'], window)

    def test_nothing_shown_stays_unconditional(self):
        self.assertIsNone(self.call(dict(shown_fields=[], bands={})))

    def test_probability_zero_keeps_the_window(self):
        self.ds.song_style = 0.
        self.assertIsNone(self.call(dict(shown_fields=['hold_head_ratio'], bands=dict(hold_head_ratio=[0., .02]))))


if __name__ == '__main__':
    unittest.main()
