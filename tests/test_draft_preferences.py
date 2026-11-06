import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import scripts.draft_song as ds


def args(**kw):
    base = dict(mode='phira', hold_head_ratio=None, chord_ratio=None, drag_ratio=None, flick_ratio=None, hold_occupancy_ratio=None)
    return SimpleNamespace(**{**base, **kw})


class StyleBandsTest(unittest.TestCase):
    def test_value_snaps_to_its_table_band(self):
        self.assertEqual(ds.style_bands(args(hold_head_ratio=.35)), {'hold_head_ratio': [.3, .4]})

    def test_empty_when_nothing_is_requested(self):
        self.assertEqual(ds.style_bands(args()), {})

    def test_field_of_the_other_mode_is_refused(self):
        with self.assertRaises(SystemExit):
            ds.kind_bias(args(mode='mania_4k', flick_ratio=.1))

    def test_kind_share_maps_to_a_bias_with_the_right_sign(self):
        up, down = ds.kind_bias(args(drag_ratio=.15)), ds.kind_bias(args(flick_ratio=.02))
        self.assertGreater(up['drag'], .5)
        self.assertLess(down['flick'], -.5)

    def test_value_out_of_range_is_refused(self):
        with self.assertRaises(SystemExit):
            ds.style_bands(args(chord_ratio=1.5))


class DensityCurveFileTest(unittest.TestCase):
    def test_curve_cannot_mix_with_a_level_request(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d)/'curve.json'
            f.write_text(json.dumps([dict(start_s=0, end_s=8, density=4.)]))
            a = args(density_curve=f, tier='low', density=None, phira_level=None, mania_star=None, guidance=None)
            with self.assertRaises(SystemExit):
                ds.condition_request(a, SimpleNamespace(c=SimpleNamespace(intensity_dim=1)), 100_000)


if __name__ == '__main__':
    unittest.main()
