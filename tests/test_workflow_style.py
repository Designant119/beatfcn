import unittest
from types import SimpleNamespace
import numpy as np
from scripts import run_workflow_benchmark as W
from beatfcn.model.full_event_codec import EventRequest

class StyleRequestTests(unittest.TestCase):
    def test_parse_and_provider_and_response(self):
        bands = W.parse_control_bands(['hold_occupancy_ratio=0.6,0.8', 'jack_ratio=0.0,0.02'])
        self.assertEqual(bands, {'hold_occupancy_ratio': [0.6, 0.8], 'jack_ratio': [0.0, 0.02]})
        with self.assertRaises(SystemExit):
            W.parse_control_bands(['hold_occupancy_ratio=0.61,0.8'])
        prov = W.style_condition_provider(bands)
        panel = dict(songs=[dict(song_id='mania_4k:a', mode='mania_4k', duration_us=10_000_000),
                            dict(song_id='phira:b', mode='phira', duration_us=10_000_000)])
        drafts = {'mania_4k:a': dict(events=[dict(onset_us=0, end_us=5_000_000, type='hold', lane=0),
                                             dict(onset_us=6_000_000, end_us=6_000_000, type='tap', lane=0),
                                             dict(onset_us=6_100_000, end_us=6_100_000, type='tap', lane=0)]),
                  'phira:b': dict(events=[dict(onset_us=0, end_us=0, type='tap', lane=0)])}
        r = W.style_response(drafts, panel, bands)
        m = r['summary']['mania_4k/hold_occupancy_ratio']
        self.assertAlmostEqual(m['mean'], 0.5); self.assertEqual(m['in_band'], 0.0)
        self.assertEqual(r['summary']['mania_4k/jack_ratio']['mean'], 1.0)
        self.assertNotIn('phira/jack_ratio', r['summary'])          # no jack field for Phira
        self.assertIn('phira/hold_occupancy_ratio', r['summary'])

if __name__ == '__main__':
    unittest.main()
