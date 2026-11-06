import unittest

import numpy as np

from beatfcn.train.reference_intensity import ReferenceIntensityPanel


class FakeFeatures:
    def window(self, audio, start, end):
        return dict(mel=np.ones((16, 4), np.float32),
            mel_times=np.arange(16, dtype=np.float32)*.125+.0625,
            mel_valid=np.ones(16, bool))


class FakePanel:
    def __init__(self):
        self.records = [
            dict(chart_id='a', audio_sha256='x', split='val', mode='phira',
                 stratum='EZ', chart_index=0,
                 request=dict(start_us=0, end_us=2_000_000), controls=[]),
            dict(chart_id='b', audio_sha256='x', split='val', mode='phira',
                 stratum='HD', chart_index=1,
                 request=dict(start_us=0, end_us=2_000_000), controls=[]),
            dict(chart_id='c', audio_sha256='y', split='val', mode='phira',
                 stratum='HD', chart_index=2,
                 request=dict(start_us=0, end_us=2_000_000), controls=[]),
        ]
        self.features = FakeFeatures()

    def _arrays(self, chart_index):
        onsets = {0: [0, 1_000_000], 1: [0, 500_000, 1_000_000, 1_500_000],
                  2: []}[chart_index]
        return None, dict(onset_us=np.asarray(onsets, np.int64))

    def targets(self, index):
        return []

    def row(self, index, controls=None):
        return dict(inputs={}, sequence={})


class ReferenceIntensityTest(unittest.TestCase):
    def test_nonzero_origin_uses_request_relative_times(self):
        panel = FakePanel()
        panel.records[0]['request'] = dict(start_us=8_000_000, end_us=10_000_000)
        wrapped = ReferenceIntensityPanel(panel, source='audio', version=1)
        np.testing.assert_array_equal(wrapped.curve_for(0)['intensity_times'],
            np.asarray([.25, .75, 1.25, 1.75], np.float32))

    def test_audio_curve_excludes_context_before_request(self):
        class ContextFeatures:
            def window(self, audio, start, end):
                return dict(mel=np.array([[10.], [1.], [2.], [3.], [4.]], np.float32),
                    mel_times=np.array([-.005, .005, .505, 1.005, 1.505], np.float32),
                    mel_valid=np.ones(5, bool))
        panel = FakePanel()
        panel.features = ContextFeatures()
        curve = ReferenceIntensityPanel(panel, source='audio', version=1).curve_for(0)
        np.testing.assert_array_equal(curve['intensity'][:, 0], [.25, .5, .75, 1.])

    def test_chart_donor_and_audio_fallback(self):
        panel = FakePanel()
        wrapped = ReferenceIntensityPanel(panel, source='chart', fallback='audio',
            cell_us=500_000, scale=16.0, hide_probability=0.0, seed=0, version=1)
        first = wrapped.curve_for(0)['intensity'].reshape(-1)
        self.assertEqual(len(first), 4)
        self.assertGreater(first.sum(), 0)
        self.assertTrue(wrapped.sources[0].startswith('chart:b'))
        # No donor for song y: audio fallback produces a non-zero curve.
        third = wrapped.curve_for(2)['intensity'].reshape(-1)
        self.assertEqual(wrapped.sources[2], 'audio')
        self.assertGreater(third.sum(), 0)


if __name__ == '__main__':
    unittest.main()
