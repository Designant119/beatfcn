import importlib.util
from pathlib import Path
import unittest

import numpy as np
from beatfcn.events.requests import build_request
from events_helpers import synthetic_mania, DURATION

spec = importlib.util.spec_from_file_location('condition_benchmark', Path(__file__).parents[1] / 'scripts/condition_benchmark_app.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ConditionBenchmarkTests(unittest.TestCase):
    def test_audio_baseline_ignores_truth_when_selecting_points_and_respects_scope(self):
        chart = synthetic_mania()
        sample = build_request(chart, kind='middle_holes', seed=4, audio_id='a', audio_duration_us=DURATION)
        audio = {'onsets_us': np.arange(0, DURATION, 100_000), 'strong_onsets_us': np.arange(0, DURATION, 400_000)}
        a = module.timing_baselines(audio, sample, list(sample.hidden), chart.mode)
        b = module.timing_baselines(audio, sample, [], chart.mode)
        for field, times in audio.items():
            expected = sum(any(r['start_us'] <= t < r['end_us'] for r in sample.request['regions'] if r['state'] == 'editable')
                           for t in times)
            self.assertEqual(a[field]['20ms']['pred'], expected)
            self.assertEqual(a[field]['20ms']['pred'], b[field]['20ms']['pred'])
            self.assertIn('timing-only', a[field]['purpose'])


if __name__ == '__main__':
    unittest.main()
