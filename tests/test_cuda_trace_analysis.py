import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_cuda_trace import analyze, union


class TraceAnalysisTests(unittest.TestCase):
    def test_overlapping_streams_count_only_once(self):
        self.assertEqual(union([(10, 30), (20, 40), (50, 60)]), [[10, 40], [50, 60]])
        events = [{'ph': 'X', 'cat': 'kernel', 'name': 'k', 'ts': t, 'dur': d}
                  for t, d in [(0, 30000), (10000, 30000), (80000, 20000)]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.json'
            path.write_text(json.dumps({'traceEvents': events}))
            timeline = analyze(path)['timeline']
        self.assertAlmostEqual(timeline['device_busy_fraction'], .6)
        self.assertEqual(timeline['gaps_over_20ms'], 1)
        self.assertAlmostEqual(timeline['gaps_over_20ms_total_s'], .04)

    def test_partial_download_cannot_claim_device_utilization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'partial'
            path.write_text('{"traceEvents": [{"ph":"X","cat":"kernel","name":"k","ts":0,"dur":5},')
            with self.assertRaisesRegex(ValueError, 'incomplete trace'):
                analyze(path)
            result = analyze(path, allow_partial=True)
            self.assertFalse(result['complete'])
            self.assertIsNone(result['timeline'])


if __name__ == '__main__':
    unittest.main()
