from pathlib import Path
import json
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MertEagerStreamsProbeTests(unittest.TestCase):
    def test_app_requires_bitwise_identity_and_stream_phases(self):
        text = (ROOT / 'beatfcn/cloud/mert_eager_streams_probe_app.py').read_text()
        self.assertIn("report['max_abs'] == 0", text)
        self.assertIn("report['float16_exact_fraction'] == 1", text)
        self.assertIn('torch.cuda.stream(streams[slot % stream_count])', text)
        self.assertIn('PHASE_STREAMS = (1, 2, 3, 4)', text)
        self.assertIn('CONCURRENCY_ROUNDS = 2', text)
        self.assertIn("probe='mert_eager_streams_v1'", text)
        self.assertNotIn('feature writes', text)

    def test_selection_covers_decode_paths_and_windowing(self):
        selection = json.loads(
            (ROOT / 'reports/full-database-cloud-v1/mert-eager-streams-selection-v1.json').read_text())
        self.assertEqual(len(selection['jobs']), 4)
        sources = {job['source'] for job in selection['jobs']}
        self.assertEqual(sources, {'osu', 'phira'})
        self.assertTrue(any(job['audio_sec'] > 360 for job in selection['jobs']))
        self.assertTrue(any(250 < job['audio_sec'] < 350 for job in selection['jobs']))
        self.assertEqual(sum(80 < job['audio_sec'] < 100 for job in selection['jobs']), 2)

    def test_launcher_is_bounded_diagnostic(self):
        text = (ROOT / 'scripts/run_mert_eager_streams_probe.py').read_text()
        self.assertIn("RUN_ID = 'full-db-mert-eager-streams-probe-v1'", text)
        self.assertIn("probe='mert_eager_streams_v1'", text)
        self.assertIn("result.get('feature_writes') != 0", text)
        self.assertIn('concurrent_vs_serial_max_abs=0', text)
        self.assertIn('formal_85_percent_acceptance_claim=False', text)


if __name__ == '__main__':
    unittest.main()
