from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MertDecodeProbeTests(unittest.TestCase):
    def test_selection_is_balanced_and_probe_is_cpu_only(self):
        import json
        selection = json.loads((ROOT / 'reports/full-database-cloud-v1/mert-decode-probe-selection-v1.json').read_text())
        self.assertEqual(selection['songs'], 192)
        self.assertEqual(sum(job['source'] == 'osu' for job in selection['jobs']), 96)
        self.assertEqual(sum(job['source'] == 'phira' for job in selection['jobs']), 96)
        app = (ROOT / 'beatfcn/cloud/mert_decode_probe_app.py').read_text()
        self.assertNotIn("gpu='", app)
        self.assertIn("worker_counts = contract.get('worker_counts')", app)
        self.assertIn('4 * (len(wave) // CELL_SAMPLES) != job[\'expected_mel_frames\']', app)
        pipeline = (ROOT / 'beatfcn/train/cuda_pipeline.py').read_text()
        self.assertIn('return audio, original, wave', pipeline)
        probe_cleanup = (ROOT / 'beatfcn/cloud/mert_decode_probe_app.py').read_text()
        self.assertIn('_cleanup(original)', probe_cleanup)

    def test_launcher_charges_no_gpu_for_cpu_probe(self):
        from scripts.run_full_database_mert_decode_probe import RESOURCES, normalized_rates, upper_bound
        text = (ROOT / 'scripts/run_full_database_mert_decode_probe.py').read_text()
        self.assertNotIn("'gpu':", text)
        self.assertIn("resources.get('gpu')", (ROOT / 'beatfcn/train/data_preparation_budget.py').read_text())
        upper = upper_bound(RESOURCES, normalized_rates())
        self.assertLess(upper, 0.15)

    def test_v2_launcher_scales_cpu16_workers(self):
        from scripts.run_full_database_mert_decode_probe_v2 import RESOURCES, RUN_ID, upper_bound, normalized_rates
        text = (ROOT / 'scripts/run_full_database_mert_decode_probe_v2.py').read_text()
        self.assertEqual(RUN_ID, 'full-db-mert-decode-probe-v2')
        self.assertEqual(RESOURCES['cpu'], 16)
        self.assertIn('worker_counts=[32, 48, 64, 64, 48, 32]', text)
        self.assertIn('cpu=(16, 16)', text)
        upper = upper_bound(RESOURCES, normalized_rates())
        self.assertLess(upper, 0.35)


if __name__ == '__main__':
    unittest.main()
