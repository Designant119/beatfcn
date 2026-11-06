import json
from pathlib import Path
import unittest

from beatfcn.cloud.full_database_batch_app import _gpu_batch_ready


ROOT = Path(__file__).resolve().parents[1]


class MertBatchV3Tests(unittest.TestCase):
    def test_selection_has_real_work_for_gate_and_lifecycle(self):
        value = json.loads((ROOT / 'reports/full-database-cloud-v1/mert-batch-selection-v3.json').read_text())
        v2 = json.loads((ROOT / 'reports/full-database-cloud-v1/mert-pilot-selection-v2.json').read_text())
        self.assertEqual(value['count'], 3072)
        self.assertEqual(len(value['jobs']), 3072)
        self.assertEqual(len({job['audio_sha256'] for job in value['jobs']}), 3072)
        self.assertTrue({job['audio_sha256'] for job in v2['jobs']} <= {
            job['audio_sha256'] for job in value['jobs']})
        self.assertGreater(value['total_audio_seconds'] / 1000, 480)
        self.assertLess(value['total_audio_seconds'] / 563.2127669824057, 3600)

    def test_early_stop_flushes_partial_gpu_batch(self):
        self.assertTrue(_gpu_batch_ready([object()], {}, 10, 100, {}, True))
        self.assertTrue(_gpu_batch_ready([object()] * 3, {}, 100, 100, {}, False))
        self.assertFalse(_gpu_batch_ready([object()], {}, 10, 100, {}, False))
        self.assertFalse(_gpu_batch_ready([object()], {object(): 'running'}, 100, 100, {}, True))

    def test_batch_worker_preflights_parity_before_stable_sampling(self):
        text = (ROOT / 'beatfcn/cloud/full_database_batch_app.py').read_text()
        self.assertLess(text.index('parity = _parity_preflight'), text.index('monitor.start()'))
        self.assertIn('encode_windows_batch(model', text)
        self.assertIn('window_batch=WINDOW_BATCH', text)
        self.assertIn("feature_parity(reference['layers'], candidate['layers'])", text)
        self.assertIn('max_workers=1', text)
        self.assertNotIn('@app.local_entrypoint', text)


if __name__ == '__main__':
    unittest.main()
