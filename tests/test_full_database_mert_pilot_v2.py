import json
from pathlib import Path
import unittest

from scripts.prepare_full_database_mert_pilot_v2 import RAW_EXCLUDE, select


ROOT = Path(__file__).resolve().parents[1]


class MertPilotV2Tests(unittest.TestCase):
    def test_selection_is_unique_duration_spread_and_includes_v1(self):
        jobs = [dict(audio_sha256=f'a{i}', split='train', audio_sec=float(i)) for i in range(30)]
        jobs += [dict(audio_sha256='val', split='val', audio_sec=2.),
                 dict(audio_sha256=next(iter(RAW_EXCLUDE)), split='train', audio_sec=2.)]
        v1 = [jobs[2], jobs[27]]
        values = select(jobs, v1, count=6)
        self.assertEqual(len(values), 6)
        self.assertEqual(len({job['audio_sha256'] for job in values}), 6)
        self.assertTrue({job['audio_sha256'] for job in v1} <= {job['audio_sha256'] for job in values})
        self.assertTrue(all(job['split'] == 'train' for job in values))
        self.assertFalse(RAW_EXCLUDE & {job['audio_sha256'] for job in values})

    def test_actual_v2_selection_has_real_work_for_the_stable_window(self):
        value = json.loads((ROOT / 'reports/full-database-cloud-v1/mert-pilot-selection-v2.json').read_text())
        v1 = json.loads((ROOT / 'reports/full-database-cloud-v1/mert-pilot-selection-v1.json').read_text())
        self.assertEqual(value['count'], 1536)
        self.assertEqual(len(value['jobs']), 1536)
        self.assertEqual(len({job['audio_sha256'] for job in value['jobs']}), 1536)
        self.assertTrue({job['audio_sha256'] for job in v1['jobs']} <= {
            job['audio_sha256'] for job in value['jobs']})
        self.assertGreater(value['total_audio_seconds'] / 500, 480)
        self.assertLess(value['total_audio_seconds'] / 126.88367365864836, 2400)

    def test_concurrent_worker_is_bounded_and_has_no_local_entrypoint(self):
        text = (ROOT / 'beatfcn/cloud/full_database_concurrent_app.py').read_text()
        self.assertIn('GPU_STREAMS = 2', text)
        self.assertIn('IN_FLIGHT = 8', text)
        self.assertIn('torch.cuda.Stream()', text)
        self.assertIn('models = [model, FrozenMert2', text)
        self.assertIn('max_workers=1', text)
        self.assertIn('independent_models=2', text)
        self.assertIn('return_when=FIRST_COMPLETED', text)
        self.assertIn('data.reload()', text)
        self.assertNotIn('@app.local_entrypoint', text)


if __name__ == '__main__':
    unittest.main()
