from pathlib import Path
import json
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MertEagerFormalTests(unittest.TestCase):
    def test_app_is_concurrent_per_song_reference_path(self):
        text = (ROOT / 'beatfcn/cloud/full_database_mert_eager_app.py').read_text()
        self.assertIn("GPU_STREAMS = 2", text)
        self.assertIn("DECODE_WORKERS = 28", text)
        self.assertIn('torch.cuda.stream(streams[slot % GPU_STREAMS])', text)
        self.assertIn('feature_payloads(', text)
        self.assertIn("feature_payloads(\n                                item['wave'], model, mel=item['mel'])", text)
        self.assertIn('first_decode_done', text)
        self.assertIn('first_encode_done', text)
        self.assertIn('write_slots = threading.Semaphore(WRITE_QUEUE_LIMIT)', text)
        self.assertIn('Unconditional time-based Volume commit', text)
        self.assertNotIn('monitor.summary(), stage_totals=stage', text)
        self.assertIn('write_slots.release()', text)
        self.assertIn('4 * (len(wave) // CELL_SAMPLES) != job[\'expected_mel_frames\']', text)
        self.assertIn('shutil.move(str(folder), str(quarantine_root / audio))', text)
        self.assertNotIn('ShapeGraph', text)
        self.assertNotIn('decode_payload', text)

    def test_app_pipeline_constant_matches_frozen_config(self):
        from beatfcn.cloud.full_database_mert_eager_app import (
            DECODE_QUEUE, DECODE_WORKERS, EXPECTED_PIPELINE, GPU_STREAMS, RECOVERY_SMOKE,
            WRITE_QUEUE_LIMIT)
        self.assertEqual(EXPECTED_PIPELINE, dict(path='mert_per_song_eager_streams_v1',
            gpu_streams=2, decode_workers=28, decode_queue=12, write_workers=8,
            write_queue_limit=24, commit_every=16, verified_raw_receipt_decode=True,
            cpu_mel=True, resume='local_audit_contract_v1', quarantine_invalid=True,
            recovery_smoke=8, instrumented='unconditional_time_commit_v2'))
        self.assertEqual((GPU_STREAMS, DECODE_WORKERS, DECODE_QUEUE, RECOVERY_SMOKE,
            WRITE_QUEUE_LIMIT), (2, 28, 12, 8, 24))

    def test_resume_is_contract_driven_no_container_sweep(self):
        text = (ROOT / 'beatfcn/cloud/full_database_mert_eager_app.py').read_text()
        self.assertIn("resume.get('skip_ids')", text)
        self.assertIn("stage='resume-contract'", text)
        self.assertIn("stage='resume-done'", text)
        self.assertNotIn('iterdir()', text)
        self.assertNotIn("receipt.json').exists()", text)
        self.assertNotIn("pool.map(existing", text)

    def test_launcher_enforces_formal_acceptance_gate(self):
        text = (ROOT / 'scripts/run_full_database_mert_eager_v7.py').read_text()
        self.assertIn("RUN_ID = 'full-db-mert-eager-formal-v7'", text)
        self.assertIn('timeout_s=900', text)
        self.assertIn("census['valid_ids']", text)
        self.assertIn('full_corpus_encode=True', text)
        self.assertIn("gpu='A10G'", text)
        self.assertIn("cpu=16", text)
        self.assertIn("'formal_acceptance_passed'", text)
        self.assertIn('stable_period_mean_utilization_percent', text)
        self.assertIn('last_eight_interval_mean_utilization_percent', text)
        self.assertIn('recovery_smoke', text)

    def test_launcher_covers_full_manifest(self):
        from scripts.run_full_database_mert_eager_v7 import build_contract
        contract = build_contract()
        jobs = [json.loads(line)
            for line in (ROOT / 'inputs/full-database-cloud-v1/mert-jobs.jsonl')
            .read_text().splitlines() if line.strip()]
        self.assertEqual(len(contract['jobs']), len(jobs))
        self.assertEqual(len(contract['jobs']), 10898)
        self.assertTrue(contract['full_corpus_encode'])
        self.assertFalse(contract['diagnostic_only'])
        self.assertTrue(contract['pipeline']['cpu_mel'])
        self.assertEqual(contract['pipeline']['resume'], 'local_audit_contract_v1')
        self.assertEqual(len(contract['resume']['skip_ids']), 9764)
        self.assertEqual(len(contract['resume']['invalid_ids']), 0)


if __name__ == '__main__':
    unittest.main()
