from pathlib import Path
import json
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MertBatchGraphProbeTests(unittest.TestCase):
    def test_batch_graph_requires_exact_parity_and_three_phases(self):
        text = (ROOT / 'beatfcn/cloud/mert_batch_graph_probe_app.py').read_text()
        self.assertIn('class BatchShapeGraph', text)
        self.assertIn("report['max_abs']==0", text)
        self.assertIn("report['float16_exact_fraction']==1", text)
        self.assertIn('batch.replay(padded)', text)
        self.assertIn('batch.output[:, index].float().cpu().numpy()', text)
        self.assertIn('model.model._encode(batch.input,True)', text)
        self.assertIn("phases=dict(single_graph=single_phase,batch_graph=batch_phase,batch_eager=eager_phase)", text)
        self.assertIn("len(contract['jobs']) != contract.get('batch')", text)
        self.assertNotIn('feature writes', text)

    def test_batch2_selection_is_two_exact_shape_songs(self):
        selection = json.loads(
            (ROOT / 'reports/full-database-cloud-v1/mert-batch2-graph-probe-selection-v1.json').read_text())
        self.assertEqual(selection['batch'], 2)
        self.assertEqual(len(selection['jobs']), 2)
        keys = {job['expected_mel_frames'] // 4 + 1 for job in selection['jobs']}
        self.assertEqual(len(keys), 1)
        self.assertTrue(all(job['audio_sec'] <= 360 for job in selection['jobs']))
        self.assertEqual({job['source'] for job in selection['jobs']}, {'osu', 'phira'})

    def test_launcher_is_bounded_batch2_diagnostic(self):
        text = (ROOT / 'scripts/run_mert_batch_graph_probe.py').read_text()
        self.assertIn("probe='mert_batch_cuda_graph_v1'", text)
        self.assertIn('batch=2,', text)
        self.assertIn("RUN_ID = 'full-db-mert-batch2-graph-probe-v1'", text)
        self.assertIn("len(result.get('parity', [])) != len(contract['jobs'])", text)
        self.assertIn("result.get('feature_writes') != 0", text)
        self.assertIn('formal_85_percent_acceptance_claim=False', text)


if __name__ == '__main__':
    unittest.main()
