import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MertGraphDiagnosticTests(unittest.TestCase):
    def test_selection_and_worker_are_exact_shape_graph_not_batch(self):
        selection = json.loads((ROOT / 'reports/full-database-cloud-v1/mert-graph-diagnostic-selection-v1.json').read_text())
        self.assertEqual(selection['songs'], 898)
        self.assertEqual(selection['repeated_shape_groups'], 128)
        self.assertGreater(selection['total_audio_seconds'], 100000)
        text = (ROOT / 'beatfcn/cloud/full_database_mert_graph_app.py').read_text()
        self.assertIn('class ShapeGraph', text)
        self.assertIn('self.graph.replay()', text)
        self.assertIn('model.model._encode(self.input, True)', text)
        self.assertIn('runner.output[:, 0]', text)
        self.assertIn('in_flight_groups=24', text)
        self.assertIn('runner.close()', text)
        self.assertIn('previous_key = key', text)
        self.assertNotIn('ready[index:index + 8]', text)
        self.assertNotIn("'-threads', '1'", (ROOT / 'beatfcn/ingest/pipeline.py').read_text())
        self.assertNotIn('decode_payload', (ROOT / 'beatfcn/ingest/pipeline.py').read_text())
        self.assertNotIn('encode_windows_batch(model', text)

    def test_launcher_does_not_claim_formal_85_acceptance(self):
        text = (ROOT / 'scripts/run_full_database_mert_graph_diagnostic.py').read_text()
        self.assertIn("diagnostic_only=True", text)
        self.assertIn('formal_85_percent_acceptance_claim=False', text)
        self.assertIn("result.get('success_ids'", text)
        self.assertNotIn("utilization.get('passed') is not True", text)


if __name__ == '__main__':
    unittest.main()
