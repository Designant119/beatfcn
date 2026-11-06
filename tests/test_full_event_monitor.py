from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from scripts import train_monitor as monitor


class FullEventMonitor(unittest.TestCase):
    def test_fixed_batch_metrics_are_separate_and_stale_evidence_is_hidden(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reports = root / 'reports'
            reports.mkdir()
            plan = root / 'plan.json'
            plan.write_text('{}')
            result = dict(complete=True, passed=True, parameters=57922563,
                optimizer_updates=100, plan_sha256=sha256(plan.read_bytes()).hexdigest(),
                initial={'ce': 1.9}, final={'ce': .0000875, 'fields': {
                    'kind': dict(count=26, accuracy=1.)}}, wall_s=93.)
            path = reports / 'result.json'
            path.write_text(json.dumps(result))
            verification = dict(complete=True, passed=True, generated_streams_identical=2,
                generated_events=43, source_sha256={'reports/result.json': sha256(path.read_bytes()).hexdigest()})
            (reports / 'verification.json').write_text(json.dumps(verification))
            target = {'full_event_probe': dict(result='result.json', verification='verification.json', plan='plan.json')}
            snapshot = monitor.Monitor().snapshot('old', now=20)
            old_metrics = dict(snapshot['metrics'])
            monitor.full_event_probe_status(snapshot, target, reports)
            self.assertEqual(snapshot['full_event_probe']['status'], 'passed')
            self.assertEqual(snapshot['metrics'], old_metrics)
            self.assertIn('不是新整曲 benchmark', monitor.render_text(snapshot))
            self.assertIn('57.92M', monitor.render_text(snapshot))
            path.write_text('{}')
            monitor.full_event_probe_status(snapshot, target, reports)
            self.assertEqual(snapshot['full_event_probe']['status'], 'evidence_incomplete')
            self.assertNotIn('57.92M', monitor.render_text(snapshot))


if __name__ == '__main__':
    unittest.main()
