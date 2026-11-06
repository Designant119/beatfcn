from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from scripts import train_monitor as monitor


class TrainingHorizonMonitorTests(unittest.TestCase):
    def test_training_fit_is_separate_and_tampered_predictions_hide_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = root / 'reports'
            reports.mkdir()
            plan = root / 'plan.json'
            plan.write_text('{}')
            predictions = root / 'predictions-100.json'
            predictions.write_text('[]')
            summary = {part: {'all': dict(full_note_f1=score, timing_point_f1=score, teacher_loss=1.)}
                       for part, score in [('train', 1.), ('heldout', .02)]}
            result = dict(complete=True, optimizer_updates=1000, diagnostic_horizon_benefit=False,
                plan_sha256=sha256(plan.read_bytes()).hexdigest(), new_cloud_spend_usd=0,
                checkpoints={'100': dict(summary=summary, assets_sha256={
                    'predictions-100.json': sha256(predictions.read_bytes()).hexdigest()})})
            (reports / 'result.json').write_text(json.dumps(result))
            (reports / 'resource.json').write_text(json.dumps({'complete': True, 'exit_code': 0}))
            target = {'training_horizon': dict(result='result.json', resources='resource.json',
                       plan='plan.json', script='probe.py', steps=1000)}
            snapshot = monitor.Monitor().snapshot('old', now=20)
            monitor.training_horizon_status(snapshot, target, reports, command_lookup=lambda _: '')
            self.assertEqual(snapshot['training_horizon']['status'], 'failed')
            self.assertFalse(snapshot['training_horizon']['model_promotion'])
            rendered = monitor.render_text(snapshot)
            self.assertIn('训练完整F1', rendered)
            self.assertIn('留出完整F1', rendered)
            self.assertIn('100.0%', rendered)
            self.assertIn('2.0%', rendered)
            predictions.write_text('[{}]')
            monitor.training_horizon_status(snapshot, target, reports, command_lookup=lambda _: '')
            self.assertEqual(snapshot['training_horizon']['checkpoints'], {})
            self.assertNotIn('100.0%', monitor.render_text(snapshot))


if __name__ == '__main__':
    unittest.main()
