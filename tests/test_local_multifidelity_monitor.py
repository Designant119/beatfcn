import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import train_monitor as monitor


class LocalMultifidelityMonitorTests(unittest.TestCase):
    def test_identity_failure_and_dead_process_never_claim_live_or_quality_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);reports=root/'reports';reports.mkdir()
            plan=root/'plan.json';plan.write_text('{}')
            source=root/'runner.py';source.write_text('# probe')
            result=dict(complete=False,updates=25,plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
                source_sha256={'runner.py':hashlib.sha256(source.read_bytes()).hexdigest()},
                curve=[dict(step=0,mode_val_ce={'phira':4.},generation_summary={
                    'phira':dict(full_note_f1=0.,complete=0,requests=3),
                    'mania_4k':dict(full_note_f1=0.,complete=1,requests=1)})])
            (reports/'result.json').write_text(json.dumps(result))
            target=dict(local_multifidelity=dict(result='result.json',plan='plan.json',pid=123,health_reports=[]))
            snap=monitor.Monitor().snapshot('local',now=20)
            monitor.local_multifidelity_status(snap,target,reports,command_lookup=lambda _: '')
            self.assertEqual(snap['local_multifidelity']['status'],'interrupted')
            self.assertFalse(snap['local_multifidelity']['model_promotion'])
            text=monitor.render_text(snap)
            self.assertIn('Phira完整音符F1',text)
            self.assertIn('0/3',text)
            monitor.local_multifidelity_status(snap,target,reports,
                command_lookup=lambda _: 'python -m scripts.run_local_learning_curve --plan plan.json')
            self.assertEqual(snap['local_multifidelity']['status'],'running')
            source.write_text('# different code')
            monitor.local_multifidelity_status(snap,target,reports,command_lookup=lambda _: '')
            self.assertEqual(snap['local_multifidelity']['status'],'evidence_incomplete')
            self.assertEqual(snap['local_multifidelity']['curve'],[])


if __name__=='__main__':unittest.main()
