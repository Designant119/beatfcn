import unittest
from unittest.mock import patch
from types import SimpleNamespace
import tempfile
from pathlib import Path

from scripts import monitor_600m as monitor


class Monitor600mTests(unittest.TestCase):
    def test_replayed_log_cannot_rewind_current_call_progress(self):
        state=monitor.State();state.call_id='call'
        state.apply(dict(phase='train',step=100,updates=6,elapsed=30))
        record=state.ingest_line('call {"phase":"train","step":90,"elapsed":20}')
        self.assertIsNone(record)
        self.assertEqual(state.latest()['step'],100)
    def setUp(self):
        mock=patch.object(monitor,'watch_contract',return_value=dict(
            operator_override=dict(minimum_util=70),quality_eval_protocol=dict(panel_sha256='test')))
        mock.start();self.addCleanup(mock.stop)
    def test_evaluation_download_confirmation_does_not_hide_results(self):
        state=monitor.State();state.call_id='call'
        contract=dict(active_run=dict(call_id='call',evaluation_remote='report.json',plan_sha256='sha'))
        result=SimpleNamespace(returncode=0,stdout='{"plan_sha256":"sha","checkpoints":[]}\n✓ Finished downloading files to local!\n')
        with tempfile.TemporaryDirectory()as folder, patch.object(monitor,'WATCH',Path(folder)), \
                patch.object(monitor,'watch_contract',return_value=contract), \
                patch.object(monitor.subprocess,'run',return_value=result):
            monitor.refresh_evaluation(state)
            self.assertEqual(state.evaluation['plan_sha256'],'sha')
            self.assertTrue((Path(folder)/'evaluation_latest.json').is_file())

    def test_eval_is_visible_before_training_and_is_cleared_on_new_call(self):
        state=monitor.State()
        state.ingest_line('fc-one {"phase":"learning_eval_done","results":[{"optimizer_updates":228,"summary":{"complete_cases":8,"expected":8,"teacher_joint_ce":1.2,"macro_note_f1_50ms":0.1,"hold_joint_f1_50ms":0,"silence_false_events":3,"silence_cases":2,"per_type":{"hold":{"f1":0}}}}]}')
        state.evaluation=dict(protocol_sha256='test',checkpoints=[dict(optimizer_updates=228,sha256='checkpoint',
            records=[dict(case_role='positive',teacher_joint_ce=1.2,prediction=dict(complete=False,events=[]))],
            summary=dict(expected=1,hold_joint_f1_50ms=0,silence_false_events=0,per_type={}))])
        text=monitor.render(state)
        self.assertIn('opt 228',text)
        self.assertIn('1.2',text)
        self.assertIn('正例留出 CE',text)
        self.assertIn('音频反事实',text)
        state.ingest_line('fc-two {"phase":"bootstrap"}')
        self.assertNotIn('opt 228',monitor.render(state))

    def test_auto_target_ignores_unrelated_apps(self):
        apps = [
            dict(app_id='old', description='stage0-cuda-training', state='stopped', created_at='1'),
            dict(app_id='new', description='stage0-cuda-training', state='deployed', created_at='2'),
            dict(app_id='other', description='unrelated', state='deployed', created_at='3'),
        ]
        with patch.object(monitor, 'modal_json', return_value=apps):
            self.assertEqual(monitor.active_app(None), ('new', 'deployed'))
            self.assertEqual(monitor.active_app('old'), ('old', 'stopped'))

    def test_redeploy_clears_progress_and_stale_source(self):
        state = monitor.State()
        state.select_app('old', 'deployed')
        state.apply(dict(phase='train', step=354, elapsed=350, loss=1.7))
        state.checkpoints = ['old.pt']
        state.select_app('new', 'deployed')
        self.assertEqual(state.latest(), {})
        self.assertEqual(state.checkpoints, [])
        self.assertIsNone(state.started_wall)
        self.assertIn('waiting', monitor.render(state))
        state.apply(dict(phase='train', step=1, elapsed=1, loss=5))
        self.assertEqual(state.latest()['step'], 1)

    def test_discovery_failure_is_not_reported_as_idle(self):
        with patch.object(monitor, 'modal_json', return_value=None):
            self.assertEqual(monitor.active_app(None), (None, 'discovery unavailable'))

    def test_zero_tasks_and_exception_replace_live_eta(self):
        apps = [dict(app_id='new', description='stage0-cuda-training',
                     state='deployed', tasks='0', created_at='2')]
        with patch.object(monitor, 'modal_json', return_value=apps):
            app_id, status = monitor.active_app(None)
        state = monitor.State()
        state.select_app(app_id, status)
        state.total_steps = 300000
        state.apply(dict(phase='train', step=353, elapsed=300))
        state.apply(dict(phase='train', step=354, elapsed=301))
        state.ingest_line('fc-test ValueError: Illegal mania:45 in mania')
        text = monitor.render(state)
        self.assertIn('tasks=0', text)
        self.assertIn('nothing is training', text)
        self.assertIn('ValueError: Illegal mania:45 in mania', text)
        self.assertNotIn('eta       ', text)

    def test_failed_phase_stops_live_eta_before_container_disappears(self):
        state = monitor.State()
        state.select_app('new', 'deployed; tasks=1')
        state.total_steps = 300000
        state.apply(dict(phase='train', step=353, elapsed=300))
        state.apply(dict(phase='train', step=354, elapsed=301))
        state.ingest_line('{"phase":"failed","step":354,"elapsed":301.68}')
        text = monitor.render(state)
        self.assertIn('training reported failure', text)
        self.assertNotIn('eta       ', text)

    def test_new_call_clears_previous_failure_during_bootstrap(self):
        state = monitor.State()
        state.select_app('app', 'deployed; tasks=1')
        state.ingest_line('fc-old {"phase":"train","step":354,"elapsed":301}')
        state.ingest_line('fc-old {"phase":"failed","step":354}')
        state.ingest_line('fc-new {"phase":"bootstrap"}')
        text = monitor.render(state)
        self.assertNotIn('training reported failure', text)
        self.assertNotIn('354', text)
        self.assertIn('fc-new', text)


if __name__ == '__main__':
    unittest.main()
