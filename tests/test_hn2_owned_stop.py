import unittest

try:
    import modal
except ModuleNotFoundError:
    raise unittest.SkipTest('modal is not installed in this environment')

from pathlib import Path
from types import SimpleNamespace
import subprocess
import threading
import time
import unittest
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner
from modal.cli.app import app_cli
from modal_proto import api_pb2

from beatfcn.train.hn2_stop_supervisor import OwnedStopSupervisor
from scripts.hn2_owned_stop import stop_owned_app


APP = 'ap-abcdefghijklmnopqrstuv'


class StopTests(unittest.TestCase):
    def actual_cli(self, yes):
        rpc = AsyncMock()
        tasks = AsyncMock(return_value=SimpleNamespace(tasks=[]))
        client = SimpleNamespace(_stub=SimpleNamespace(AppStop=rpc, TaskList=tasks))
        lifecycle = api_pb2.AppLifecycle(app_state=api_pb2.APP_STATE_EPHEMERAL)
        with patch('modal.cli.app.ensure_env', return_value='main'), \
             patch('modal.cli.app._Client.from_env', AsyncMock(return_value=client)), \
             patch('modal.cli.app.resolve_app_identifier', AsyncMock(return_value=(APP, None, lifecycle))):
            result = CliRunner().invoke(app_cli, ['stop', *(['--yes'] if yes else []), APP])
        return result, rpc, tasks

    def test_actual_sdk_yes_non_tty_sends_exact_appstop_without_confirmation(self):
        result, rpc, tasks = self.actual_cli(True)
        self.assertEqual(result.exit_code, 0, result.output)
        rpc.assert_awaited_once()
        self.assertEqual(rpc.await_args.args[0].app_id, APP)
        self.assertEqual(rpc.await_args.args[0].source, api_pb2.APP_STOP_SOURCE_CLI)
        tasks.assert_not_awaited()

    def test_actual_sdk_without_yes_non_tty_does_not_send_appstop(self):
        result, rpc, tasks = self.actual_cli(False)
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('no interactive terminal', result.output)
        rpc.assert_not_awaited()
        tasks.assert_awaited_once()

    def test_launcher_helper_argv_is_noninteractive_bounded_and_records_failure(self):
        response = SimpleNamespace(returncode=1, stdout='already stopped', stderr='')
        with patch('scripts.hn2_owned_stop.subprocess.run', return_value=response) as run:
            value = stop_owned_app(Path('/task'), APP)
        self.assertEqual(run.call_args.args[0], ['/task/.venv/bin/modal', 'app', 'stop', '--yes', APP])
        self.assertEqual(run.call_args.kwargs['timeout'], 10)
        self.assertEqual(value['returncode'], 1)
        self.assertNotIn('stop_verified', value)

    def test_command_timeout_is_recorded_without_claiming_stop(self):
        with patch('scripts.hn2_owned_stop.subprocess.run', side_effect=subprocess.TimeoutExpired('modal', 10)):
            value = stop_owned_app('/task', APP)
        self.assertTrue(value['timed_out'])
        self.assertNotIn('stop_verified', value)

    def test_failed_stop_still_verifies_but_cannot_close_before_real_proof(self):
        called = threading.Event(); allow_proof = threading.Event(); cleanup = threading.Event()
        def command(app_id):
            called.set()
            return dict(returncode=1)
        ctl = OwnedStopSupervisor(time.time()+.04, .02, app_id=lambda: APP,
            cancel=lambda: None, stop_app=command,
            verify=lambda app_id: dict(app_id=app_id,state='stopped',tasks=0) if allow_proof.is_set() else None,
            record=lambda app_id, proof: dict(job=dict(stop_required=False)),
            cleanup=lambda: (cleanup.set() or dict(passed=True)), publish=lambda value: None,
            interval=.002, record_interval=.005, stop_interval=.005).start()
        try:
            self.assertTrue(called.wait(.5))
            self.assertFalse(ctl.state['stop_verified'])
            self.assertFalse(ctl.state['accounting_closed'])
            self.assertFalse(cleanup.is_set())
            allow_proof.set()
            self.assertTrue(ctl.finished.wait(1))
            self.assertTrue(cleanup.is_set())
        finally:
            allow_proof.set();ctl.request_stop('cleanup');ctl.finished.wait(1);ctl.clock.join(1)

    def test_supervisor_sends_actual_sdk_appstop_while_main_and_accounting_wait(self):
        sent = threading.Event(); release = threading.Event(); values = []
        def run(command, **kwargs):
            self.assertEqual(command[-2:], ['--yes', APP])
            result, rpc, tasks = self.actual_cli(True)
            values.append((result, rpc))
            sent.set()
            return SimpleNamespace(returncode=result.exit_code, stdout=result.output, stderr='')
        def record(app_id, proof):
            release.wait()
            return dict(job=dict(stop_required=False))
        with patch('scripts.hn2_owned_stop.subprocess.run', side_effect=run):
            ctl = OwnedStopSupervisor(time.time()+.04, .02, app_id=lambda: APP,
                cancel=lambda: None, stop_app=lambda app_id: stop_owned_app('/task', app_id),
                verify=lambda app_id: dict(app_id=app_id,state='stopped',tasks=0) if sent.is_set() else None,
                record=record, cleanup=lambda: dict(passed=True), publish=lambda value: None,
                interval=.002, record_interval=.005, stop_interval=.1).start()
            try:
                self.assertTrue(sent.wait(1), 'Active stop must not need main/accounting progress')
                self.assertFalse(release.is_set())
                result, rpc = values[0]
                self.assertEqual(result.exit_code, 0, result.output)
                rpc.assert_awaited_once()
            finally:
                release.set();ctl.request_stop('cleanup')
                self.assertTrue(ctl.finished.wait(1));ctl.clock.join(1)


if __name__ == '__main__':
    unittest.main()
