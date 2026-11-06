import unittest
from unittest.mock import Mock,patch
from beatfcn.train.paid_call import await_bounded_call

class PaidCallTests(unittest.TestCase):
    def test_deadline_is_forwarded_without_stopping_success(self):
        call=Mock();call.get.return_value={'ok':True}
        with patch('beatfcn.train.paid_call.subprocess.run') as run:
            self.assertEqual(await_bounded_call(call,timeout_s=60,app_id='ap-owned',modal_command=['modal']),{'ok':True})
            call.get.assert_called_once_with(timeout=60);run.assert_not_called()

    def test_failed_or_timed_out_call_stops_exact_owned_app(self):
        call=Mock();call.get.side_effect=TimeoutError('deadline')
        with patch('beatfcn.train.paid_call.subprocess.run',return_value=Mock(returncode=0)) as run:
            with self.assertRaises(TimeoutError):await_bounded_call(call,timeout_s=60,app_id='ap-owned',modal_command=['modal'])
            self.assertEqual(run.call_args.args[0],['modal','app','stop','--yes','ap-owned'])

    def test_stop_failure_is_not_reported_as_success(self):
        call=Mock();call.get.side_effect=RuntimeError('container import failed')
        with patch('beatfcn.train.paid_call.subprocess.run',return_value=Mock(returncode=1)):
            with self.assertRaisesRegex(RuntimeError,'stop not confirmed'):
                await_bounded_call(call,timeout_s=60,app_id='ap-owned',modal_command=['modal'])

if __name__=='__main__':unittest.main()
