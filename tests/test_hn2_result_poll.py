"""Actual SDK timeout construction, without networking or model calls."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from modal._functions import _Invocation
from modal.exception import FunctionTimeoutError, TimeoutError as ModalTimeoutError
from scripts.hn2_result_poll import wait_existing_call


POLL_CODE = _Invocation.poll_function.__code__


class PendingCall:
    def __init__(self, pending=2):
        self.calls = 0
        self.pending = pending

    async def pop_function_call_outputs(self, **kwargs):
        return SimpleNamespace(outputs=[], num_unfinished_inputs=1)

    def get(self, timeout):
        self.calls += 1
        if self.calls <= self.pending:
            return asyncio.run(_Invocation.poll_function(self, timeout))
        return 'existing-call-result'


class PollTests(unittest.TestCase):
    def wait(self, call, check=lambda: None, deadline=100):
        return wait_existing_call(call, check=check, soft_deadline=deadline,
                                  poll_code=POLL_CODE, now=lambda: 0)

    def test_actual_sdk_two_builtin_poll_timeouts_then_same_call_result(self):
        call = PendingCall()
        self.assertEqual(self.wait(call), 'existing-call-result')
        self.assertEqual(call.calls, 3)

    def test_remote_builtin_timeout_from_actual_sdk_result_processing_is_terminal(self):
        call = PendingCall(1)
        call.pop_function_call_outputs = AsyncMock(return_value=SimpleNamespace(
            outputs=[SimpleNamespace(result=None, data_format=0)], num_unfinished_inputs=0))
        call.stub = None
        call.client = None
        with patch('modal._functions._process_result', AsyncMock(side_effect=TimeoutError('remote'))):
            with self.assertRaisesRegex(TimeoutError, 'remote'):
                self.wait(call)
        self.assertEqual(call.calls, 1)

    def test_remote_function_execution_timeout_is_terminal(self):
        call = SimpleNamespace(get=lambda **kw: (_ for _ in ()).throw(FunctionTimeoutError('execution')))
        with self.assertRaises(FunctionTimeoutError):
            self.wait(call)

    def test_legacy_modal_timeout_is_not_assumed_to_be_local_poll(self):
        call = SimpleNamespace(get=lambda **kw: (_ for _ in ()).throw(ModalTimeoutError('unknown')))
        with self.assertRaises(ModalTimeoutError):
            self.wait(call)

    def test_supervisor_failure_escapes_before_another_get(self):
        call = PendingCall(20)
        def check():
            if call.calls == 1:
                raise TimeoutError('stop requested')
        with self.assertRaisesRegex(TimeoutError, 'stop requested'):
            self.wait(call, check)
        self.assertEqual(call.calls, 1)

    def test_local_deadline_prevents_get(self):
        call = PendingCall()
        with self.assertRaisesRegex(TimeoutError, 'soft deadline'):
            self.wait(call, deadline=0)
        self.assertEqual(call.calls, 0)

    def test_wrong_sdk_code_identity_fails_closed(self):
        call = PendingCall()
        with self.assertRaises(TimeoutError):
            wait_existing_call(call, check=lambda: None, soft_deadline=100,
                               poll_code=None, now=lambda: 0)
        self.assertEqual(call.calls, 1)


if __name__ == '__main__':
    unittest.main()
