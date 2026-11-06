"""Bounded waiting for an existing call, never a remote invocation retry."""
import time


def local_poll_timeout(exc, poll_code):
    """Only the pinned SDK's direct empty-output raise is retryable.

    Remote built-in TimeoutError is also possible. Its final traceback frame
    belongs to result processing or remote code, rather than poll_function.
    A changed SDK path fails closed instead of suppressing unknown errors.
    """
    if type(exc) is not TimeoutError or poll_code is None:
        return False
    tail = exc.__traceback__
    if tail is None:
        return False
    while tail.tb_next is not None:
        tail = tail.tb_next
    return tail.tb_frame.f_code is poll_code


def wait_existing_call(call, *, check, soft_deadline, poll_code, now=time.time):
    while True:
        check()
        remaining = soft_deadline - now()
        if remaining <= 0:
            raise TimeoutError('Existing-call wait reached lifecycle soft deadline')
        try:
            return call.get(timeout=min(10, remaining))
        except Exception as exc:
            if not local_poll_timeout(exc, poll_code):
                raise
