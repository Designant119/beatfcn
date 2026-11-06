"""Bound the caller wait too: container import failures precede function timeout."""
import subprocess


def await_bounded_call(call,*,timeout_s,app_id,modal_command):
    if timeout_s<=0 or not app_id.startswith('ap-'):
        raise ValueError('Positive wall deadline and exact owned app id required')
    try:
        return call.get(timeout=timeout_s)
    except BaseException as original:
        # Stop only this dispatch's app. A timeout does not cancel the remote call.
        try:
            stopped=subprocess.run([*modal_command,'app','stop','--yes',app_id],
                capture_output=True,text=True,timeout=30)
        except Exception as stop_error:
            raise RuntimeError(f'Paid call failed; app stop not confirmed for {app_id}: {stop_error}') from original
        if stopped.returncode:
            raise RuntimeError(f'Paid call failed; app stop not confirmed for {app_id}: exit {stopped.returncode}') from original
        raise
