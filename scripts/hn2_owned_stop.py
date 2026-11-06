"""Bounded, noninteractive stop of one actual owned Modal app."""
from pathlib import Path
import re
import subprocess


def stop_owned_app(root, app_id):
    if not re.fullmatch(r'ap-[A-Za-z0-9]{22}', app_id):
        raise ValueError('Actual owned Modal app ID required')
    command = [str(Path(root)/'.venv/bin/modal'), 'app', 'stop', '--yes', app_id]
    try:
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return dict(app_id=app_id, command=command, timed_out=True, returncode=None)
    except OSError as exc:
        return dict(app_id=app_id, command=command, timed_out=False, returncode=None,
                    error=type(exc).__name__)
    # This is command evidence only. The supervisor always independently
    # verifies the same app's stopped/tasks0 state before closing or cleaning.
    return dict(app_id=app_id, command=command, timed_out=False, returncode=result.returncode,
                stdout=result.stdout[-2000:], stderr=result.stderr[-2000:])
