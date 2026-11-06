"""One bounded serial follow-up: frozen readout audit and matched access arms."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/local-portfolio-v1/followup-queue-v1.json'


def main():
    if REPORT.exists():
        raise FileExistsError('No automatic rerun of the follow-up queue')
    old_queue = json.loads((ROOT / 'reports/local-portfolio-v1/quality-queue-v1.json').read_text())
    package = json.loads((ROOT / 'reports/local_portfolio_package_quality_v1.json').read_text())
    if old_queue['status'] != 'completed' or package['status'] != 'completed' or package['updates'] != 2000:
        raise ValueError('Prior MPS queue must have completed before launching this queue')
    audit = json.loads((ROOT / 'reports/local_access_input_audit_v1.json').read_text())
    from scripts.run_local_portfolio_mamba_health import sha
    if not audit['passed'] or audit['plan_sha256'] != sha(ROOT / 'bench/rhythm-v2/local-access-comparison-v1.json'):
        raise ValueError('Require matching actual-input audit before health')
    data = dict(id='local-portfolio-followup-queue-v1', status='running', concurrency=1,
        automatic_retries=0, cloud_compute_calls=0,
        started_at=datetime.now(timezone.utc).isoformat(), stages=[])
    child = None

    def save():
        temporary = REPORT.with_suffix('.partial')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(REPORT)

    def stop(*_):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
        raise KeyboardInterrupt('External queue stop')

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    save()
    jobs = [
        ('package-readout', 'scripts.probe_local_package_readout', 'run', 1350),
        ('access-health', 'scripts.run_local_access_comparison', 'health', 3300),
        ('access-quality', 'scripts.run_local_access_comparison', 'quality', 7500),
    ]
    try:
        for name, module, stage_name, wall in jobs:
            if name == 'access-quality':
                health = json.loads((ROOT / 'reports/local_access_health_v1.json').read_text())
                if not health.get('health_passed', False):
                    data.update(status='stopped_at_health_gate', failure='Access health did not pass; no quality training.')
                    save()
                    return
            log = ROOT / f'artifacts/local-portfolio-v1-{name}.log'
            entry = dict(stage=name, status='starting', log=str(log.relative_to(ROOT)), outer_wall_s=wall)
            data['stages'].append(entry)
            save()
            with log.open('x') as handle:
                child = subprocess.Popen([sys.executable, '-u', '-m', module, stage_name], cwd=ROOT,
                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                entry.update(status='running', pid=child.pid)
                save()
                print(json.dumps(entry), flush=True)
                try:
                    code = child.wait(timeout=wall)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGINT)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    entry.update(status='outer_deadline_stopped', exit_code=child.returncode)
                    save()
                    raise
            entry.update(status='completed' if code == 0 else 'failed', exit_code=code,
                         finished_at=datetime.now(timezone.utc).isoformat())
            save()
            print(json.dumps(entry), flush=True)
            child = None
            if code != 0:
                raise RuntimeError(f'{name} exited {code}; no retry or silent continuation')
        data['status'] = 'completed'
        save()
    except BaseException as exc:
        data.update(status='stopped', failure=f'{type(exc).__name__}: {exc}')
        save()
        raise


if __name__ == '__main__':
    main()
