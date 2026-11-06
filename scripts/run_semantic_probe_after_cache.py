"""Bounded single-owner handoff from the existing D96 cache to the D97 probe."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/d97_probe_dispatch_v1.json'
CHILD_REPORT = ROOT/'reports/d97_semantic_objective_probe_v1.json'
PLAN = ROOT/'bench/rhythm-v2/d97-semantic-objective-probe-v1.json'
CACHE = ROOT/'reports/d96_local_mert_cache_workers_v2.json'


def write(report):
    report['observed_at_unix'] = time.time()
    tmp = REPORT.with_suffix('.tmp')
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(REPORT)


def cache_process_alive(pid):
    if not pid:
        return False
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], text=True, capture_output=True)
    return result.returncode == 0 and any(name in result.stdout for name in
        ('scripts.run_local_mert_cache_workers', 'scripts.cache_local_mert_features'))


def main():
    if CHILD_REPORT.exists():
        raise FileExistsError('Inspect existing D97 probe evidence; do not repeat it')
    plan = json.loads(PLAN.read_text())
    report = dict(id='D97-bounded-probe-dispatch-v1', status='waiting_for_mps',
        pid=os.getpid(), child_pid=None, complete=False, started_at_unix=time.time(),
        plan=str(PLAN.relative_to(ROOT)), plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        probe_report=str(CHILD_REPORT.relative_to(ROOT)),
        new_cloud_compute_calls=0, new_cloud_spend_usd=0, optimizer_updates=0,
        max_wait_s=1200, child_timeout_s=plan['max_wall_s']+30)
    with REPORT.open('x') as stream:
        json.dump(report, stream)
    child = None
    try:
        while True:
            cache = json.loads(CACHE.read_text())
            active = cache_process_alive(cache.get('pid')) or cache_process_alive(cache.get('current_worker_pid'))
            report.update(cache_committed_songs=cache.get('committed_songs'), cache_active=active)
            write(report)
            if cache.get('execution_complete') and not active:
                report['cache_terminal_success'] = bool(cache.get('complete'))
                break
            if not active and not cache.get('execution_complete'):
                raise RuntimeError('Cache processes disappeared without a terminal record; inspect before dispatch')
            if time.time()-report['started_at_unix'] >= report['max_wait_s']:
                raise TimeoutError('Handoff wait limit; no training was started')
            time.sleep(15)
        if CHILD_REPORT.exists():
            raise FileExistsError('Another task already produced D97 probe evidence')
        log = ROOT/'reports/d97_semantic_objective_probe_v1.log'
        with log.open('x') as stream:
            child = subprocess.Popen([sys.executable, '-m', 'scripts.probe_local_semantic_objective',
                                      '--plan', str(PLAN)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            report.update(status='running_probe', child_pid=child.pid, child_started_at_unix=time.time())
            write(report)
            try:
                code = child.wait(timeout=report['child_timeout_s'])
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL);child.wait()
                raise TimeoutError('Actual D97 probe exceeded its frozen wall limit')
        result = json.loads(CHILD_REPORT.read_text()) if CHILD_REPORT.exists() else {}
        passed = code == 0 and result.get('complete') and result.get('passed')
        report.update(status='probe_passed' if passed else 'probe_failed', complete=True,
                      child_exit_code=code, passed=bool(passed),
                      optimizer_updates=result.get('optimizer_updates', 0),
                      quality_training_dispatched=False)
        write(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return 0 if passed else 1
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL);child.wait()
        report.update(status='failed', complete=True, error=f'{type(exc).__name__}: {exc}',
                      child_exit_code=child.returncode if child else None)
        write(report)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
