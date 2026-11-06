"""Bounded D94 download-completion -> admission -> real-health execution.

Does not start/retry downloads or quality training. Stops if the registered
transfer fails, any frozen source changes, or an admission/health gate fails.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def run(plan_path):
    plan = json.loads(plan_path.read_text())
    output = ROOT / plan['output']
    report = dict(id=plan['id'], pid=os.getpid(), plan_sha256=sha(plan_path),
        complete=False, passed=False, started_at_unix=time.time(),
        phase='waiting_for_registered_transfer', child_pid=None,
        cloud_compute_calls=0, quality_training_started=False, stages=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')

    def save():
        report['observed_at_unix'] = time.time()
        temp = output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temp.replace(output)

    def verify_sources():
        for path, expected in plan['source_sha256'].items():
            if sha(path) != expected:
                raise ValueError('Frozen chain dependency changed: ' + path)

    child = None
    try:
        verify_sources()
        print(json.dumps(dict(phase=report['phase'], pid=os.getpid(), transfer_pid=plan['transfer_pid'])), flush=True)
        deadline = time.monotonic() + plan['max_transfer_wait_s']
        while True:
            receipt = json.loads((ROOT / plan['transfer_receipt']).read_text())
            if receipt['plan_sha256'] != sha(plan['transfer_plan']):
                raise ValueError('Registered transfer identity changed')
            process = subprocess.run(['ps', '-p', str(plan['transfer_pid']), '-o', 'command='],
                                     capture_output=True, text=True, check=False)
            if process.returncode:
                if not receipt['complete'] or 'ended_at_unix' not in receipt or receipt['failures']:
                    raise RuntimeError('Registered transfer terminated without complete success; no automatic retry')
                report['completed_transfer_receipt_sha256'] = sha(plan['transfer_receipt'])
                break
            if not process.stdout.strip().endswith('scripts.fetch_frozen_volume_files --plan ' + plan['transfer_plan']):
                raise ValueError('Transfer PID belongs to another command')
            if time.monotonic() > deadline:
                raise TimeoutError('Registered local wait limit reached; transfer is not terminated by this chain')
            report['transfer_completed_files'] = len(receipt['files'])
            save()
            time.sleep(10)
        commands = [
            ('feature_admission', ['-m', 'scripts.audit_local_features', '--plan', plan['audit_plan']], 210),
            ('freeze_new_data_probe', ['-m', 'scripts.prepare_local_data_scale_probe'], 180),
            ('real_fixed_batch_probe', ['-m', 'scripts.probe_local_data_scale', '--plan',
                                       'bench/rhythm-v2/d94-new-data-probe-v1.json'], 960),
        ]
        for phase, args, limit in commands:
            verify_sources()
            report['phase'] = phase
            if phase == 'feature_admission' and json.loads((ROOT / plan['audit_report']).read_text())['complete']:
                raise ValueError('Admission already completed outside this chain; inspect rather than repeat')
            child = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', *args], cwd=ROOT)
            report['child_pid'] = child.pid
            stage = dict(phase=phase, pid=child.pid, started_at_unix=time.time(), command=args)
            report['stages'].append(stage)
            save()
            print(json.dumps(dict(phase=phase, child_pid=child.pid)), flush=True)
            try:
                code = child.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait(timeout=10)
                raise TimeoutError('Bounded child limit reached: ' + phase)
            stage.update(exit_code=code, ended_at_unix=time.time())
            report['child_pid'] = None
            child = None
            save()
            if code:
                raise RuntimeError('Local stage failed: ' + phase)
        health = json.loads((ROOT / 'reports/d94_new_data_probe_v1.json').read_text())
        report.update(complete=True, passed=health['complete'] and health['passed'],
            health='reports/d94_new_data_probe_v1.json', health_sha256=sha('reports/d94_new_data_probe_v1.json'),
            phase='health_finished_requires_review', ended_at_unix=time.time())
        save()
        print(json.dumps(dict(phase=report['phase'], passed=report['passed'])), flush=True)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait(timeout=10)
        report.update(phase='stopped', passed=False, error=type(exc).__name__ + ': ' + str(exc),
                      child_pid=None, ended_at_unix=time.time())
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan)
