"""One bounded download of the pinned public MERT checkpoint, with a receipt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    output = ROOT / plan['report']
    if output.exists():
        raise FileExistsError('Inspect the exact existing attempt; do not duplicate it')
    gate = json.loads((ROOT / plan['operator_gate']).read_text())
    assert gate['complete'] and gate['execution_complete']
    for name, digest in plan['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    target = ROOT / plan['destination']
    part = target.with_suffix('.safetensors.part')
    if target.exists() or part.exists():
        raise FileExistsError('First attempt cannot overwrite or resume unreceipted content')
    target.parent.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(target.parent).free >= plan['minimum_free_bytes']
    command = ['curl', '--location', '--fail', '--silent', '--show-error',
               '--connect-timeout', '15', '--max-time', str(plan['timeout_s']),
               '--max-filesize', str(plan['bytes']), '--output', str(part), plan['url']]
    report = dict(id=plan['id'], complete=False, execution_complete=False, pid=os.getpid(),
        expected_bytes=plan['bytes'], received_bytes=0, expected_sha256=plan['sha256'],
        destination=plan['destination'], command=command, new_cloud_compute_calls=0,
        new_cloud_spend_usd=0, started_at_unix=time.time(),
        plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest())
    start = time.monotonic()
    save(output, report)
    with (ROOT / plan['log']).open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        report['child_pid'] = child.pid
        save(output, report)
        while True:
            try:
                report['returncode'] = child.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                report.update(received_bytes=part.stat().st_size if part.exists() else 0,
                              wall_s=time.monotonic()-start, observed_at_unix=time.time())
                save(output, report)
                if report['wall_s'] > plan['timeout_s'] + 15:
                    child.kill()
                    report['returncode'] = child.wait()
                    report['error'] = 'Parent deadline exceeded'
                    break
    report['received_bytes'] = part.stat().st_size if part.exists() else 0
    if report['returncode'] == 0 and report['received_bytes'] == plan['bytes']:
        digest = hashlib.sha256()
        with part.open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                digest.update(block)
        report['actual_sha256'] = digest.hexdigest()
        if digest.hexdigest() == plan['sha256']:
            part.replace(target)
            report['complete'] = True
        else:
            report['error'] = 'Downloaded checkpoint SHA256 mismatch'
    else:
        report.setdefault('error', 'Transfer incomplete; partial bytes retained for a separately bounded resume')
    report.update(execution_complete=True, wall_s=time.monotonic()-start, ended_at_unix=time.time())
    save(output, report)
    print(json.dumps({k: report[k] for k in ('complete', 'execution_complete', 'received_bytes', 'wall_s')}))
    raise SystemExit(0 if report['complete'] else 1)


if __name__ == '__main__':
    main()
