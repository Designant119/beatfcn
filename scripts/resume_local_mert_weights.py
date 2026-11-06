"""Resume one terminal, receipted public checkpoint download without restarting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.acquire_local_mert_weights import ROOT, save


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    plan_path = parser.parse_args().plan
    plan = json.loads(plan_path.read_text())
    output = ROOT / plan['report']
    if output.exists():
        raise FileExistsError('Inspect existing resume attempt, do not duplicate')
    for name, expected in plan['source_sha256'].items():
        assert digest(ROOT / name) == expected, name
    prior = json.loads((ROOT / plan['previous_report']).read_text())
    assert prior['execution_complete'] and not prior['complete']
    target = ROOT / plan['destination']
    part = target.with_suffix('.safetensors.part')
    assert not target.exists() and part.stat().st_size == plan['prefix_bytes'] == prior['received_bytes']
    assert digest(part) == plan['prefix_sha256']
    assert 0 < plan['prefix_bytes'] < plan['bytes']
    command = ['curl', '--location', '--fail', '--silent', '--show-error', '--continue-at', '-',
        '--connect-timeout', '15', '--max-time', str(plan['timeout_s']), '--max-filesize', str(plan['bytes']),
        '--output', str(part), '--write-out', '%{http_code}', plan['url']]
    report = dict(id=plan['id'], complete=False, execution_complete=False, pid=os.getpid(),
        command=command, destination=plan['destination'], previous_report=plan['previous_report'],
        prefix_bytes=plan['prefix_bytes'], prefix_sha256=plan['prefix_sha256'],
        received_bytes=plan['prefix_bytes'], expected_bytes=plan['bytes'],
        expected_sha256=plan['sha256'], new_cloud_compute_calls=0, new_cloud_spend_usd=0,
        started_at_unix=time.time(), plan_sha256=digest(plan_path))
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
                report.update(received_bytes=part.stat().st_size, wall_s=time.monotonic()-start,
                              observed_at_unix=time.time())
                save(output, report)
                if report['wall_s'] > plan['timeout_s']+15:
                    child.kill()
                    report['returncode'] = child.wait()
                    report['error'] = 'Parent deadline exceeded'
                    break
    report['received_bytes'] = part.stat().st_size
    report['new_received_bytes'] = report['received_bytes']-plan['prefix_bytes']
    if report['returncode'] == 0 and report['received_bytes'] == plan['bytes']:
        report['actual_sha256'] = digest(part)
        if report['actual_sha256'] == plan['sha256']:
            part.replace(target)
            report['complete'] = True
        else:
            report['error'] = 'Full checkpoint SHA256 mismatch'
    else:
        report.setdefault('error', 'Resume incomplete; keep receipted prefix for a separately bounded continuation')
    report.update(execution_complete=True, wall_s=time.monotonic()-start, ended_at_unix=time.time())
    save(output, report)
    print(json.dumps({k: report[k] for k in ('complete', 'received_bytes', 'new_received_bytes', 'wall_s')}))
    raise SystemExit(0 if report['complete'] else 1)


if __name__ == '__main__':
    main()
