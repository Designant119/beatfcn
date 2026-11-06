"""Watch a specific local probe child with fixed wall-time and RSS limits."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    started = time.time()
    log = ROOT / plan['console']
    output = ROOT / plan['resource_report']
    report = dict(complete=False, started_at_unix=started, maximum_rss_gib=0., samples=0,
        child_script=plan['script'], timeout_s=plan['timeout_s'], rss_limit_gib=plan['maximum_rss_gib'])
    def save():
        output.write_text(json.dumps(report, indent=2)+'\n')
    with log.open('w') as stream:
        child = subprocess.Popen([sys.executable, plan['script'], '--plan', str(args.plan)], cwd=ROOT,
            stdout=stream, stderr=subprocess.STDOUT)
        report['pid'] = child.pid
        save()
        while child.poll() is None:
            # The child does not spawn workers. ps reports RSS in KiB on macOS.
            snapshot = subprocess.run(['ps','-p',str(child.pid),'-o','rss='],capture_output=True,text=True)
            if snapshot.returncode == 0 and snapshot.stdout.strip():
                rss = int(snapshot.stdout.strip()) / 2**20
                report['maximum_rss_gib'] = max(report['maximum_rss_gib'], rss)
                report['samples'] += 1
            reason = ('wall_time_limit' if time.time()-started > plan['timeout_s'] else
                      'rss_limit' if report['maximum_rss_gib'] > plan['maximum_rss_gib'] else None)
            if reason:
                report['terminated_by'] = reason
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
                break
            save()
            time.sleep(.5)
        report.update(exit_code=child.wait(), ended_at_unix=time.time(), complete=True)
        save()
    print(json.dumps(report))
    if report['exit_code'] != 0 or report.get('terminated_by'):
        raise SystemExit('bounded local probe did not pass')


if __name__ == '__main__':
    main()
