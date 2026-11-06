"""Finish the two previously omitted research pairs, then frozen finalists."""
import json
from pathlib import Path
import subprocess
import time

from scripts.run_local_portfolio_scale import write


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/local-portfolio-v1/research-then-finalists-v1.json'


def main():
    if REPORT.exists():
        raise FileExistsError('No automatic research retry')
    report = dict(status='waiting_for_remaining_pool', stages=[], concurrency=1,
                  cloud_compute_calls=0, automatic_retries=0)
    write(REPORT, report)
    start = time.monotonic()
    try:
        while True:
            value = json.loads((ROOT / 'reports/local-portfolio-v1/remaining-pool-v4.json').read_text())
            if value['status'] == 'completed':
                break
            if value['status'] == 'stopped':
                raise RuntimeError('Predecessor stopped; explicit diagnosis required')
            if time.monotonic() - start > 58000:
                raise TimeoutError('Research predecessor wait bound')
            time.sleep(5)
        for kind in ('depth', 'ssl'):
            report['status'] = 'running_' + kind
            write(REPORT, report)
            tick = time.monotonic()
            with (ROOT / f'artifacts/local-music-{kind}-pair-v1.log').open('x') as out:
                child = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', '-m',
                    'scripts.run_local_music_research', '--kind', kind], cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
                report['pid'] = child.pid
                write(REPORT, report)
                try:
                    code = child.wait(timeout=14800)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait()
                    raise
            value = json.loads((ROOT / f'reports/local_music_{kind}_pair_v1.json').read_text())
            report['stages'].append(dict(id=kind, status=value['status'], exit_code=code,
                                        wall_s=time.monotonic() - tick))
            write(REPORT, report)
            if code or value['status'] != 'completed':
                raise RuntimeError('Explicit research diagnosis required: ' + kind)
        report.update(status='completed_research_pairs', wall_s=time.monotonic() - start)
        write(REPORT, report)
        from scripts import run_frozen_finalist_protocol as final
        final.REPORT = ROOT / 'reports/local-portfolio-v1/frozen-finalist-protocol-v2.json'
        final.main()
        report.update(status='completed', wall_s=time.monotonic() - start)
        write(REPORT, report)
    except BaseException as exc:
        report.update(status='stopped', error=f'{type(exc).__name__}: {exc}', wall_s=time.monotonic() - start)
        write(REPORT, report)
        raise


if __name__ == '__main__':
    main()
