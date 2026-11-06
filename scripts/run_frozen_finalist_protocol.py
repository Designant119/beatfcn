"""Bounded continuation of already-frozen selection/replication/confirmation.

No new model suggestions, adaptive finalists, repeat seeds or cloud jobs.
An unsupported adapter or implementation error stops for explicit diagnosis.
"""
import json
from pathlib import Path
import subprocess
import time

from scripts.run_local_portfolio_scale import write


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/local-portfolio-v1/frozen-finalist-protocol-v1.json'


def main():
    if REPORT.exists():
        raise FileExistsError('No retry of the frozen finalist protocol')
    predecessor = ROOT / 'reports/local-portfolio-v1/remaining-pool-v4.json'
    report = dict(status='waiting_for_remaining_pool', stages=[], concurrency=1,
        policy='bench/rhythm-v2/local-portfolio-final-selection-v1.json',
        cloud_compute_calls=0, automatic_retries=0, confirmation_scored=False)
    write(REPORT, report)
    started = time.monotonic()

    def run(name, module, args=(), wall=10000):
        report.update(status='running_' + name)
        write(REPORT, report)
        tick = time.monotonic()
        with (ROOT / f'artifacts/{name}.log').open('x') as out:
            child = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', '-m', module, *args],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
            report['pid'] = child.pid
            write(REPORT, report)
            try:
                code = child.wait(timeout=wall)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
        report['stages'].append(dict(id=name, exit_code=code, wall_s=time.monotonic() - tick))
        write(REPORT, report)
        if code:
            raise RuntimeError('Explicit diagnosis required: ' + name)

    try:
        while True:
            state = json.loads(predecessor.read_text())
            if state['status'] == 'completed':
                break
            if state['status'] == 'stopped':
                raise RuntimeError('Predecessor stopped; no automatic restart')
            if time.monotonic() - started > 58000:
                raise TimeoutError('Serial predecessor wait bound')
            time.sleep(5)
        run('local-finalist-selection-v1', 'scripts.portfolio_terminal_catalog', ('--select',), wall=120)
        selection = json.loads((ROOT / 'reports/local_portfolio_finalist_selection_v1.json').read_text())
        short = False
        for rank, entry in enumerate(selection['finalist_identities'], 1):
            if entry['factory'] not in ('mamba', 'direct_song', 'canvas', 'mel', 'access', 'depth_budget', 'ssl_context'):
                raise ValueError('Selected method needs its own faithful adapter: ' + entry['factory'])
            run(f'local-finalist-r{rank}-seed2-v1', 'scripts.replicate_canvas_finalist', ('--rank', str(rank)))
            value = json.loads((ROOT / f'reports/local_finalist_r{rank}_seed2_v1.json').read_text())
            if value['status'] not in ('completed', 'health_failed'):
                raise ValueError('No hidden recovery of finalist')
            short |= entry['arm'] == 'short32' and value.get('reproduced', False)
        if short:
            run('local-mert-confirmation-context-cache-v1', 'scripts.cache_confirmation_short_mert', wall=1500)
        run('local-portfolio-confirmation-v1', 'scripts.confirm_canvas_finalists', wall=4000)
        value = json.loads((ROOT / 'reports/local_portfolio_confirmation_v1.json').read_text())
        if value['status'] not in ('completed', 'not_opened_no_reproduced_finalist'):
            raise ValueError('Confirmation incomplete')
        report.update(status='completed', confirmation_scored=value['confirmation_scored'], wall_s=time.monotonic() - started)
        write(REPORT, report)
    except BaseException as exc:
        report.update(status='stopped', error=f'{type(exc).__name__}: {exc}', wall_s=time.monotonic() - started)
        write(REPORT, report)
        raise


if __name__ == '__main__':
    main()
