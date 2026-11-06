"""Matched old-routing control after the previously frozen serial tail."""
import json
from pathlib import Path
import subprocess
import time

from scripts.run_local_portfolio_scale import write

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/local-portfolio-v1/concat-after-attention-v1.json'


def main():
    if REPORT.exists():raise FileExistsError('No duplicate routing control')
    predecessor=ROOT/'reports/local-portfolio-v1/attention-after-pool-v1.json'
    report=dict(status='waiting_for_attention_tail',predecessor=str(predecessor.relative_to(ROOT)),cloud_compute_calls=0,automatic_retries=0)
    write(REPORT,report);started=time.monotonic()
    try:
        while True:
            state=json.loads(predecessor.read_text())
            if state['status']=='completed':break
            if state['status']=='stopped':raise RuntimeError('Predecessor needs explicit review')
            if time.monotonic()-started>48000:raise TimeoutError('Wait bound')
            time.sleep(5)
        with (ROOT/'artifacts/local-concat-control-quality-v1.log').open('x') as out:
            child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','scripts.run_concat_control_quality'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
            report.update(status='running_concat_control',pid=child.pid);write(REPORT,report)
            try:code=child.wait(timeout=9000)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
        result=json.loads((ROOT/'reports/local_concat_control_quality_v1.json').read_text())
        report.update(status='completed' if code==0 and result['status']=='completed' else 'stopped',exit_code=code,
            experiment_status=result['status'],wall_s=time.monotonic()-started);write(REPORT,report)
    except BaseException as exc:
        report.update(status='stopped',error=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-started);write(REPORT,report);raise


if __name__=='__main__':main()
