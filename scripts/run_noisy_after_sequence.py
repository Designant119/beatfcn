"""Execute the complete noisy-proposal candidate after existing local jobs."""
import json
from pathlib import Path
import subprocess
import time

from scripts.run_local_portfolio_scale import write


ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'reports/local-portfolio-v1/noisy-refiner-queue-v1.json'


def main():
    if REPORT.exists():raise FileExistsError('Preserve noisy-refiner queue')
    report=dict(status='waiting_for_prior_sequence',cloud_compute_calls=0);write(REPORT,report);started=time.monotonic()
    try:
        while True:
            prior=json.loads((ROOT/'reports/local-portfolio-v1/repaired-sequence-v2.json').read_text())
            cache=json.loads((ROOT/'reports/local_noisy_proposal_cache_v1.json').read_text())
            if prior['status']=='completed' and cache['status']=='completed':break
            if prior['status']=='stopped' or cache['status']=='stopped':raise RuntimeError('Prior task needs explicit recovery')
            if time.monotonic()-started>30000:raise TimeoutError('Prior sequence wall')
            time.sleep(10)
        report['status']='running';write(REPORT,report)
        with (ROOT/'artifacts/local-noisy-proposal-refiner-v1.log').open('ab') as out:
            child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','scripts.run_noisy_proposal_refiner'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
            report['pid']=child.pid;write(REPORT,report);code=child.wait(timeout=5000)
        result=json.loads((ROOT/'reports/local_noisy_proposal_refiner_v1.json').read_text())
        report.update(status='completed' if code==0 and result['status'] in ('completed','health_failed') else 'stopped',exit_code=code,experiment_status=result['status'],wall_s=time.monotonic()-started)
        write(REPORT,report)
    except BaseException as exc:
        report.update(status='stopped',error=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-started);write(REPORT,report);raise


if __name__=='__main__':main()
