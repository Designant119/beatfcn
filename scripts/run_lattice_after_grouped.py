"""Serial follow-up for the complete lattice; no retry, no cloud or council."""
import json
from pathlib import Path
import subprocess
import time

from scripts.run_local_portfolio_scale import write


ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/local-portfolio-v1/lattice-queue-v1.json'


def main():
    if REPORT.exists():raise FileExistsError('Preserve lattice queue')
    report=dict(status='waiting_for_grouped_queue',cloud_compute_calls=0)
    write(REPORT,report);started=time.monotonic()
    try:
        while True:
            path=ROOT/'reports/local-portfolio-v1/final-candidates-queue-v1.json'
            state=json.loads(path.read_text())
            if state['status']=='completed':break
            if state['status']=='stopped':raise RuntimeError('Prior queue incomplete; do not silently continue')
            if time.monotonic()-started>20000:raise TimeoutError('Prior queues total wall')
            time.sleep(10)
        report['status']='running';write(REPORT,report)
        with (ROOT/'artifacts/local-lattice-complete-v1.log').open('ab') as out:
            child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','scripts.run_local_lattice_experiment'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
            report['pid']=child.pid;write(REPORT,report);code=child.wait(timeout=5000)
        path=ROOT/'reports/local_lattice_complete_v1.json'
        result=json.loads(path.read_text()) if path.exists() else {}
        report.update(status='completed' if code==0 and result.get('status') in ('completed','health_failed') else 'stopped',
            exit_code=code,experiment_status=result.get('status','no_report'),wall_s=time.monotonic()-started)
        write(REPORT,report)
    except BaseException as exc:
        report.update(status='stopped',error=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-started);write(REPORT,report);raise


if __name__=='__main__':main()
