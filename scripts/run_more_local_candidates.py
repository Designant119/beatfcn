"""After the lattice, execute complete continuous flow and SSD/GRU serially."""
import json
from pathlib import Path
import subprocess
import time

from scripts.run_local_portfolio_scale import write


ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'reports/local-portfolio-v1/more-candidates-queue-v1.json'


def main():
    if REPORT.exists():raise FileExistsError('Preserve extension queue')
    report=dict(status='waiting_for_lattice',stages=[],cloud_compute_calls=0);write(REPORT,report);started=time.monotonic()
    try:
        while True:
            state=json.loads((ROOT/'reports/local-portfolio-v1/lattice-queue-v1.json').read_text())
            if state['status']=='completed':break
            if state['status']=='stopped':raise RuntimeError('Prior queue interrupted; explicit recovery required')
            if time.monotonic()-started>26000:raise TimeoutError('Prior queue total wall')
            time.sleep(10)
        for name,module,target,wall in (
            ('event-code-flow','scripts.run_event_code_flow','reports/local_event_code_flow_v1.json',8000),
            ('ssm-event-package','scripts.run_ssm_event_package','reports/local_ssm_event_package_v1.json',7500)):
            report['status']='running_'+name;write(REPORT,report)
            with (ROOT/f'artifacts/local-{name}-v1.log').open('ab') as out:
                child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m',module],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
                report['pid']=child.pid;write(REPORT,report);code=child.wait(timeout=wall)
            path=ROOT/target;result=json.loads(path.read_text()) if path.exists() else {}
            report['stages'].append(dict(id=name,exit_code=code,status=result.get('status','no_report'),report=target));write(REPORT,report)
            if code or result.get('status') not in ('completed','health_failed'):raise RuntimeError('Candidate requires explicit failure review: '+name)
        report.update(status='completed',wall_s=time.monotonic()-started);write(REPORT,report)
    except BaseException as exc:
        report.update(status='stopped',error=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-started);write(REPORT,report);raise


if __name__=='__main__':main()
