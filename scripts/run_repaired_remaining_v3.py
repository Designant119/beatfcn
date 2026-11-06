"""Resume after the zero-update MPS operator fix; preserve all prior attempts."""
import json
from pathlib import Path
import subprocess
import time

from scripts.run_local_portfolio_scale import write


ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'reports/local-portfolio-v1/repaired-remaining-v3.json'


def main():
    if REPORT.exists():raise FileExistsError('Preserve explicit recovery sequence')
    for name in ('repaired-sequence-v2','final-refinement-sequence-v2'):
        prior=json.loads((ROOT/f'reports/local-portfolio-v1/{name}.json').read_text())
        if prior['status']!='stopped':raise ValueError('Superseded queues must be terminal')
    report=dict(status='running',stages=[],cloud_compute_calls=0);write(REPORT,report);started=time.monotonic()
    try:
        for name,module,args,target,wall in (
            ('event-code-flow-v2','scripts.run_event_code_flow',['--repair'],'reports/local_event_code_flow_v2.json',7200),
            ('ssm-event-package-v1','scripts.run_ssm_event_package',[],'reports/local_ssm_event_package_v1.json',7500),
            ('canonical-package-quality-v2','scripts.run_canonical_quality_v2',[],'reports/local_canonical_package_quality_v2.json',6300),
            ('noisy-proposal-refiner-v1','scripts.run_noisy_proposal_refiner',[],'reports/local_noisy_proposal_refiner_v1.json',5000)):
            moment=time.monotonic();report['status']='running_'+name;write(REPORT,report)
            with (ROOT/f'artifacts/local-{name}.log').open('ab') as out:
                child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m',module]+args,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
                report['pid']=child.pid;write(REPORT,report);code=child.wait(timeout=wall)
            path=ROOT/target;result=json.loads(path.read_text()) if path.exists() else {}
            report['stages'].append(dict(id=name,exit_code=code,status=result.get('status','no_report'),report=target,full_stage_wall_s=time.monotonic()-moment));write(REPORT,report)
            if code or result.get('status') not in ('completed','health_failed'):raise RuntimeError('Explicit failure review required: '+name)
        report.update(status='completed',wall_s=time.monotonic()-started);write(REPORT,report)
    except BaseException as exc:
        report.update(status='stopped',error=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-started);write(REPORT,report);raise


if __name__=='__main__':main()
