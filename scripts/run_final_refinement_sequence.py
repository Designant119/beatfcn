"""After current recipes: explicit A admission revision, then noisy refiner."""
import json
from pathlib import Path
import subprocess
import time

from scripts.run_local_portfolio_scale import write


ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'reports/local-portfolio-v1/final-refinement-sequence-v2.json'


def main():
    if REPORT.exists():raise FileExistsError('Preserve final refinement sequence')
    report=dict(status='waiting_for_prior_sequence',stages=[],cloud_compute_calls=0);write(REPORT,report);started=time.monotonic()
    try:
        while True:
            prior=json.loads((ROOT/'reports/local-portfolio-v1/repaired-sequence-v2.json').read_text());cache=json.loads((ROOT/'reports/local_noisy_proposal_cache_v1.json').read_text())
            if prior['status']=='completed' and cache['status']=='completed':break
            if prior['status']=='stopped' or cache['status']=='stopped':raise RuntimeError('Prior task needs explicit recovery')
            if time.monotonic()-started>30000:raise TimeoutError('Prior sequence wall')
            time.sleep(10)
        for name,module,target,wall in (
            ('canonical-package-quality-v2','scripts.run_canonical_quality_v2','reports/local_canonical_package_quality_v2.json',6300),
            ('noisy-proposal-refiner-v1','scripts.run_noisy_proposal_refiner','reports/local_noisy_proposal_refiner_v1.json',5000)):
            report['status']='running_'+name;write(REPORT,report)
            with (ROOT/f'artifacts/local-{name}.log').open('ab') as out:
                child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m',module],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
                report['pid']=child.pid;write(REPORT,report);code=child.wait(timeout=wall)
            result=json.loads((ROOT/target).read_text());report['stages'].append(dict(id=name,exit_code=code,status=result['status'],report=target));write(REPORT,report)
            if code or result['status'] not in ('completed','health_failed'):raise RuntimeError('Explicit candidate failure review required: '+name)
        report.update(status='completed',wall_s=time.monotonic()-started);write(REPORT,report)
    except BaseException as exc:
        report.update(status='stopped',error=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-started);write(REPORT,report);raise


if __name__=='__main__':main()
