"""Complete the explicitly reopened and still-unmeasured local contracts.

Wait for the existing serial supervisor to terminate; an implementation
failure stops here for root diagnosis. No cloud or automatic retry.
"""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time

from scripts.run_local_portfolio_scale import write


ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'reports/local-portfolio-v1/pool-completion-queue-v1.json'


def main():
    if REPORT.exists():raise FileExistsError('Preserve queue')
    predecessor=ROOT/'reports/local-portfolio-v1/repaired-remaining-v3.json'
    report=dict(status='waiting_for_serial_predecessor',predecessor=str(predecessor.relative_to(ROOT)),stages=[],
        started_at=datetime.now(timezone.utc).isoformat(),concurrency=1,cloud_compute_calls=0,automatic_retries=0)
    write(REPORT,report);started=time.monotonic()
    try:
        while True:
            state=json.loads(predecessor.read_text())
            if state['status']=='completed':break
            if state['status']=='stopped':raise RuntimeError('Predecessor stopped; explicit root diagnosis required')
            if time.monotonic()-started>24000:raise TimeoutError('Predecessor wait bound')
            time.sleep(5)
        for name,module,python,target,wall in (
            ('group-diffusion-quality-v1','scripts.run_group_diffusion_quality','.venv/bin/python','reports/local_group_diffusion_quality_v1.json',6900),
            ('mert-context-cache-v1','scripts.cache_short_context_mert','.venv/bin/python','reports/local_mert_context_cache_v1.json',3750),
            ('mert-context-pair-v1','scripts.run_mert_context_pair','.venv/bin/python','reports/local_mert_context_pair_v1.json',17500)):
            report['status']='running_'+name;moment=time.monotonic();write(REPORT,report)
            with (ROOT/f'artifacts/local-{name}.log').open('x') as out:
                child=subprocess.Popen([str(ROOT/python),'-u','-m',module],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
                report['pid']=child.pid;write(REPORT,report)
                try:code=child.wait(timeout=wall)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    raise
            result=json.loads((ROOT/target).read_text()) if (ROOT/target).exists() else {}
            report['stages'].append(dict(id=name,exit_code=code,status=result.get('status','no_report'),report=target,full_stage_wall_s=time.monotonic()-moment));write(REPORT,report)
            if code or result.get('status')!='completed':raise RuntimeError('Explicit diagnosis required: '+name)
        report.update(status='completed',wall_s=time.monotonic()-started);write(REPORT,report)
    except BaseException as exc:
        report.update(status='stopped',error=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-started);write(REPORT,report);raise


if __name__=='__main__':main()
