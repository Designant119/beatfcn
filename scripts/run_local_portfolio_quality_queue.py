"""Run the two already-healthy quality candidates serially, without retries."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/local-portfolio-v1/quality-queue-v1.json'


def main():
    if REPORT.exists():raise FileExistsError('Preserve the existing queue; no automatic reruns')
    data=dict(id='local-portfolio-quality-queue-v1',status='running',cloud_compute_calls=0,
        rationale='Continue authorized local experiments: pretrained encoders are optional means; evaluate actual cross-song learned acoustic and event-package routes.',
        concurrency=1,automatic_retries=0,
        started_at=datetime.now(timezone.utc).isoformat(),stages=[])
    child=None
    def save():
        REPORT.parent.mkdir(parents=True,exist_ok=True)
        temp=REPORT.with_suffix('.partial');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(REPORT)
    def stop(*_):
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGINT)
        raise KeyboardInterrupt('External queue stop')
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop);save()
    try:
        for candidate,wall in (('mel',3900),('package',5700)):
            health=json.loads((ROOT/f'reports/local_portfolio_{candidate}_health_v1.json').read_text())
            if not health['health_passed']:raise ValueError('Queued candidate did not pass health')
            log=ROOT/f'artifacts/local-portfolio-v1-{candidate}-quality.log'
            stage=dict(candidate=candidate,status='starting',outer_wall_limit_s=wall,log=str(log.relative_to(ROOT)))
            data['stages'].append(stage);save()
            with log.open('x') as handle:
                child=subprocess.Popen([sys.executable,'-u','-m','scripts.run_local_portfolio_candidate_quality',
                    '--candidate',candidate],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                stage.update(pid=child.pid,status='running');save()
                print(json.dumps(stage),flush=True)
                try:code=child.wait(timeout=wall)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGINT)
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL);child.wait()
                    stage.update(status='outer_deadline_stopped',exit_code=child.returncode);save();raise
            stage.update(status='completed' if code==0 else 'failed',exit_code=code,
                finished_at=datetime.now(timezone.utc).isoformat());save();print(json.dumps(stage),flush=True)
            if code!=0:raise RuntimeError(f'{candidate} exited {code}; no retries or silent continuation')
            child=None
        data['status']='completed';save()
    except BaseException as exc:
        data.update(status='stopped',failure=f'{type(exc).__name__}: {exc}');save();raise


if __name__=='__main__':main()
