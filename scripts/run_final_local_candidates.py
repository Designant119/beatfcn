"""Serial local supervisor: finish the active full-song job, then grouped AR.

No retries or new council. A missing live process without a terminal report is
an interruption, never a scientific failure or a reason to invent completion.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from scripts.run_local_portfolio_scale import write


ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/local-portfolio-v1/final-candidates-queue-v1.json'


def main():
    if REPORT.exists():raise FileExistsError('Preserve queue')
    process=json.loads((ROOT/'artifacts/local-whole-song-arrangement-v1/process.json').read_text())
    report=dict(status='waiting_for_active_whole_song',whole_song_pid=process['pid'],stages=[],cloud_compute_calls=0)
    write(REPORT,report);started=time.monotonic();child=None
    def stopped(*_):
        if child is not None and child.poll() is None:child.terminate()
        raise KeyboardInterrupt('Supervisor stopped; no automatic restart')
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    try:
        while True:
            path=ROOT/'reports/local_whole_song_arrangement_v1.json'
            state=json.loads(path.read_text()) if path.exists() else {}
            if state.get('status') in ('completed','health_failed','stopped'):
                report['stages'].append(dict(stage='whole_song',status=state['status'],report=str(path.relative_to(ROOT))));break
            try:os.kill(process['pid'],0)
            except ProcessLookupError:raise RuntimeError('Whole-song process absent without terminal report; preserve interruption')
            if time.monotonic()-started>11000:raise TimeoutError('Whole-song supervision wall')
            time.sleep(10)
        report['status']='running_grouped_chart';write(REPORT,report)
        log=ROOT/'artifacts/local-grouped-chart-v1.log'
        with log.open('ab') as out:
            child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','scripts.run_grouped_chart_experiment'],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
            report['grouped_chart_pid']=child.pid;write(REPORT,report)
            code=child.wait(timeout=7900)
        path=ROOT/'reports/local_grouped_chart_v1.json';state=json.loads(path.read_text()) if path.exists() else {}
        report['stages'].append(dict(stage='grouped_chart',exit_code=code,status=state.get('status','no_terminal_report'),report=str(path.relative_to(ROOT))))
        report['status']='completed' if code==0 and state.get('status') in ('completed','health_failed') else 'stopped'
        report['wall_s']=time.monotonic()-started;write(REPORT,report)
    except BaseException as exc:
        report.update(status='stopped',error=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-started)
        write(REPORT,report);raise


if __name__=='__main__':main()
