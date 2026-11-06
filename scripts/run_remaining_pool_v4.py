"""Explicit root recovery after the diagnosed empty-proposal implementation stop.

All earlier supervisors are terminal and preserved. One MPS child at a time;
no retries, changed quality gates, cloud calls or hidden training continuation.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from scripts.run_local_portfolio_scale import write


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/local-portfolio-v1/remaining-pool-v4.json'
STAGES = (
    ('noisy-proposal-refiner-v2', 'scripts.run_noisy_proposal_refiner_v2', 'reports/local_noisy_proposal_refiner_v2.json', 5500),
    ('group-diffusion-quality-v1', 'scripts.run_group_diffusion_quality', 'reports/local_group_diffusion_quality_v1.json', 6900),
    ('mert-context-cache-v1', 'scripts.cache_short_context_mert', 'reports/local_mert_context_cache_v1.json', 3750),
    ('mert-context-pair-v1', 'scripts.run_mert_context_pair', 'reports/local_mert_context_pair_v1.json', 17500),
    ('attention-control-quality-v1', 'scripts.run_attention_control_quality', 'reports/local_attention_control_quality_v1.json', 9000),
    ('concat-control-quality-v1', 'scripts.run_concat_control_quality', 'reports/local_concat_control_quality_v1.json', 9000),
    ('raw-audio-translation-v1', 'scripts.audit_raw_audio_translation', 'reports/local_raw_audio_translation_v1.json', 1900),
)


def main():
    if REPORT.exists():
        raise FileExistsError('No automatic recovery/retry')
    old = ROOT / 'reports/local_noisy_proposal_refiner_v1.json'
    failure = json.loads(old.read_text())
    if failure['status'] != 'stopped' or failure['quality_updates'] != 267:
        raise ValueError('Explicit recovery must match the preserved failure')
    for name in ('repaired-remaining-v3', 'pool-completion-queue-v1',
                 'attention-after-pool-v1', 'concat-after-attention-v1',
                 'raw-translation-after-concat-v1'):
        if json.loads((ROOT / f'reports/local-portfolio-v1/{name}.json').read_text())['status'] != 'stopped':
            raise ValueError('Prior supervisors must all be terminal')
    report = dict(status='running', stages=[], concurrency=1, automatic_retries=0,
                  started_at=datetime.now(timezone.utc).isoformat(),
                  predecessor='reports/local_noisy_proposal_refiner_v1.json',
                  root_repair='Real zero-proposal request; fixed zero contribution, original warmstart, new version/fresh trial. Old100health+267quality retained and charged.',
                  cloud_compute_calls=0)
    write(REPORT, report)
    start = time.monotonic()
    try:
        for ident, module, target, wall in STAGES:
            report.update(status='running_' + ident)
            write(REPORT, report)
            moment = time.monotonic()
            with (ROOT / f'artifacts/local-{ident}.log').open('x') as out:
                child = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', '-m', module], cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
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
            value = json.loads((ROOT / target).read_text()) if (ROOT / target).exists() else {}
            report['stages'].append(dict(id=ident, exit_code=code, status=value.get('status'),
                                        report=target, full_stage_wall_s=time.monotonic() - moment))
            write(REPORT, report)
            if code or value.get('status') != 'completed':
                raise RuntimeError('Explicit diagnosis required: ' + ident)
        report.update(status='completed', wall_s=time.monotonic() - start)
        write(REPORT, report)
    except BaseException as exc:
        report.update(status='stopped', error=f'{type(exc).__name__}: {exc}', wall_s=time.monotonic() - start)
        write(REPORT, report)
        raise


if __name__ == '__main__':
    main()
