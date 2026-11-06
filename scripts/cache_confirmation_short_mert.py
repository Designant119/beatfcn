"""Feature-only confirmation extension, permitted only after finalist freeze.

Uses the exact already-tested32s extractor. No chart targets, optimizer,
thresholds or model selection; existing81audio payloads are never overwritten.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from scripts import cache_short_context_mert as engine
from scripts.run_local_portfolio_scale import sha, write


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'bench/rhythm-v2/local-mert-confirmation-context-cache-v1.json'
REPORT = ROOT / 'reports/local_mert_confirmation_context_cache_v1.json'
ART = ROOT / 'artifacts/local-mert-confirmation-context-cache-v1'


def prepare():
    from scripts.run_frontend_canvas_comparison import setup
    selection_path = ROOT / 'reports/local_portfolio_finalist_selection_v1.json'
    selection = json.loads(selection_path.read_text())
    qualified = []
    for rank, entry in enumerate(selection['finalist_identities'], 1):
        rp = ROOT / f'reports/local_finalist_r{rank}_seed2_v1.json'
        r = json.loads(rp.read_text())
        if r['status'] not in ('completed', 'health_failed', 'stopped'):
            raise ValueError('Record every secondseed terminal before feature extension')
        if entry['arm'] == 'short32' and r.get('reproduced'):
            qualified.append(dict(rank=rank, original_weight_sha256=entry['checkpoint_sha256'],
                                  seed2_weight_sha256=r['final_weight_sha256'], report_sha256=sha(rp)))
    if not qualified:
        raise ValueError('No frozen reproduced short-context finalist')
    original_plan = ROOT / 'bench/rhythm-v2/local-mert-context-cache-v1.json'
    original_report = ROOT / 'reports/local_mert_context_cache_v1.json'
    base = json.loads(original_plan.read_text())
    if sha(ROOT / base['checkpoint_dir'] / 'model.safetensors') != base['weight_sha256']:
        raise ValueError('Frozen MERT checkpoint changed before confirmation extraction')
    existing = json.loads(original_report.read_text())
    if not existing['complete']:
        raise ValueError('Existing context cache incomplete')
    comparison = json.loads((ROOT / 'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text())
    panel, _ = setup(comparison)
    indices = json.loads((ROOT / 'bench/rhythm-v2/local-portfolio-scale-v1.json').read_text())['indices']['confirm']
    audios = sorted({panel.records[i]['audio_sha256'] for i in indices})
    if set(audios) & {r['audio_sha256'] for r in existing['records']}:
        raise ValueError('Confirmation audio already in training/development cache')
    acquisition = json.loads((ROOT / 'bench/rhythm-v2/d96-local-mert-cache-v1.json').read_text())
    sources = {r['audio_sha256']: r for r in acquisition['audios']}
    plan = dict(base, id='local-mert-confirmation-context-cache-v1', audios=[sources[a] for a in audios],
        selection_sha256=sha(selection_path), frozen_finalists=qualified,
        original_plan_sha256=sha(original_plan), original_report_sha256=sha(original_report),
        total_wall_s=1200, confirmation_scored=False,
        note='Audio feature extraction only, after model/seed identities frozen. No hidden targets read or scores produced; original81cache preserved.')
    plan['source_sha256'] = {**base['source_sha256'], 'scripts/cache_confirmation_short_mert.py': sha(Path(__file__))}
    write(PLAN, plan)
    for name in plan['source_sha256']:
        dest = ART / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    shutil.copy2(PLAN, ART / 'source/frozen-plan.json')
    return plan, existing


def main():
    if PLAN.exists() or REPORT.exists():
        raise FileExistsError('No confirmation cache retry')
    plan, existing = prepare()
    start = time.monotonic()
    result = dict(id=plan['id'], status='encoding', complete=False, planned_songs=len(plan['audios']),
                  committed_songs=0, records=[], plan_sha256=sha(PLAN), confirmation_scored=False, cloud_compute_calls=0)
    write(REPORT, result)
    try:
        new = []
        for group in range((len(plan['audios']) + 7) // 8):
            with (ART / f'worker-{group:02d}.log').open('x') as out:
                child = subprocess.Popen([str(ROOT / plan['environment'] / 'bin/python'), '-u', '-m',
                    'scripts.cache_confirmation_short_mert', '--worker', str(group)], cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
                try:
                    code = child.wait(timeout=min(660, plan['total_wall_s'] - (time.monotonic() - start)))
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    raise
            if code:
                raise RuntimeError('Explicit confirmation feature worker failure')
            value = json.loads((ART / f'worker-{group:02d}.json').read_text())
            if not value['complete']:
                raise ValueError('Incomplete feature worker')
            new.extend(value['records'])
            write(ART / 'new-records.json', new)
        records = existing['records'] + new
        result.update(status='completed', complete=True, planned_songs=len(records), committed_songs=len(records),
            records=records, newly_extracted_songs=len(new), previous_cache_sha256=plan['original_report_sha256'],
            wall_s=time.monotonic() - start)
        write(REPORT, result)
    except BaseException as exc:
        result.update(status='stopped', error=f'{type(exc).__name__}: {exc}', wall_s=time.monotonic() - start)
        write(REPORT, result)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    if args.worker is None:
        main()
    else:
        engine.PLAN = PLAN
        engine.REPORT = REPORT
        engine.ART = ART
        engine.worker(args.worker)
