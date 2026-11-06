"""Local concurrent resume audit: read every feature receipt.json over the API.

The in-container sweep is gone: Modal Volume metadata operations serialize, and
~46k per-folder FUSE ops stalled formal-v2/v3. This audit reads receipt.json via
the Volume API from the Mac (the proven mel-verification pattern), freezes the
skip/quarantine lists, and the container resume becomes pure set membership.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

from beatfcn.audio.mert2 import REVISION
from beatfcn.cloud.full_database_app import FEATURE_FILES


ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / 'inputs/full-database-cloud-v1/mert-jobs.jsonl'
DEST = ROOT / 'reports/full-database-cloud-v1/mert-eager-resume-audit-v1.json'
REMOTE = 'full-database-v1/mert-graph-diagnostic-v2/features/{}/receipt.json'


def identity(receipt, job):
    decoded = receipt.get('decoded_samples')
    if type(decoded) is not int or decoded <= 0:
        return 'decoded_samples invalid'
    frames = (decoded + 959) // 960
    if (not receipt.get('complete') or receipt.get('audio_sha256') != job['audio_sha256']
            or receipt.get('metadata', {}).get('revision') != REVISION
            or receipt.get('metadata', {}).get('real_samples') != decoded
            or receipt.get('frames') != frames
            or job['expected_mel_frames'] != 4 * (decoded // 960)
            or set(receipt.get('files', {})) != FEATURE_FILES):
        return 'receipt identity or closure differs'
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--concurrency', type=int, default=48)
    arguments = parser.parse_args()
    jobs = [json.loads(line) for line in JOBS.read_text().splitlines() if line.strip()]
    volume = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
    started = time.monotonic()
    skip_ids, invalid_ids, missing_ids = [], [], []
    failures = []

    def audit(job):
        try:
            payload = b''.join(volume.read_file(REMOTE.format(job['audio_sha256'])))
            receipt = json.loads(payload)
        except Exception:
            return job, 'missing'
        except BaseException as exc:  # pragma: no cover
            return job, f'read_error: {type(exc).__name__}'
        reason = identity(receipt, job)
        return job, reason or 'ok'

    with ThreadPoolExecutor(max_workers=arguments.concurrency) as pool:
        for index, (job, outcome) in enumerate(pool.map(audit, jobs)):
            if outcome == 'ok':
                skip_ids.append(job['audio_sha256'])
            elif outcome == 'missing':
                missing_ids.append(job['audio_sha256'])
            elif outcome.startswith('read_error'):
                failures.append(dict(audio_sha256=job['audio_sha256'], error=outcome))
            else:
                invalid_ids.append(job['audio_sha256'])
            if (index + 1) % 1000 == 0:
                print(dict(stage='resume-audit', checked=index + 1, total=len(jobs),
                    skip=len(skip_ids), invalid=len(invalid_ids),
                    missing=len(missing_ids), wall_s=round(time.monotonic() - started, 1)),
                    flush=True)
    result = dict(schema='beatfcn.full-database-mert-eager-resume-audit.v1',
        revision=REVISION, remote_root='full-database-v1/mert-graph-diagnostic-v2',
        jobs=len(jobs), skip_ids=skip_ids, invalid_ids=invalid_ids,
        missing_ids=missing_ids, read_failures=failures,
        wall_s=time.monotonic() - started)
    DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: (len(value) if isinstance(value, list) else value)
        for key, value in result.items() if key not in ('schema',)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
