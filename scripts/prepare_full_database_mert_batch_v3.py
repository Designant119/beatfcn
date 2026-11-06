"""Freeze the true-batch MERT acceptance selection."""
import hashlib
import json
import math
from pathlib import Path

from scripts.prepare_full_database_mert_pilot_v2 import RAW_EXCLUDE


ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / 'inputs/full-database-cloud-v1/mert-jobs.jsonl'
V2 = ROOT / 'reports/full-database-cloud-v1/mert-pilot-selection-v2.json'
DEST = ROOT / 'reports/full-database-cloud-v1/mert-batch-selection-v3.json'
COUNT = 3072


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    jobs = [json.loads(line) for line in JOBS.read_text().splitlines() if line.strip()]
    v2 = json.loads(V2.read_text())
    eligible = {job['audio_sha256']: job for job in jobs if job['split'] == 'train'
        and job['audio_sha256'] not in RAW_EXCLUDE}
    selected = {job['audio_sha256']: eligible[job['audio_sha256']] for job in v2['jobs']}
    candidates = sorted((job for audio, job in eligible.items() if audio not in selected),
        key=lambda job: (job['audio_sec'], job['audio_sha256']))
    needed = COUNT - len(selected)
    for index in range(needed):
        job = candidates[math.floor(index * len(candidates) / needed)]
        selected[job['audio_sha256']] = job
    values = sorted(selected.values(), key=lambda job: (job['audio_sec'], job['audio_sha256']))
    if len(values) != COUNT or len({job['audio_sha256'] for job in values}) != COUNT:
        raise ValueError('Invalid true-batch selection')
    result = dict(schema='beatfcn.full-database-mert-batch-selection.v1',
        selection='3072 train-only duration quantiles; all v2 IDs included; raw failure excluded',
        count=COUNT, total_audio_seconds=sum(job['audio_sec'] for job in values),
        jobs_source='inputs/full-database-cloud-v1/mert-jobs.jsonl', jobs_sha256=sha(JOBS),
        v2_selection_sha256=sha(V2), v2_ids_included=len(v2['jobs']),
        raw_content_excluded=sorted(RAW_EXCLUDE), output_root='full-database-v1/mert-batch-v3',
        jobs=values)
    DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'jobs'}, indent=2))


if __name__ == '__main__':
    main()
