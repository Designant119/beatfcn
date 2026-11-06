"""Freeze a balanced 192-song CPU decode concurrency probe selection."""
import hashlib
import json
import math
from pathlib import Path

from scripts.prepare_full_database_mert_pilot_v2 import RAW_EXCLUDE


ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / 'inputs/full-database-cloud-v1/mert-jobs.jsonl'
DEST = ROOT / 'reports/full-database-cloud-v1/mert-decode-probe-selection-v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantile(values, count):
    return [values[math.floor(index * len(values) / count)]
            for index in range(count)]


def main():
    jobs = [json.loads(line) for line in JOBS.read_text().splitlines() if line.strip()]
    selected = []
    for source in ('osu', 'phira'):
        values = sorted((job for job in jobs if job['split'] == 'train'
                and job['source'] == source and job['audio_sha256'] not in RAW_EXCLUDE),
            key=lambda job: (job['audio_sec'], job['audio_sha256']))
        selected.extend(quantile(values, 96))
    if len(selected) != 192 or len({job['audio_sha256'] for job in selected}) != 192:
        raise ValueError('Unexpected decode-probe selection')
    result = dict(schema='beatfcn.full-database-mert-decode-probe-selection.v1',
        selection='96 OSU + 96 Phira train songs across duration quantiles',
        songs=len(selected), total_audio_seconds=sum(job['audio_sec'] for job in selected),
        jobs_source='inputs/full-database-cloud-v1/mert-jobs.jsonl', jobs_sha256=sha(JOBS),
        jobs=selected)
    DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'jobs'}, indent=2))


if __name__ == '__main__':
    main()
