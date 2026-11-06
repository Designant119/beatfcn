"""Freeze the exact-shape repeated MERT CUDA-graph acceptance selection."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from scripts.prepare_full_database_mert_pilot_v2 import RAW_EXCLUDE


ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / 'inputs/full-database-cloud-v1/mert-jobs.jsonl'
DEST = ROOT / 'reports/full-database-cloud-v1/mert-graph-selection-v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    jobs = [json.loads(line) for line in JOBS.read_text().splitlines() if line.strip()]
    groups = defaultdict(list)
    for job in jobs:
        if job['split'] != 'train' or job['audio_sec'] > 360:
            continue
        if job['audio_sha256'] in RAW_EXCLUDE:
            continue
        # Existing mel receipts carry floor(decoded_samples/960). The MERT padded
        # cell count is normally that plus one; decode verifies the exact shape.
        groups[job['expected_mel_frames'] // 4 + 1].append(job)
    repeated = sorted((values for values in groups.values() if len(values) >= 2),
        key=lambda values: (-len(values), values[0]['audio_sha256']))
    values = [job for group in repeated for job in group]
    if len(values) < 6000 or len({job['audio_sha256'] for job in values}) != len(values):
        raise ValueError('Unexpected exact-shape graph selection')
    result = dict(schema='beatfcn.full-database-mert-graph-selection.v1',
        selection='train-only songs <=360s with repeated exact padded MERT shape',
        songs=len(values), total_audio_seconds=sum(job['audio_sec'] for job in values),
        repeated_shape_groups=len(repeated), largest_group=max(map(len, repeated)),
        jobs_source='inputs/full-database-cloud-v1/mert-jobs.jsonl', jobs_sha256=sha(JOBS),
        output_root='full-database-v1/mert-graph-v1', jobs=values)
    DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'jobs'}, indent=2))


if __name__ == '__main__':
    main()
