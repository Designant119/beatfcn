"""Freeze the two-song exact-shape batch=2 CUDA-graph probe selection."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / 'inputs/full-database-cloud-v1/mert-jobs.jsonl'
DEST = ROOT / 'reports/full-database-cloud-v1/mert-batch2-graph-probe-selection-v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    jobs = [json.loads(line) for line in JOBS.read_text().splitlines() if line.strip()]
    groups = defaultdict(list)
    for job in jobs:
        if job['split'] == 'train' and job['audio_sec'] <= 360:
            groups[job['expected_mel_frames'] // 4 + 1].append(job)
    key, values = max(groups.items(), key=lambda item: len(item[1]))
    # Prefer one OSU and one Phira so both verified decode paths are exercised.
    chosen = []
    for source in ('osu', 'phira'):
        chosen.append(next(job for job in values if job['source'] == source))
    if len({job['audio_sha256'] for job in chosen}) != 2:
        raise ValueError('Batch2 probe selection must contain two distinct songs')
    result = dict(schema='beatfcn.full-database-mert-batch2-graph-selection.v1',
        batch=2, shape_cells=int(key),
        selection='one OSU + one Phira from the largest repeated exact padded-shape group, both <=360s',
        songs=len(chosen), total_audio_seconds=sum(job['audio_sec'] for job in chosen),
        jobs_source='inputs/full-database-cloud-v1/mert-jobs.jsonl', jobs_sha256=sha(JOBS),
        output_root='full-database-v1/mert-batch2-graph-probe-v1', jobs=chosen)
    DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key_: value for key_, value in result.items() if key_ != 'jobs'}, indent=2))


if __name__ == '__main__':
    main()
