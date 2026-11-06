"""Freeze a small hot-path selection after the first 898 completed graph songs."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / 'inputs/full-database-cloud-v1/mert-jobs.jsonl'
PRIOR = ROOT / 'reports/full-database-cloud-v1/mert-graph-diagnostic-selection-v1.json'
DEST = ROOT / 'reports/full-database-cloud-v1/mert-graph-hot-selection-v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    prior = {job['audio_sha256'] for job in json.loads(PRIOR.read_text())['jobs']}
    jobs = [json.loads(line) for line in JOBS.read_text().splitlines() if line.strip()]
    groups = defaultdict(list)
    for job in jobs:
        if (job['split'] == 'train' and job['audio_sec'] <= 360
                and job['audio_sha256'] not in prior):
            groups[job['expected_mel_frames'] // 4 + 1].append(job)
    selected = sorted((values for values in groups.values() if len(values) >= 2),
        key=lambda values: (-sum(job['audio_sec'] for job in values),
                            values[0]['audio_sha256']))[:24]
    values = [job for group in selected for job in group]
    result = dict(schema='beatfcn.full-database-mert-graph-hot-selection.v1',
        selection='24 highest-work remaining repeated exact-shape groups; writes into the completed diagnostic-v2 output root',
        songs=len(values), total_audio_seconds=sum(job['audio_sec'] for job in values),
        repeated_shape_groups=len(selected), largest_group=max(map(len, selected)),
        jobs_source='inputs/full-database-cloud-v1/mert-jobs.jsonl', jobs_sha256=sha(JOBS),
        prior_selection_sha256=sha(PRIOR),
        output_root='full-database-v1/mert-graph-diagnostic-v2', jobs=values)
    DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'jobs'}, indent=2))


if __name__ == '__main__':
    main()
