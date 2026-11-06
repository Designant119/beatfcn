"""Freeze a larger real-work pilot selection with the v1 jobs included."""
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / 'inputs/full-database-cloud-v1/mert-jobs.jsonl'
V1 = ROOT / 'reports/full-database-cloud-v1/mert-pilot-selection-v1.json'
DEST = ROOT / 'reports/full-database-cloud-v1/mert-pilot-selection-v2.json'
COUNT = 1536
RAW_EXCLUDE = {'31123bc5b0a08b31486386e55eb2214b887971d8bfdc27313efaf779d92b442f'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(jobs, v1_jobs, count=COUNT):
    by_id = {job['audio_sha256']: job for job in jobs if job['split'] == 'train'
             and job['audio_sha256'] not in RAW_EXCLUDE}
    selected = {job['audio_sha256']: by_id[job['audio_sha256']] for job in v1_jobs}
    candidates = sorted((job for audio, job in by_id.items() if audio not in selected),
        key=lambda job: (job['audio_sec'], job['audio_sha256']))
    needed = count - len(selected)
    if needed < 0 or len(candidates) < needed:
        raise ValueError('Insufficient eligible train jobs for pilot v2')
    for index in range(needed):
        job = candidates[math.floor(index * len(candidates) / needed)]
        selected[job['audio_sha256']] = job
    values = sorted(selected.values(), key=lambda job: (job['audio_sec'], job['audio_sha256']))
    if len(values) != count or len({job['audio_sha256'] for job in values}) != count:
        raise ValueError('Pilot v2 selection is not unique at the requested size')
    return values


def main():
    jobs = [json.loads(line) for line in JOBS.read_text().splitlines() if line.strip()]
    v1 = json.loads(V1.read_text())
    values = select(jobs, v1['jobs'])
    result = dict(schema='beatfcn.full-database-mert-pilot-selection.v2',
        selection='1536 train-only deterministic duration quantiles; all v1 IDs forced; raw content failure excluded',
        count=len(values), total_audio_seconds=sum(job['audio_sec'] for job in values),
        jobs_source='inputs/full-database-cloud-v1/mert-jobs.jsonl', jobs_sha256=sha(JOBS),
        v1_selection_sha256=sha(V1), v1_ids_included=len(v1['jobs']),
        raw_content_excluded=sorted(RAW_EXCLUDE), output_root='full-database-v1/mert-pilot-v2',
        jobs=values)
    DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'jobs'}, indent=2))


if __name__ == '__main__':
    main()
