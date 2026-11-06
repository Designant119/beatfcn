"""Freeze the four-song multi-stream eager numerics probe selection."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / 'inputs/full-database-cloud-v1/mert-jobs.jsonl'
DEST = ROOT / 'reports/full-database-cloud-v1/mert-eager-streams-selection-v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    jobs = [json.loads(line) for line in JOBS.read_text().splitlines() if line.strip()]
    train = [job for job in jobs if job['split'] == 'train']
    chosen = [
        min((job for job in train if job['source'] == 'osu' and job['audio_sec'] <= 360),
            key=lambda job: abs(job['audio_sec'] - 90)),
        min((job for job in train if job['source'] == 'phira' and job['audio_sec'] <= 360),
            key=lambda job: abs(job['audio_sec'] - 90)),
        min((job for job in train if job['audio_sec'] <= 360),
            key=lambda job: abs(job['audio_sec'] - 300)),
        min((job for job in train if job['audio_sec'] > 360), key=lambda job: job['audio_sec']),
    ]
    if len({job['audio_sha256'] for job in chosen}) != 4:
        raise ValueError('Eager streams probe selection must be four distinct songs')
    result = dict(schema='beatfcn.full-database-mert-eager-streams-selection.v1',
        selection='two ~90s (osu+phira), one ~300s, and the shortest >360s windowed train song',
        songs=len(chosen), total_audio_seconds=sum(job['audio_sec'] for job in chosen),
        jobs_source='inputs/full-database-cloud-v1/mert-jobs.jsonl', jobs_sha256=sha(JOBS),
        output_root='full-database-v1/mert-eager-streams-probe-v1', jobs=chosen)
    DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'jobs'}, indent=2))
    print(json.dumps([dict(source=job['source'], audio_sec=job['audio_sec'],
        sha=job['audio_sha256'][:12]) for job in chosen], indent=2))


if __name__ == '__main__':
    main()
