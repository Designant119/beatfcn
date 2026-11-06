"""Deterministically select a duration-stratified train-only MERT pilot set."""
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / 'inputs/full-database-cloud-v1/mert-jobs.jsonl'
RECEIPT = ROOT / 'reports/full-database-cloud-v1/mert-pilot-selection-v1.json'
QUANTILES = (0., .05, .2, .4, .6, .8, .9, .95, .99, 1.)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded_seconds(duration):
    if duration <= 360:
        return duration
    return duration + math.ceil((duration - 300) / 200) * 100


def main():
    if RECEIPT.exists():
        raise FileExistsError('Pilot selection already exists')
    jobs = [json.loads(line) for line in JOBS.read_text().splitlines() if line.strip()]
    train = [job for job in jobs if job['split'] == 'train']
    selected = []
    for source in ('osu', 'phira'):
        values = sorted((job for job in train if job['source'] == source),
                        key=lambda job: (job['audio_sec'], job['audio_sha256']))
        if len(values) < len(QUANTILES):
            raise ValueError(f'Insufficient {source} train jobs for stratified pilot')
        for quantile in QUANTILES:
            index = min(len(values) - 1, max(0, round(quantile * (len(values) - 1))))
            job = values[index]
            if job['audio_sha256'] not in {item['audio_sha256'] for item in selected}:
                selected.append(job)
    selected.sort(key=lambda job: job['audio_sha256'])
    raw = sum(job['audio_sec'] for job in selected)
    encoded = sum(encoded_seconds(job['audio_sec']) for job in selected)
    report = dict(schema='beatfcn.full-database-mert-pilot-selection.v1',
        jobs_source=str(JOBS.relative_to(ROOT)), jobs_sha256=digest(JOBS),
        selection='train-only deterministic duration quantiles; no quality or prediction feedback',
        quantiles=list(QUANTILES), jobs=selected, job_count=len(selected),
        by_source=dict(osu=sum(job['source'] == 'osu' for job in selected),
            phira=sum(job['source'] == 'phira' for job in selected)),
        raw_audio_hours=raw / 3600, encoded_audio_hours=encoded / 3600,
        long_songs=sum(job['audio_sec'] > 360 for job in selected),
        cloud_compute_calls=0,
        limitations=['Selection is for throughput and correctness pilot design; not a quality benchmark.',
            'Validation/test audio remains unused.',
            'Final paid contract must add fresh rates, source closure, prerequisites and Owner reservation.'])
    RECEIPT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'jobs'},
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
