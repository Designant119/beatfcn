"""Freeze five disjoint 96-song decode blocks for the sustained-cold probe."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / 'inputs/full-database-cloud-v1/mert-jobs.jsonl'
PRIOR = ROOT / 'reports/full-database-cloud-v1/mert-decode-probe-selection-v1.json'
DEST = ROOT / 'reports/full-database-cloud-v1/mert-decode-probe-selection-v2.json'
BLOCKS = 5
BLOCK_SIZE = 96


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    prior = {job['audio_sha256'] for job in json.loads(PRIOR.read_text())['jobs']}
    jobs = [json.loads(line) for line in JOBS.read_text().splitlines() if line.strip()]
    # Interleave splits/sources then cut consecutive disjoint blocks: every phase
    # decodes songs no earlier phase touched, so later phases measure sustained
    # cold-file throughput rather than page-cache reuse.
    ordered = [job for index, job in enumerate(jobs) if index % 7 == 0
        and job['audio_sha256'] not in prior][:BLOCKS * BLOCK_SIZE]
    if len(ordered) != BLOCKS * BLOCK_SIZE:
        raise ValueError('Sustained-cold probe selection is short')
    blocks = [ordered[offset:offset + BLOCK_SIZE] for offset in
        range(0, len(ordered), BLOCK_SIZE)]
    result = dict(schema='beatfcn.full-database-mert-decode-probe-selection.v2',
        selection='five disjoint 96-song blocks; each phase decodes never-before-read files',
        blocks=[dict(index=index, songs=len(block),
            total_audio_seconds=sum(job['audio_sec'] for job in block))
            for index, block in enumerate(blocks)],
        jobs_source='inputs/full-database-cloud-v1/mert-jobs.jsonl', jobs_sha256=sha(JOBS),
        prior_selection_sha256=sha(PRIOR), flat_jobs=ordered)
    DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'flat_jobs'},
        indent=2))


if __name__ == '__main__':
    main()
