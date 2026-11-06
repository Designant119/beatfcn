"""Fetch all small mel10 JSON receipts read-only; never download feature arrays."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import modal


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / 'reports/full-database-cloud-v1/mel10.json'
DEST = ROOT / 'inputs/full-database-cloud-v1/mel-metadata.jsonl'
RECEIPT = ROOT / 'reports/full-database-cloud-v1/mel-metadata-receipt.json'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


async def fetch(volume, record, semaphore, attempts=4):
    async with semaphore:
        for attempt in range(1, attempts + 1):
            try:
                payload = b''.join([chunk async for chunk in volume.read_file.aio(record['path'])])
                value = json.loads(payload)
                expected = Path(record['path']).stem
                if set(value) != {'audio_sec', 'codec', 'frames', 'hop_sec', 'dtype'}:
                    raise ValueError('Unexpected mel receipt fields')
                if value['hop_sec'] != .01 or value['dtype'] != 'float16' or value['frames'] % 4:
                    raise ValueError('Mel receipt clock contract mismatch')
                return dict(audio_sha256=expected, path=record['path'], **value)
            except Exception:
                if attempt == attempts:
                    raise
                await asyncio.sleep(min(2 ** attempt, 8))


async def main(args):
    if DEST.exists():
        raise FileExistsError('Metadata output already exists')
    records = [record for record in json.loads(INVENTORY.read_text())['records']
               if record['path'].endswith('.json')]
    volume = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
    semaphore = asyncio.Semaphore(args.concurrency)
    partial = DEST.with_suffix('.jsonl.partial')
    partial.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                value = json.loads(line)
                done.add(value['audio_sha256'])
    started = time.monotonic()
    failures = []
    count_before = len(done)
    with partial.open('a') as stream:
        for batch_start in range(0, len(records), args.batch_size):
            jobs = [record for record in records[batch_start:batch_start + args.batch_size]
                    if Path(record['path']).stem not in done]
            if not jobs:
                continue
            results = await asyncio.gather(*(fetch(volume, record, semaphore) for record in jobs),
                return_exceptions=True)
            for record, result in zip(jobs, results):
                if isinstance(result, Exception):
                    failures.append(dict(path=record['path'],
                        error=f'{type(result).__name__}: {result}'))
                    continue
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                stream.flush()
                done.add(result['audio_sha256'])
            write_json(RECEIPT, dict(status='running', complete=False,
                fetched=len(done), total=len(records), failures=failures,
                wall_s=time.monotonic() - started))
    values = []
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                values.append(json.loads(line))
    values.sort(key=lambda value: value['audio_sha256'])
    lines = [json.dumps(value, ensure_ascii=False) for value in values]
    destination_text = '\n'.join(lines) + ('\n' if lines else '')
    DEST.write_text(destination_text)
    partial.unlink()
    frames = [value['frames'] for value in values]
    report = dict(schema='beatfcn.full-database-mel-metadata-v1', readonly=True,
        cloud_compute_calls=0, output=str(DEST), sha256=hashlib.sha256(DEST.read_bytes()).hexdigest(),
        records=len(values), expected=len(records), complete=len(values) == len(records) and not failures,
        failures=failures, wall_s=time.monotonic() - started,
        unique_audio=len({value['audio_sha256'] for value in values}),
        frames=dict(min=min(frames) if frames else None, max=max(frames) if frames else None,
            total=sum(frames), divisible_by_four=all(frame % 4 == 0 for frame in frames)),
        codecs=dict(Counter(value['codec'] for value in values)),
        limitations=['Metadata receipts only; feature array bytes and numerical values were not read.',
            'Receipt frames come from actual decoded sample padding, not source audio_sec rounding.',
            'No Modal compute function was called.'])
    write_json(RECEIPT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['complete']:
        raise SystemExit(1)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--concurrency', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=256)
    asyncio.run(main(parser.parse_args()))


if __name__ == '__main__':
    cli()
