"""Read and verify every parsed catalogue mel10 array from cloud storage."""
import argparse
import asyncio
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import time

import modal
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'inputs/full-database-cloud-v1/manifest/charts.offset-v2-3d99ff6bd5e7.jsonl'
METADATA = ROOT / 'inputs/full-database-cloud-v1/mel-metadata.jsonl'
DEST = ROOT / 'inputs/full-database-cloud-v1/mel-array-content.jsonl'
RECEIPT = ROOT / 'reports/full-database-cloud-v1/mel-array-content-receipt-v1.json'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


async def read_with_digest(volume, remote, attempts=4):
    for attempt in range(1, attempts + 1):
        try:
            chunks = []
            hasher = hashlib.sha256()
            size = 0
            async for chunk in volume.read_file.aio(remote):
                chunks.append(chunk)
                hasher.update(chunk)
                size += len(chunk)
            return hasher.hexdigest(), size, b''.join(chunks)
        except Exception:
            if attempt == attempts:
                raise
            await asyncio.sleep(min(2 ** attempt, 10))


async def verify(volume, audio, expected_frames, semaphore):
    async with semaphore:
        remote = f'mel10/{audio}.npy'
        digest_value, size, payload = await read_with_digest(volume, remote)
        try:
            value = np.load(io.BytesIO(payload), allow_pickle=False)
        except Exception as exc:
            raise ValueError(f'Invalid mel NPY payload: {type(exc).__name__}: {exc}') from exc
        del payload
        finite = bool(np.isfinite(value).all())
        result = dict(audio_sha256=audio, remote=remote, bytes=size,
            sha256=digest_value, dtype=str(value.dtype), shape=list(value.shape),
            c_contiguous=bool(value.flags.c_contiguous), finite=finite,
            expected_frames=expected_frames,
            verified=value.dtype == np.float16 and value.shape == (expected_frames, 128)
                and finite and value.flags.c_contiguous)
        return result


async def main(args):
    if DEST.exists():
        raise FileExistsError('Mel array content output already exists')
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    parsed = {row['audio_sha256'] for row in rows if row.get('status') == 'parsed'}
    metadata = {json.loads(line)['audio_sha256']: json.loads(line)
                for line in METADATA.read_text().splitlines()}
    missing_metadata = sorted(parsed - set(metadata))
    if missing_metadata:
        raise ValueError(f'{len(missing_metadata)} parsed audio lack mel metadata')
    volume = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
    partial = DEST.with_suffix('.jsonl.partial')
    done = set()
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                value = json.loads(line)
                if value.get('verified'):
                    done.add(value['audio_sha256'])
    jobs = sorted(parsed - done)
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.monotonic()
    failures = []
    records = []
    with partial.open('a') as stream:
        async def run(audio):
            try:
                result = await verify(volume, audio, metadata[audio]['frames'], semaphore)
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                stream.flush()
                records.append(result)
            except Exception as exc:
                failures.append(dict(audio_sha256=audio,
                    error=f'{type(exc).__name__}: {exc}'))
        for index in range(0, len(jobs), args.batch_size):
            batch = jobs[index:index + args.batch_size]
            await asyncio.gather(*(run(audio) for audio in batch))
            write_json(RECEIPT, dict(status='running', complete=False,
                verified=len(done) + len(records), total=len(parsed), failures=failures,
                wall_s=time.monotonic() - started))
    lines = []
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                lines.append(line)
    DEST.write_text('\n'.join(lines) + ('\n' if lines else ''))
    partial.unlink()
    values = [json.loads(line) for line in lines]
    report = dict(schema='beatfcn.full-database-mel-array-content-v1', readonly=True,
        cloud_compute_calls=0, output=str(DEST),
        sha256=hashlib.sha256(DEST.read_bytes()).hexdigest(),
        parsed_audio=len(parsed), records=len(values),
        verified=sum(value['verified'] for value in values), failures=failures,
        bytes=sum(value['bytes'] for value in values),
        complete=not failures and len(values) == len(parsed)
            and all(value['verified'] for value in values),
        wall_s=time.monotonic() - started,
        limitations=['Arrays are read to the local process and numerically checked; no Modal compute function runs.',
            'This verifies catalogue-parsed audio only; extra cloud mel arrays remain unjudged.',
            'Hashes cover the stored NPY bytes, not a recompute from decoded audio.'])
    write_json(RECEIPT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['complete']:
        raise SystemExit(1)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    asyncio.run(main(parser.parse_args()))


if __name__ == '__main__':
    cli()
