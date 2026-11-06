"""Read-only content-hash verification for parsed-catalogue raw audio archives."""
import argparse
import asyncio
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import modal


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'inputs/full-database-cloud-v1/manifest/charts.offset-v2-3d99ff6bd5e7.jsonl'
INVENTORY = ROOT / 'reports/full-database-cloud-v1'
DEST = ROOT / 'inputs/full-database-cloud-v1/raw-content.jsonl'
RECEIPT = ROOT / 'reports/full-database-cloud-v1/raw-content-receipt.json'
AUDIO_SUFFIXES = {'.ogg', '.mp3', '.wav', '.flac', '.m4a', '.aac', '.opus'}
MAX_ARCHIVE_BYTES = 512 * 2**20


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def load_inventory(name):
    value = json.loads((INVENTORY / name).read_text())
    records = value if isinstance(value, list) else value['records']
    if not (isinstance(value, list) or value.get('complete')):
        raise ValueError(f'Incomplete inventory: {name}')
    return {Path(record['path']).name: record for record in records}


async def read_stream(volume, remote, *, attempts=4):
    for attempt in range(1, attempts + 1):
        try:
            hasher = hashlib.sha256()
            size = 0
            chunks = []
            async for chunk in volume.read_file.aio(remote):
                hasher.update(chunk)
                size += len(chunk)
                chunks.append(chunk)
            return hasher.hexdigest(), size, chunks
        except Exception:
            if attempt == attempts:
                raise
            await asyncio.sleep(min(2 ** attempt, 10))


async def verify_osu(volume, audio, row, size_by_name, semaphore):
    async with semaphore:
        ext = row['audio_ext']
    name = f'{audio}.{ext}'
    inventory = size_by_name.get(name)
    if inventory is None:
        raise FileNotFoundError('OSU raw archive missing: ' + name)
    digest_value, size, _ = await read_stream(volume, 'raw/osu/audio/' + name)
    return [dict(audio_sha256=audio, source='osu', archive='raw/osu/audio/' + name,
        archive_member=None, archive_bytes=inventory['bytes'], payload_bytes=size,
        payload_sha256=digest_value, verified=digest_value == audio and size == inventory['bytes'])]


async def verify_phira(volume, native, rows, size_by_name, semaphore):
    async with semaphore:
        name = native + '.zip'
        inventory = size_by_name.get(name)
        if inventory is None:
            raise FileNotFoundError('Phira raw archive missing: ' + name)
        _, size, chunks = await read_stream(volume, 'raw/phira/zips/' + name)
        if size != inventory['bytes'] or size > MAX_ARCHIVE_BYTES:
            raise ValueError('Phira archive size differs: ' + name)
        payload = b''.join(chunks)
        del chunks
        expected = {row['audio_sha256'] for row in rows}
        found = {}
        with zipfile.ZipFile(io.BytesIO(payload), 'r') as archive:
            for member in archive.infolist():
                if member.is_dir() or Path(member.filename).suffix.lower() not in AUDIO_SUFFIXES:
                    continue
                hasher = hashlib.sha256()
                member_size = 0
                with archive.open(member) as stream:
                    while True:
                        chunk = stream.read(2**20)
                        if not chunk:
                            break
                        hasher.update(chunk)
                        member_size += len(chunk)
                digest_value = hasher.hexdigest()
                if digest_value in expected:
                    found[digest_value] = (member.filename, member_size)
        result = []
        for audio in sorted(expected):
            member, member_size = found.get(audio, (None, None))
            result.append(dict(audio_sha256=audio, source='phira',
                archive='raw/phira/zips/' + name, archive_member=member,
                archive_bytes=inventory['bytes'], payload_bytes=member_size,
                payload_sha256=audio if member is not None else None,
                verified=member is not None and member_size == archive.getinfo(member).file_size if member is not None else False))
        return result


async def main(args):
    if DEST.exists():
        raise FileExistsError('Raw content output already exists')
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get('status') == 'parsed']
    osu_jobs = {}
    phira_jobs = defaultdict(list)
    for row in rows:
        if row['source'] == 'osu':
            osu_jobs[row['audio_sha256']] = row
        elif row['source'] == 'phira':
            phira_jobs[row['native_id']].append(row)
        else:
            raise ValueError('Unknown source ' + row['source'])
    osu_size = load_inventory('raw-osu-audio.json')
    phira_size = load_inventory('raw-phira-zips.json')
    volume = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
    partial = DEST.with_suffix('.jsonl.partial')
    done = set()
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                value = json.loads(line)
                if value.get('verified'):
                    done.add(value['audio_sha256'])
    osu_jobs = {key: value for key, value in osu_jobs.items() if key not in done}
    phira_jobs = {key: values for key, values in phira_jobs.items()
                  if not all(row['audio_sha256'] in done for row in values)}
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.monotonic()
    failures = []
    records = []
    with partial.open('a') as stream:
        async def run_osu(audio, row):
            try:
                values = await verify_osu(volume, audio, row, osu_size, semaphore)
                stream.write('\n'.join(json.dumps(value, ensure_ascii=False) for value in values) + '\n')
                stream.flush()
                records.extend(values)
            except Exception as exc:
                failures.append(dict(audio_sha256=audio,
                    error=f'{type(exc).__name__}: {exc}'))
        async def run_phira(native, values):
            try:
                result = await verify_phira(volume, native, values, phira_size, semaphore)
                stream.write('\n'.join(json.dumps(value, ensure_ascii=False) for value in result) + '\n')
                stream.flush()
                records.extend(result)
            except Exception as exc:
                failures.append(dict(native_id=native,
                    error=f'{type(exc).__name__}: {exc}'))
        osu_tasks = [run_osu(audio, row) for audio, row in osu_jobs.items()]
        phira_tasks = [run_phira(native, values) for native, values in phira_jobs.items()]
        results = await asyncio.gather(*osu_tasks, *phira_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                failures.append(dict(unexpected_task_error=f'{type(result).__name__}: {result}'))
    lines = []
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                lines.append(line)
    DEST.write_text('\n'.join(lines) + ('\n' if lines else ''))
    partial.unlink()
    values = [json.loads(line) for line in lines]
    report = dict(schema='beatfcn.full-database-raw-content-v1', readonly=True,
        cloud_compute_calls=0, output=str(DEST),
        sha256=hashlib.sha256(DEST.read_bytes()).hexdigest(),
        parsed_rows=len(rows), records=len(values),
        verified=sum(value['verified'] for value in values),
        osu_audio=len(osu_jobs), phira_archives=len(phira_jobs),
        failures=failures, complete=not failures and all(value['verified'] for value in values),
        wall_s=time.monotonic() - started,
        limitations=['Reads cloud storage to the local process; no Modal compute function is called.',
            'Phira package archive is fully downloaded into process memory and audio members are content-hashed.',
            'Catalogue rows outside the parsed scope are not verified in this pass.'])
    write_json(RECEIPT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['complete']:
        raise SystemExit(1)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--concurrency', type=int, default=4)
    asyncio.run(main(parser.parse_args()))


if __name__ == '__main__':
    cli()
