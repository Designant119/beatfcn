"""Read-only Modal volume inventory and full-database manifest retrieval."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import modal


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'reports/full-database-cloud-v1'
MANIFEST_DEST = ROOT / 'inputs/full-database-cloud-v1/manifest'
FOLDERS = {
    'canon-osu': 'canon/osu',
    'canon-phira': 'canon/phira',
    'mel10': 'mel10',
    'manifest': 'manifest',
    'raw-osu-audio': 'raw/osu/audio',
    'raw-phira-zips': 'raw/phira/zips',
}
MANIFEST_FILES = (
    'manifest/charts.jsonl',
    'manifest/charts.offset-v2-3d99ff6bd5e7.jsonl',
    'manifest/phira_candidates.jsonl',
    'manifest/stats.json',
)


def digest(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def safe_remote(path):
    value = Path(path)
    if value.is_absolute() or '..' in value.parts:
        raise ValueError('Unsafe remote path')
    return value.as_posix()


async def list_folder(volume, folder, *, timeout_s=90, attempts=5):
    last = None
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        try:
            entries = await asyncio.wait_for(volume.listdir.aio(folder), timeout_s)
            records = [dict(path=entry.path, bytes=entry.size, mtime=entry.mtime)
                       for entry in entries if not entry.path.endswith('/')]
            records.sort(key=lambda value: value['path'])
            return dict(folder=folder, complete=True, files=len(records),
                bytes=sum(value['bytes'] for value in records), attempts=attempt,
                wall_s=time.monotonic() - started, records=records)
        except Exception as exc:
            last = f'{type(exc).__name__}: {exc}'
            if attempt < attempts:
                await asyncio.sleep(min(2 ** attempt, 15))
    return dict(folder=folder, complete=False, attempts=attempts,
        wall_s=time.monotonic() - started, error=last, records=[])


async def inventory(only_missing):
    volume = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
    summary = dict(schema='beatfcn.full-database-cloud-inventory-v2', readonly=True,
        cloud_compute_calls=0, records=[])
    for name, folder in FOLDERS.items():
        output = DEST / f'{name}.json'
        if only_missing and output.exists():
            previous = json.loads(output.read_text())
            # The first raw inventories are plain arrays; keep those immutable.
            if isinstance(previous, list):
                summary['records'].append(dict(name=name, folder=folder, output=str(output),
                    reused=True, complete=True, files=len(previous),
                    bytes=sum(item.get('bytes', 0) for item in previous), legacy_format=True))
                continue
            if previous.get('complete'):
                summary['records'].append(dict(name=name, folder=folder, output=str(output),
                    reused=True, complete=True, files=previous.get('files'), bytes=previous.get('bytes')))
                continue
        result = await list_folder(volume, folder)
        result.update(name=name, output=str(output), cloud_compute_calls=0,
            code_sha256=digest(Path(__file__)))
        write_json(output, result)
        summary['records'].append(dict(name=name, folder=folder, output=str(output),
            complete=result['complete'], files=result.get('files'), bytes=result.get('bytes'),
            error=result.get('error')))
    summary['complete'] = all(item['complete'] for item in summary['records'])
    write_json(DEST / 'inventory-v2.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary['complete']:
        raise SystemExit(1)


async def download_manifest(volume, remote, output, *, chunk_size=8 << 20):
    remote = safe_remote(remote)
    output = Path(output)
    if output.exists():
        expected = output.with_suffix(output.suffix + '.sha256')
        if not expected.is_file() or expected.read_text().strip() != digest(output):
            raise ValueError(f'Existing manifest identity mismatch: {output}')
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.partial')
    hasher = hashlib.sha256()
    started = time.monotonic()
    size = 0
    with temporary.open('wb') as target:
        async for chunk in volume.read_file.aio(remote):
            target.write(chunk)
            hasher.update(chunk)
            size += len(chunk)
    temporary.replace(output)
    output.with_suffix(output.suffix + '.sha256').write_text(hasher.hexdigest() + '\n')
    write_json(output.with_suffix('.receipt.json'), dict(remote=remote, output=str(output),
        bytes=size, sha256=hasher.hexdigest(), wall_s=time.monotonic() - started,
        readonly=True, cloud_compute_calls=0))
    return True


async def manifests(only_missing):
    volume = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
    result = dict(schema='beatfcn.full-database-manifest-download-v1', readonly=True,
        cloud_compute_calls=0, files=[])
    for remote in MANIFEST_FILES:
        name = Path(remote).name
        output = MANIFEST_DEST / name
        downloaded = await download_manifest(volume, remote, output)
        result['files'].append(dict(remote=remote, output=str(output), downloaded=downloaded,
            bytes=output.stat().st_size, sha256=digest(output)))
    result['complete'] = len(result['files']) == len(MANIFEST_FILES)
    write_json(MANIFEST_DEST / 'download-receipt.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def audit_manifests():
    charts = MANIFEST_DEST / 'charts.jsonl'
    if not charts.is_file():
        raise FileNotFoundError(charts)
    rows = []
    for number, line in enumerate(charts.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception as exc:
            raise ValueError(f'Invalid charts.jsonl line {number}: {exc}') from exc
    audio_missing = [row for row in rows if not row.get('audio_sha256')]
    split_counts = Counter(row.get('split') for row in rows)
    mode_counts = Counter(row.get('mode') for row in rows)
    source_counts = Counter(row.get('source') for row in rows)
    status_counts = Counter(row.get('status') for row in rows)
    kept_counts = Counter(bool(row.get('kept')) for row in rows)
    result = dict(schema='beatfcn.full-database-manifest-audit-v1',
        source=str(charts), sha256=digest(charts), rows=len(rows),
        modes=dict(mode_counts), sources=dict(source_counts), splits=dict(split_counts),
        statuses=dict(status_counts), kept=dict(kept_counts),
        rows_without_audio_sha256=len(audio_missing),
        unique_chart_ids=len({row.get('chart_id') for row in rows}),
        unique_audio_sha256=len({row.get('audio_sha256') for row in rows if row.get('audio_sha256')}),
        unique_song_groups=len({row.get('song_group') for row in rows if row.get('song_group')}),
        field_frequency=dict(Counter(key for row in rows for key in row)),
        cloud_compute_calls=0,
        limitations=['Catalogue audit only; metadata presence is not audio/chart/feature completeness.',
            'No metadata admission, clock verification, playable-content audit or split reassignment was performed.'])
    write_json(DEST / 'manifest-audit-v1.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'field_frequency'},
        ensure_ascii=False, indent=2))


async def async_main(args):
    if args.action in ('inventory', 'all'):
        await inventory(args.only_missing)
    if args.action in ('manifests', 'all'):
        await manifests(args.only_missing)
    if args.action in ('audit', 'all'):
        audit_manifests()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('inventory', 'manifests', 'audit', 'all'))
    parser.add_argument('--only-missing', action='store_true')
    asyncio.run(async_main(parser.parse_args()))


if __name__ == '__main__':
    main()
