"""Materialize every parseable cloud chart into lossless target batches.

Reads canonical JSON from the Modal volume, runs the exact Owner drag rule and
model shift, then uploads per-batch immutable target folders and markers. This
driver performs no optimizer update and calls no Modal compute function.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import modal

from beatfcn.train.full_database import prepare_chart


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'inputs/full-database-cloud-v1/manifest/charts.offset-v2-3d99ff6bd5e7.jsonl'
DEST = ROOT / 'reports/full-database-cloud-v1/target-preparation-v1.json'
REMOTE_ROOT = 'full-database-v1'
BATCH_SIZE = 256
CONCURRENCY = 128


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


async def read_remote(volume, remote, *, attempts=5):
    for attempt in range(1, attempts + 1):
        try:
            payload = b''.join([chunk async for chunk in volume.read_file.aio(remote)])
            return payload
        except Exception:
            if attempt == attempts:
                raise
            await asyncio.sleep(min(2 ** attempt, 12))


async def existing_markers(volume):
    try:
        entries = await volume.listdir.aio(f'{REMOTE_ROOT}/batches')
    except modal.exception.NotFoundError:
        return {}
    markers = {}
    for entry in entries:
        if not entry.path.endswith('.json'):
            continue
        payload = await read_remote(volume, entry.path)
        value = json.loads(payload)
        markers[value['batch']] = value
    return markers


async def process_batch(volume, rows, batch, manifest_sha, work):
    stage = work / f'batch-{batch:04d}'
    targets = stage / 'targets'
    source_root = work / 'source'
    stage.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    successes = []
    permanent_failures = []

    async def prepare(row):
        async with semaphore:
            remote = f"canon/{'osu' if row['source'] == 'osu' else 'phira'}/{row['native_id']}.json"
            payload = await read_remote(volume, remote)
            local = source_root / f"{row['source']}-{row['native_id']}.json"
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(payload)
            try:
                result = prepare_chart(row, local, targets)
                successes.append(dict(chart_id=row['chart_id'], target_identity=Path(result['target_path']).parent.name,
                    canonical_sha256=result['canonical_sha256'],
                    statistics=result['statistics']))
            except Exception as exc:
                permanent_failures.append(dict(chart_id=row['chart_id'],
                    error=f'{type(exc).__name__}: {exc}'))
            finally:
                local.unlink(missing_ok=True)

    started = time.monotonic()
    await asyncio.gather(*(prepare(row) for row in rows))
    marker = dict(schema='beatfcn.full-database-target-batch.v1', batch=batch,
        manifest_sha256=manifest_sha, planned=[row['chart_id'] for row in rows],
        successes=successes, permanent_failures=permanent_failures,
        success_count=len(successes), permanent_failure_count=len(permanent_failures),
        wall_s=time.monotonic() - started, cloud_compute_calls=0)
    marker_path = stage / 'batches' / f'{batch:04d}.json'
    write_json(marker_path, marker)
    async with volume.batch_upload.aio() as upload:
        upload.put_directory(str(stage), REMOTE_ROOT)
    shutil.rmtree(stage)
    return marker


async def main(args):
    manifest_payload = MANIFEST.read_bytes()
    manifest_sha = digest_bytes(manifest_payload)
    report = None
    if DEST.exists():
        report = json.loads(DEST.read_text())
        if report.get('complete'):
            raise FileExistsError('Target preparation already completed')
        if report.get('manifest_sha256') != manifest_sha:
            raise ValueError('Incomplete target report belongs to another manifest')
    all_rows = [json.loads(line) for line in manifest_payload.decode().splitlines() if line.strip()]
    rows = [row for row in all_rows if row.get('status') == 'parsed']
    batches = [rows[index:index + BATCH_SIZE] for index in range(0, len(rows), BATCH_SIZE)]
    volume = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
    markers = await existing_markers(volume)
    for batch, marker in markers.items():
        if marker.get('manifest_sha256') != manifest_sha:
            raise ValueError(f'Remote target batch {batch} belongs to another manifest')
    completed = set(markers)
    started = time.monotonic()
    if report is None:
        report = dict(schema='beatfcn.full-database-target-preparation.v1',
            manifest=str(MANIFEST.relative_to(ROOT)), manifest_sha256=manifest_sha,
            parsed_rows=len(rows), planned_batches=len(batches), batch_size=BATCH_SIZE,
            completed_batches=sorted(completed), markers={}, cloud_compute_calls=0,
            optimizer_updates=0, volume_storage_writes=True, complete=False)
    else:
        completed.update(report.get('completed_batches', []))
        report['markers'].update(markers)
        report['parsed_rows'] = len(rows)
        report['planned_batches'] = len(batches)
        report['batch_size'] = BATCH_SIZE
    try:
        with tempfile.TemporaryDirectory(prefix='beatfcn-full-targets-') as temporary:
            work = Path(temporary)
            for batch, rows_in_batch in enumerate(batches):
                if batch in completed:
                    report['markers'][str(batch)] = markers[batch]
                    continue
                marker = await process_batch(volume, rows_in_batch, batch, manifest_sha, work)
                markers[batch] = marker
                completed.add(batch)
                report['markers'][str(batch)] = marker
                report['completed_batches'] = sorted(completed)
                report['complete'] = len(completed) == len(batches)
                report['wall_s'] = time.monotonic() - started
                write_json(DEST, report)
                print(json.dumps(dict(batch=batch, successes=marker['success_count'],
                    failures=marker['permanent_failure_count'], wall_s=marker['wall_s'])), flush=True)
    finally:
        report['completed_batches'] = sorted(completed)
        report['complete'] = len(completed) == len(batches)
        report['wall_s'] = time.monotonic() - started
        report['success_count'] = sum(marker['success_count'] for marker in markers.values())
        report['permanent_failure_count'] = sum(marker['permanent_failure_count'] for marker in markers.values())
        write_json(DEST, report)
    print(json.dumps({key: value for key, value in report.items() if key != 'markers'},
        ensure_ascii=False, indent=2))


def cli():
    parser = argparse.ArgumentParser()
    asyncio.run(main(parser.parse_args()))


if __name__ == '__main__':
    cli()
