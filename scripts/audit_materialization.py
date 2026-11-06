"""D12 read-only inventory of the actual complete training/validation working set."""
import asyncio
import hashlib
import json
from pathlib import Path
import time

import modal
import torch

ROOT = Path(__file__).resolve().parents[1]


async def main():
    plan_path = ROOT / 'bench/rhythm-v2/d12-materialization-audit-v1.json'
    plan = json.loads(plan_path.read_text())
    out = ROOT / 'inputs/d12_materialization'
    out.mkdir(parents=True, exist_ok=True)
    volume = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
    path = ROOT / 'reports/d12_materialization.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    report = dict(schema='beatfcn.materialization_inventory.v1', complete=False, remote_compute=False,
                  started_at_unix=time.time(), plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                  transferred_bytes=0, files=[], songs={}, failures=[])
    report['previous_transferred_bytes'] = previous.get('previous_transferred_bytes', 0) + previous.get('transferred_bytes', 0)

    def save():
        path.write_text(json.dumps(report, indent=2) + '\n')

    async def fetch(remote, filename):
        local = out / filename
        receipt = next((r for r in previous.get('files', []) if r['remote'] == remote), None)
        if receipt and local.exists():
            raw = local.read_bytes()
            if len(raw) != receipt['bytes'] or hashlib.sha256(raw).hexdigest() != receipt['sha256']:
                raise ValueError('local read cache changed')
            report['files'].append(dict(receipt, reused_verified_cache=True))
            return local
        content = bytearray()
        async for chunk in volume.read_file.aio(remote):
            report['transferred_bytes'] += len(chunk)
            if report['transferred_bytes'] > plan['maximum_download_bytes']:
                raise ValueError('read limit exceeded')
            content.extend(chunk)
        local.write_bytes(content)
        report['files'].append(dict(remote=remote, local=str(local.relative_to(ROOT)), bytes=len(content),
                                    sha256=hashlib.sha256(content).hexdigest()))
        save()
        return local

    async def work():
        samples, prep = await asyncio.gather(fetch(plan['samples_remote'], 'samples.jsonl'),
                                             fetch(plan['prep_remote'], 'prep.pt'))
        records = [json.loads(line) for line in samples.read_text().splitlines()]
        if len(records) != plan['expected_batches'] or any(len(b) != plan['micro_batch'] for b in records):
            raise ValueError('actual batch count differs')
        prefix = [json.loads(line) for line in (ROOT / 'inputs/r07_actual_samples.jsonl').read_text().splitlines()]
        # R07 adds one provenance flag for its registered input intervention.
        normalized = []
        for batch in prefix:
            clean = []
            for row in batch:
                row = dict(row)
                if 'external_chart_context_removed' in row:
                    if row.pop('external_chart_context_removed') is not True:
                        raise ValueError('unexpected R07 provenance flag')
                clean.append(row)
            normalized.append(clean)
        if records[:len(prefix)] != normalized:
            raise ValueError('R07 and R03 actual consumed sequence differ')
        train_keys = {r['song'] for batch in records for r in batch}
        if len(train_keys) != plan['expected_train_songs']:
            raise ValueError('unique training songs differ')
        z = torch.load(prep, map_location='cpu', weights_only=False)
        manifest = json.loads((ROOT / 'reports/r03_window_raw/manifest.json').read_text())
        if z['info']['rows_sha256'] != manifest['rows_sha256'] or len(z['val']) != 40:
            raise ValueError('validation prep identity differs')
        val_keys = {s for _, s in z['val']}
        if train_keys & val_keys:
            raise ValueError('training and validation audio overlap')
        report.update(training_songs=len(train_keys), validation_songs=len(val_keys),
                      actual_batches=len(records), matched_r07_batches=len(prefix), validation_cores=40)
        semaphore = asyncio.Semaphore(plan['metadata_concurrency'])

        async def inventory(sha):
            async with semaphore:
                names = {'L9.npy', 'L16.npy', 'L24.npy', 'S20.npy'}
                entries = [e async for e in volume.iterdir.aio(f'taps/v4_i8_L9-16-24/{sha}', recursive=False)]
                selected = {Path(e.path).name: dict(path=e.path, bytes=e.size) for e in entries if Path(e.path).name in names}
                if set(selected) != names:
                    raise ValueError(f'feature files missing for {sha}')
                report['songs'][sha] = dict(split='train' if sha in train_keys else 'val', files=selected,
                                             bytes=sum(x['bytes'] for x in selected.values()))
                if len(report['songs']) % 50 == 0:
                    save()
                    print(json.dumps(dict(event='inventory', songs=len(report['songs']))), flush=True)
        await asyncio.gather(*(inventory(sha) for sha in sorted(train_keys | val_keys)))
        total = sum(x['bytes'] for x in report['songs'].values())
        if total > plan['maximum_feature_bytes']:
            raise ValueError('working set above registered materialization limit')
        report.update(complete=True, feature_bytes=total, feature_gib=total/2**30,
                      maximum_song_bytes=max(x['bytes'] for x in report['songs'].values()),
                      validation_feature_bytes=sum(x['bytes'] for x in report['songs'].values() if x['split']=='val'))
    try:
        await asyncio.wait_for(work(), timeout=plan['overall_timeout_s'])
    except BaseException as exc:
        report['failures'].append(f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['ended_at_unix'] = time.time()
        save()
        print(json.dumps({k: report.get(k) for k in ('complete', 'transferred_bytes', 'training_songs', 'validation_songs', 'feature_gib', 'failures')}))


if __name__ == '__main__':
    asyncio.run(main())
