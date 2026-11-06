"""Bounded local download of fixed probe inputs; no Modal App/Function creation."""
import asyncio
import hashlib
import json
from pathlib import Path
import time

import modal

ROOT = Path(__file__).resolve().parents[1]


async def main():
    plan_path = ROOT / 'bench/rhythm-v2/r08-local-real-probe-inputs-v1.json'
    plan = json.loads(plan_path.read_text())
    volume = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
    semaphore = asyncio.Semaphore(2)
    result = dict(schema='beatfcn.local_probe_transfer.v1', complete=False, started_at_unix=time.time(),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(), transferred_bytes=0,
        files=[], remote_compute_started=False, checkpoint_inference_started=False)
    output = ROOT / 'reports/r08_local_probe_transfer.json'
    def record():
        output.write_text(json.dumps(result, indent=2)+'\n')
    async def fetch(item):
        async with semaphore:
            path = ROOT / plan['local_root'] / item['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            existing = path.exists()
            if existing:
                content = path.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                if len(content) != item['bytes'] or (item.get('sha256') and digest != item['sha256']):
                    raise ValueError(f'cached input identity differs: {item["path"]}')
                del content
            else:
                partial = path.with_suffix(path.suffix+'.partial')
                count, hasher = 0, hashlib.sha256()
                with partial.open('wb') as stream:
                    async for chunk in volume.read_file.aio(item['path']):
                        count += len(chunk)
                        result['transferred_bytes'] += len(chunk)
                        if count > item['bytes'] or result['transferred_bytes'] > plan['maximum_transfer_bytes']:
                            raise ValueError('fixed transfer byte limit exceeded')
                        stream.write(chunk)
                        hasher.update(chunk)
                digest = hasher.hexdigest()
                if count != item['bytes'] or (item.get('sha256') and digest != item['sha256']):
                    raise ValueError(f'downloaded input identity differs: {item["path"]}')
                partial.replace(path)
            result['files'].append(dict(path=item['path'], local=str(path.relative_to(ROOT)),
                bytes=item['bytes'], sha256=digest, expected_hash_verified=bool(item.get('sha256')),
                reused_cache=existing))
            record()
    record()
    try:
        await asyncio.wait_for(asyncio.gather(*(fetch(f) for f in plan['files'])), timeout=plan['timeout_s'])
        result['complete'] = len(result['files']) == len(plan['files'])
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['ended_at_unix'] = time.time()
        record()
    print(json.dumps(dict(complete=result['complete'], files=len(result['files']),
        transferred_bytes=result['transferred_bytes'], remote_compute_started=False)))


if __name__ == '__main__':
    asyncio.run(main())
