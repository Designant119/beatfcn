"""Read the frozen prospective validation canons; no cloud function or training."""
import asyncio
import hashlib
import json
from pathlib import Path
import time

import modal

ROOT = Path(__file__).resolve().parents[1]


async def main():
    path = ROOT/'bench/rhythm-v2/r09-positive-pairs-prepare-v1.json'
    plan = json.loads(path.read_text())
    limits = plan['canonical_transfer']
    report_path = ROOT/'reports/r09_validation_canon_transfer.json'
    prior = json.loads(report_path.read_text()) if report_path.exists() else {}
    identity = hashlib.sha256(path.read_bytes()).hexdigest()
    if prior and prior['plan_sha256']!=identity: raise ValueError('previous transfer plan differs')
    result = dict(plan_sha256=identity,complete=False,started_at_unix=time.time(),
        transferred_bytes=0,cumulative_bytes=prior.get('cumulative_bytes',0),
        files=[],failures=[],remote_compute_started=False)
    def save(): report_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    volume = modal.Volume.from_name('beatfcn-data',create_if_missing=False)
    semaphore = asyncio.Semaphore(limits['concurrency'])
    async def fetch(item):
        row = item['row']
        async with semaphore:
            local = ROOT/plan['validation_canonical_root']/row['source']/f'{row["native_id"]}.json'
            try:
                reused = local.exists()
                if reused:
                    raw = local.read_bytes()
                else:
                    data = bytearray()
                    async for chunk in volume.read_file.aio(f'canon/{row["source"]}/{row["native_id"]}.json'):
                        result['transferred_bytes'] += len(chunk)
                        result['cumulative_bytes'] += len(chunk)
                        if result['cumulative_bytes']>limits['maximum_bytes'] or len(data)+len(chunk)>limits['per_file_maximum_bytes']:
                            raise ValueError('frozen transfer bound exceeded')
                        data.extend(chunk)
                    raw = bytes(data)
                doc = json.loads(raw)
                if any(doc[k]!=row[k] for k in ('chart_id','mode','audio_sha256')):
                    raise ValueError('canonical identity differs from frozen metadata')
                local.parent.mkdir(parents=True,exist_ok=True)
                temporary = local.with_suffix('.partial')
                temporary.write_bytes(raw);temporary.replace(local)
                result['files'].append(dict(chart_id=row['chart_id'],local=str(local.relative_to(ROOT)),
                    bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),reused=reused))
                save()
            except BaseException as exc:
                result['failures'].append(dict(chart_id=row['chart_id'],error=f'{type(exc).__name__}: {exc}'))
                save()
                raise
    save()
    try:
        # TaskGroup cancels siblings before returning on any failed read.
        async with asyncio.timeout(limits['timeout_s']):
            async with asyncio.TaskGroup() as group:
                for item in plan['validation_rows']: group.create_task(fetch(item))
        result['complete'] = len(result['files'])==len(plan['validation_rows'])
    finally:
        result['ended_at_unix'] = time.time()
        save()
    print(json.dumps({k:result[k] for k in ('complete','transferred_bytes','cumulative_bytes','remote_compute_started')}))


if __name__=='__main__': asyncio.run(main())
