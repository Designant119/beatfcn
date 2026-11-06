"""Inspect exact existing feature directories; never starts remote compute."""
import asyncio
import hashlib
import json
from pathlib import Path
import time

import modal

ROOT=Path(__file__).resolve().parents[1]


async def main():
    path=ROOT/'bench/rhythm-v2/r09-positive-pairs-prepare-v1.json'
    plan=json.loads(path.read_text())
    train={r['audio_sha256'] for r in plan['train_rows']}
    validation={x['row']['audio_sha256'] for x in plan['validation_rows']}
    result=dict(plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),complete=False,
        started_at_unix=time.time(),files=[],failures=[],remote_compute_started=False)
    output=ROOT/'reports/r09_feature_inventory.json'
    def save(): output.write_text(json.dumps(result,indent=2)+'\n')
    volume=modal.Volume.from_name('beatfcn-data',create_if_missing=False)
    semaphore=asyncio.Semaphore(3)
    async def inspect(audio):
        async with semaphore:
            try:
                prefix=f'taps/v4_i8_L9-16-24/{audio}'
                entries=await volume.listdir.aio(prefix)
                named={Path(e.path).name:e for e in entries}
                for name in ('L9.npy','L16.npy','L24.npy','S20.npy'):
                    if name not in named: raise ValueError(f'missing feature {name}')
                    entry=named[name]
                    result['files'].append(dict(audio_sha256=audio,path=f'{prefix}/{name}',
                        bytes=entry.size,split='train' if audio in train else 'val'))
                save()
            except Exception as exc:
                result['failures'].append(dict(audio_sha256=audio,error=f'{type(exc).__name__}: {exc}'))
                save()
    save()
    try:
        async with asyncio.timeout(120):
            async with asyncio.TaskGroup() as group:
                for audio in sorted(train|validation): group.create_task(inspect(audio))
        result['complete']=not result['failures'] and len(result['files'])==4*len(train|validation)
    finally:
        result.update(ended_at_unix=time.time(),total_bytes=sum(r['bytes'] for r in result['files']))
        save()
    print(json.dumps({k:result[k] for k in ('complete','total_bytes','failures','remote_compute_started')}))


if __name__=='__main__': asyncio.run(main())
