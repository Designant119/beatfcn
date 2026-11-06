"""Stream explicitly registered existing Volume files into a local cache."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import time

import modal

ROOT=Path(__file__).resolve().parents[1]


def check_storage(plan, additional_bytes=0):
    minimum = plan.get('minimum_free_bytes')
    if minimum is not None and shutil.disk_usage(ROOT).free < minimum + additional_bytes:
        raise ValueError('Registered local free-space reserve would be crossed')


def transfer_prior(plan, identity, out):
    if out.exists():
        prior = json.loads(out.read_text())
        if prior['plan_sha256'] != identity:
            raise ValueError('transfer plan changed')
        return prior
    link = plan.get('resume_from')
    if not link:
        return {}
    receipt_path, plan_path = (ROOT/link[k] for k in ('receipt', 'plan'))
    for path, expected in [(receipt_path, link['receipt_sha256']), (plan_path, link['plan_sha256'])]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen resume evidence changed')
    prior, parent = (json.loads(p.read_text()) for p in (receipt_path, plan_path))
    if prior['plan_sha256'] != link['plan_sha256'] or prior.get('complete') or 'ended_at_unix' not in prior:
        raise ValueError('Resume requires an incomplete, terminal receipt for the parent plan')
    for key in ('local_root', 'files', 'maximum_transfer_bytes', 'minimum_free_bytes'):
        if plan.get(key) != parent.get(key):
            raise ValueError(f'Resume cannot change {key}')
    if prior.get('cumulative_bytes', -1) < 0 or prior['cumulative_bytes'] > plan['maximum_transfer_bytes']:
        raise ValueError('Invalid inherited transfer accounting')
    return prior


async def fetch_plan(path):
    plan=json.loads(path.read_text());identity=hashlib.sha256(path.read_bytes()).hexdigest()
    out=ROOT/plan['report'];prior=transfer_prior(plan,identity,out)
    prior_hashes={r['path']:r['sha256'] for r in prior.get('files',[])}
    check_storage(plan, sum(item['bytes'] for item in plan['files']
                           if not (ROOT/plan['local_root']/item['path']).exists()))
    result=dict(plan_sha256=identity,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        complete=False,started_at_unix=time.time(),
        transferred_bytes=0,cumulative_bytes=prior.get('cumulative_bytes',0),
        files=[],failures=[],remote_compute_started=False,resume_from=plan.get('resume_from'))
    def save():
        temporary=out.with_suffix('.partial')
        temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(out)
    volume=modal.Volume.from_name('beatfcn-data',create_if_missing=False)
    semaphore=asyncio.Semaphore(plan['concurrency'])
    async def fetch(item):
        async with semaphore:
            local=ROOT/plan['local_root']/item['path']
            if not local.resolve().is_relative_to((ROOT/plan['local_root']).resolve()):
                raise ValueError('file path escapes its registered cache')
            expected=item.get('sha256') or prior_hashes.get(item['path'])
            try:
                reused=local.exists()
                if reused:
                    count=local.stat().st_size
                    hasher=hashlib.sha256()
                    with local.open('rb') as stream:
                        for chunk in iter(lambda:stream.read(8*2**20),b''): hasher.update(chunk)
                else:
                    check_storage(plan, item['bytes'])
                    local.parent.mkdir(parents=True,exist_ok=True)
                    partial=local.with_suffix('.partial');count=0;hasher=hashlib.sha256()
                    with partial.open('wb') as stream:
                        async for chunk in volume.read_file.aio(item['path']):
                            count+=len(chunk);result['transferred_bytes']+=len(chunk);result['cumulative_bytes']+=len(chunk)
                            if count>item['bytes'] or result['cumulative_bytes']>plan['maximum_transfer_bytes']:
                                raise ValueError('registered transfer byte limit exceeded')
                            check_storage(plan, len(chunk))
                            stream.write(chunk);hasher.update(chunk)
                digest=hasher.hexdigest()
                if count!=item['bytes'] or (expected and digest!=expected):
                    raise ValueError('file size or frozen SHA differs')
                if not reused: partial.replace(local)
                result['files'].append(dict(path=item['path'],local=str(local.relative_to(ROOT)),bytes=count,
                    sha256=digest,expected_hash_verified=bool(expected),reused=reused))
                save()
            except BaseException as exc:
                result['failures'].append(dict(path=item['path'],error=f'{type(exc).__name__}: {exc}'))
                save();raise
    save()
    try:
        async with asyncio.timeout(plan['timeout_s']):
            async with asyncio.TaskGroup() as group:
                for item in plan['files']: group.create_task(fetch(item))
        result['complete']=len(result['files'])==len(plan['files'])
    finally:
        result['ended_at_unix']=time.time();save()
    print(json.dumps({k:result[k] for k in ('complete','transferred_bytes','cumulative_bytes','remote_compute_started')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    asyncio.run(fetch_plan(parser.parse_args().plan))
