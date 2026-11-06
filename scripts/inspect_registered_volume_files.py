"""Bounded read-only file inventory for an explicit, frozen preparation plan."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import modal

ROOT = Path(__file__).resolve().parents[1]


async def run(path):
    plan = json.loads(path.read_text())
    output = ROOT/plan['output']
    if output.exists():
        raise FileExistsError(output)
    jobs = plan['files']
    if (len(jobs) > plan['max_calls'] or len({j['path'] for j in jobs}) != len(jobs)
            or any(Path(j['path']).is_absolute() or '..' in Path(j['path']).parts for j in jobs)):
        raise ValueError('Invalid or excessive registered file paths')
    for name, expected in plan['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen inventory source changed')
    volume = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
    sem = asyncio.Semaphore(plan['concurrency'])
    stop = asyncio.Event()
    started = time.monotonic()
    report = dict(complete=False, records=[], failures=[], skipped=0, expected_files=len(jobs),
                  plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  payload_bytes=0, cloud_compute_calls=0)

    def save():
        report['wall_s'] = time.monotonic()-started
        temporary = output.with_suffix('.partial')
        temporary.write_text(json.dumps(report, indent=2)+'\n'); temporary.replace(output)

    async def inspect(job):
        async with sem:
            if stop.is_set():
                report['skipped'] += 1
                return
            try:
                entries = await asyncio.wait_for(volume.listdir.aio(job['path']), plan['request_timeout_s'])
                matches = [e for e in entries if e.path.lstrip('/') == job['path']]
                if len(matches) != 1:
                    raise ValueError('Inventory did not resolve exactly the registered file')
                entry = matches[0]
                report['records'].append(dict(**job, bytes=entry.size, mtime=entry.mtime))
            except Exception as exc:
                kind = type(exc).__name__
                report['failures'].append(dict(**job, error_type=kind, error=str(exc)[:250]))
                if kind != 'NotFoundError' and sum(r['error_type'] != 'NotFoundError' for r in report['failures']) >= 2:
                    stop.set()
            if (len(report['records'])+len(report['failures'])) % 25 == 0:
                save()
    save()
    try:
        async with asyncio.timeout(plan['timeout_s']):
            await asyncio.gather(*(inspect(job) for job in jobs))
    except TimeoutError:
        report['timed_out'] = True
    finally:
        report['complete'] = len(report['records'])+len(report['failures']) == len(jobs)
        report['found_by_kind'] = dict(Counter(r['kind'] for r in report['records']))
        report['bytes_by_kind'] = {kind: sum(r['bytes'] for r in report['records'] if r['kind'] == kind)
                                   for kind in report['found_by_kind']}
        save()
    print(json.dumps({k: v for k, v in report.items() if k not in ('records', 'failures')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    asyncio.run(run(parser.parse_args().plan.resolve()))
