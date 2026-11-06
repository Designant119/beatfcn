"""Bounded metadata-only Modal inventory; no App, Function or file transfer."""
import asyncio
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import modal

ROOT=Path(__file__).resolve().parents[1]


async def main(path):
    plan=json.loads(path.read_text())
    output=ROOT/plan['output']
    if output.exists():raise FileExistsError(output)
    candidate=ROOT/plan['candidates']
    if hashlib.sha256(candidate.read_bytes()).hexdigest()!=plan['candidates_sha256']:
        raise ValueError('Candidate identity changed')
    cohort=json.loads(candidate.read_text());rows_path=ROOT/cohort['source_rows']
    if hashlib.sha256(rows_path.read_bytes()).hexdigest()!=cohort['source_rows_sha256']:
        raise ValueError('Source rows changed')
    rows={r['chart_id']:r for r in map(json.loads,rows_path.read_text().splitlines())}
    jobs=[]
    for song in cohort['records']:
        audio=song['audio_sha256']
        jobs.append(dict(kind='taps',path=f'taps/v4_i8_L9-16-24/{audio}',audio_sha256=audio,
                         required=['L9.npy','L16.npy','L24.npy']))
        jobs.append(dict(kind='mel',path=f'mel10/{audio}.npy',audio_sha256=audio,required=[audio+'.npy']))
    for cid,row in rows.items():
        jobs.append(dict(kind='canonical',chart_id=cid,path=f"canon/{row['source']}/{row['native_id']}.json",
                         required=[row['native_id']+'.json']))
        if row['source']=='phira':
            jobs.append(dict(kind='native_package',chart_id=cid,path=f"raw/phira/zips/{row['native_id']}.zip",
                             required=[row['native_id']+'.zip']))
    if len(jobs)>plan['maximum_api_calls']:raise ValueError('Inventory request bound exceeded')
    volume=modal.Volume.from_name('beatfcn-data',create_if_missing=False)
    started=time.monotonic();sem=asyncio.Semaphore(plan['concurrency']);stop=asyncio.Event()
    report=dict(complete=False,plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_cloud_compute_calls=0,payload_files_downloaded=0,expected_jobs=len(jobs),records=[],failures=[],skipped=0)
    def save():
        report['wall_s']=time.monotonic()-started
        temp=output.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(output)
    async def inspect(job):
        async with sem:
            if stop.is_set():report['skipped']+=1;return
            try:
                entries=await asyncio.wait_for(volume.listdir.aio(job['path']),plan['request_timeout_s'])
                names={Path(e.path).name:e for e in entries}
                missing=[name for name in job['required'] if name not in names]
                files=[dict(path=e.path,bytes=e.size,mtime=e.mtime) for name,e in names.items() if name in job['required']]
                report['records'].append(dict(**job,missing=missing,files=files,available=not missing,
                                             bytes=sum(e['bytes'] for e in files)))
            except Exception as exc:
                category=type(exc).__name__
                report['failures'].append(dict(**job,error_type=category,error=str(exc)[:240]))
                if category not in ('NotFoundError',) and sum(r['error_type']!='NotFoundError' for r in report['failures'])>=2:
                    stop.set()
            if (len(report['records'])+len(report['failures']))%50==0:save()
    save()
    try:
        async with asyncio.timeout(plan['max_wall_s']):
            await asyncio.gather(*(inspect(job) for job in jobs))
    except TimeoutError:
        report['timed_out']=True
    finally:
        report['complete']=(len(report['records'])+len(report['failures'])==len(jobs))
        report['available_by_kind']=dict(Counter(r['kind'] for r in report['records'] if r['available']))
        report['bytes_by_kind']={kind:sum(r['bytes'] for r in report['records'] if r['kind']==kind)
                                 for kind in ('taps','mel','canonical','native_package')}
        report['all_required_files_available']=report['complete'] and not report['failures'] and all(r['available'] for r in report['records'])
        report['array_headers_checked']=False;report['native_content_checked']=False
        save()
    print(json.dumps({k:report[k] for k in ['complete','expected_jobs','available_by_kind','bytes_by_kind','all_required_files_available','wall_s','skipped']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    asyncio.run(main(parser.parse_args().plan.resolve()))
