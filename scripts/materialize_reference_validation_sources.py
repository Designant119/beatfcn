"""Retrieve four already-catalogued validation sources from public URLs, without cloud compute."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

from scripts.run_local_song_baseline import write_json

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    pp=ROOT/'bench/rhythm-v2/reference-validation-sources-v1.json'
    plan=json.loads(pp.read_text());output=ROOT/plan['report']
    if output.exists():raise FileExistsError('Existing receipt; inspect rather than retry automatically')
    for name,digest in plan['inputs_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen input changed: '+name)
    root=(ROOT/plan['output_root']).resolve()
    if not root.is_relative_to(ROOT) or root==ROOT:raise ValueError('Dedicated workspace output required')
    root.mkdir(parents=True,exist_ok=True)
    existing={r['chart_id']:r for r in map(json.loads,(ROOT/plan['corpus']).read_text().splitlines())}
    ids=[i['id']for i in plan['items']]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate selected source')
    for item in plan['items']:
        row=existing['phira:'+str(item['id'])]
        if row['split']!='val' or row['audio_sha256']!=item['audio_sha256']:
            raise ValueError('Expected existing validation identities only')
        if not item['url'].startswith('https://phira.5wyxi.com/files/'):
            raise ValueError('Unexpected public source host')
    started=time.monotonic();transferred=0;failures=0
    report=dict(id=plan['id'],complete=False,plan_sha256=sha(pp),records=[],optimizer_updates=0,
                cloud_compute_started=False,new_cloud_spend_usd=0,training_ready=False)
    def save():
        report.update(wall_s=time.monotonic()-started,transferred_bytes=transferred)
        write_json(output,report)
    save()
    for item in plan['items']:
        row=dict(id=item['id'],status='not_attempted',bytes=0)
        remaining=plan['max_wall_s']-(time.monotonic()-started)
        allowance=min(plan['per_file_bytes'],plan['total_bytes']-transferred)
        if remaining<=0 or allowance<=0 or failures>=2:
            report['records'].append(dict(row,status='not_attempted_bound_or_two_failures'));continue
        target=root/f"{item['id']}.zip";part=target.with_suffix('.partial')
        if target.exists() or part.exists():raise FileExistsError('Unreceipted destination exists')
        timeout=min(plan['per_file_s'],remaining)
        try:
            cmd=['curl','--location','--fail','--silent','--show-error','--connect-timeout','8',
                 '--max-time',str(timeout),'--max-filesize',str(allowance),'--proto','=https','--proto-redir','=https',
                 '--output',str(part),'--write-out','%{http_code}',item['url']]
            completed=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout+3)
            row.update(http_status=completed.stdout[-3:],curl_exit=completed.returncode)
            if completed.returncode:raise ValueError(completed.stderr[:300])
            with zipfile.ZipFile(part)as archive:
                entries=archive.infolist()
                if not entries or len(entries)>2000 or sum(e.file_size for e in entries)>256<<20:
                    raise ValueError('Archive size/member limit')
                if any(e.file_size>128<<20 or e.flag_bits&1 for e in entries):raise ValueError('Encrypted/oversized member')
            part.replace(target);failures=0
            row.update(status='downloaded_archive_verified',path=str(target.relative_to(ROOT)),sha256=sha(target),entries=len(entries))
        except (OSError,ValueError,subprocess.SubprocessError,zipfile.BadZipFile)as exc:
            failures+=1;row.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        row['bytes']=target.stat().st_size if target.exists()else part.stat().st_size if part.exists()else 0
        transferred+=row['bytes'];report['records'].append(row);save()
        if transferred>plan['total_bytes']:raise ValueError('Cumulative byte bound exceeded')
        print(json.dumps(row),flush=True)
    report.update(complete=True,all_archives_downloaded=all(r['status']=='downloaded_archive_verified'for r in report['records']))
    save()


if __name__=='__main__':main()
