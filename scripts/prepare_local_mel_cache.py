"""Read existing Modal Volume caches to local disk; never launch cloud compute."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'inputs/r09_positive_pairs/catalog.pt'
    catalog=torch.load(source,map_location='cpu',weights_only=False)
    entries=[e for p in catalog['train_pairs'] for e in p['entries']]+catalog['validation_candidates']
    songs=sorted({e['row']['audio_sha256'] for e in entries})
    if len(songs)>40:raise ValueError('Bounded local panel unexpectedly expanded')
    out=ROOT/'inputs/d65_local/mel10';out.mkdir(parents=True,exist_ok=True)
    report_path=ROOT/'reports/d65_local_mel_transfer.json'
    report=dict(complete=False,source_catalog_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        requested_songs=len(songs),files=[],errors=[],downloaded_bytes=0,
        new_cloud_functions=0,new_gpu_calls=0,egress_billing='Not inferred; existing campaign transfer reserve retained')
    start=time.monotonic();failures=0
    def save():
        report['wall_s']=time.monotonic()-start
        report_path.write_text(json.dumps(report,indent=2)+'\n')
    save()
    for sha in songs:
        if time.monotonic()-start>600:raise TimeoutError('Local cache transfer deadline')
        path=out/(sha+'.npy');raw=out/(sha+'.download.npy')
        method='existing_local'
        try:
            if not path.exists():
                existing=ROOT/'inputs/d64_transport_cache'/(sha+'.npy')
                if existing.exists():
                    shutil.copyfile(existing,path)
                else:
                    if raw.exists():raise FileExistsError(raw)
                    proc=subprocess.run([str(ROOT/'.venv/bin/modal'),'volume','get','beatfcn-data',
                        '/mel10/'+sha+'.npy',str(raw)],capture_output=True,text=True,timeout=45)
                    if proc.returncode:raise RuntimeError('Read-only Modal download returned '+str(proc.returncode))
                    report['downloaded_bytes']+=raw.stat().st_size;method='read_only_modal_volume'
                    if report['downloaded_bytes']>256*2**20:raise ValueError('Transfer exceeded 256MiB')
                    data=np.load(raw,allow_pickle=False)
                    np.save(path,np.ascontiguousarray(data))
                    if not np.array_equal(data,np.load(path,allow_pickle=False)):raise ValueError('Layout conversion changed data')
            a=np.load(path,allow_pickle=False,mmap_mode='r')
            cells=len(np.load(ROOT/'inputs/r09_positive_pairs/taps/v4_i8_L9-16-24'/sha/'L9.npy',mmap_mode='r'))
            if a.shape!=(4*cells,128) or not a.flags.c_contiguous or not np.isfinite(a).all():
                raise ValueError('Existing mel shape/clock/finiteness/layout mismatch')
            report['files'].append(dict(audio_sha256=sha,path=str(path.relative_to(ROOT)),shape=list(a.shape),
                dtype=str(a.dtype),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_file_sha256=hashlib.sha256(raw.read_bytes()).hexdigest() if raw.exists() else None,
                method=method,coverage_end_us=4*cells*10_000,
                clock='Legacy cache; coverage ends at whole 40ms cell; no fabricated tail'))
            failures=0
        except (ValueError,RuntimeError,subprocess.TimeoutExpired,OSError) as exc:
            report['errors'].append(dict(audio_sha256=sha,error=str(exc)));failures+=1
        save()
        if failures>=2:break
    report['complete']=len(report['files'])==len(songs)
    save();print(json.dumps({k:report[k] for k in ['complete','requested_songs','downloaded_bytes','errors','wall_s']}))


if __name__=='__main__':main()
