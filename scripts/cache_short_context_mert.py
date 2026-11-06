"""Same frozen MERT weights, 32s/16s-overlap extraction; no chart labels.

Parent commits <=8-song workers so MPS allocator lifetime is bounded. The
ordinary downstream song summary remains whole-song in both comparison arms.
Only the pretrained encoder's extraction context changes.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.run_local_portfolio_scale import sha,write


ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'bench/rhythm-v2/local-mert-context-cache-v1.json'
REPORT=ROOT/'reports/local_mert_context_cache_v1.json'
ART=ROOT/'artifacts/local-mert-context-cache-v1'


def prepare():
    if PLAN.exists() or REPORT.exists():raise FileExistsError('Preserve context cache attempt')
    from scripts.run_frontend_canvas_comparison import setup
    comparison=json.loads((ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text())
    panel,store=setup(comparison)
    parent=json.loads((ROOT/'bench/rhythm-v2/local-portfolio-scale-v1.json').read_text())
    ids=parent['indices']['train64']+parent['indices']['dev']+parent['indices']['empty_dev']
    audios=sorted({panel.records[i]['audio_sha256'] for i in ids})
    acquisition=json.loads((ROOT/'bench/rhythm-v2/d96-local-mert-cache-v1.json').read_text())
    sources={r['audio_sha256']:r for r in acquisition['audios']}
    plan=dict(id='local-mert-context-cache-v1',checkpoint_dir=acquisition['checkpoint_dir'],
        config=acquisition['config'],revision=acquisition['revision'],weight_sha256=acquisition['weight_sha256'],
        environment=acquisition['environment'],audios=[sources[a] for a in audios],
        full_cache_report=comparison['mert_cache_report'] if 'mert_cache_report' in comparison else 'reports/d96_local_mert_cache_workers_v2.json',
        specification=dict(layers=[12,23],context_limit_s=32,long_window_s=32,overlap_s=16),
        workers=8,worker_wall_s=600,total_wall_s=3600,cache_byte_cap=4_000_000_000,
        minimum_free_bytes=8_000_000_000,optimizer_updates=0,cloud_compute_calls=0,confirmation_scored=False,
        hypothesis='Test same pretrained weights with full versus32second extraction. Downstream pooling, clocks, layers, dtype, waveform and fine mel stay fixed.',
        source_sha256={n:sha(ROOT/n) for n in ('scripts/cache_short_context_mert.py','beatfcn/audio/mert2.py','beatfcn/audio/local_mert.py','beatfcn/audio/mert_attention.py',
            'inputs/d96_mert2_source/mert2_pinned/modeling_mert2.py','inputs/d96_mert2_source/mert2_pinned/configuration_mert2.py',acquisition['config'])})
    if sha(ROOT/plan['checkpoint_dir']/'model.safetensors')!=plan['weight_sha256']:raise ValueError('Frozen pretrained checkpoint changed')
    write(PLAN,plan)
    snapshot=ART/'source';snapshot.mkdir(parents=True,exist_ok=True)
    for n in plan['source_sha256']:
        dest=snapshot/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    shutil.copy2(PLAN,snapshot/'frozen-plan.json')
    return plan


def worker(group):
    os.environ.update(PYTORCH_ENABLE_MPS_FALLBACK='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import numpy as np
    import torch
    from beatfcn.audio.local_mert import CpuMelTransfer,LocalFrozenMert2
    from beatfcn.audio.mert2 import Mert2Spec
    from beatfcn.audio.mert_attention import install_query_chunks
    plan=json.loads(PLAN.read_text())
    for n,d in plan['source_sha256'].items():
        if sha(ROOT/n)!=d:raise ValueError('Source drift '+n)
    torch.set_num_threads(2);torch.mps.set_per_process_memory_fraction(.5)
    sys.path.insert(0,str(ROOT/'inputs/d96_mert2_source'))
    official=importlib.import_module('mert2_pinned.modeling_mert2')
    configmod=importlib.import_module('mert2_pinned.configuration_mert2')
    config=configmod.MERT2Config(**json.loads((ROOT/plan['config']).read_text()));config._attn_implementation='sdpa'
    model,info=official.MERT2Model.from_pretrained(str(ROOT/plan['checkpoint_dir']),config=config,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',output_loading_info=True,local_files_only=True)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('Pretrained load mismatch')
    model.eval().requires_grad_(False).to('mps');model.feature_extractor=CpuMelTransfer(model.feature_extractor,'mps');install_query_chunks(model,512)
    spec={**plan['specification'],'layers':tuple(plan['specification']['layers'])}
    encoder=LocalFrozenMert2(model,Mert2Spec(**spec));records=[];started=time.monotonic()
    full=json.loads((ROOT/plan['full_cache_report']).read_text());full={r['audio_sha256']:r for r in full['records']}
    path=ART/f'worker-{group:02d}.json'
    if path.exists():raise FileExistsError('No worker retry')
    for source in plan['audios'][8*group:8*(group+1)]:
        if time.monotonic()-started>plan['worker_wall_s']:raise TimeoutError('Short-context worker wall')
        if shutil.disk_usage(ROOT).free<plan['minimum_free_bytes']:raise ValueError('Free disk guard')
        audio=source['audio_sha256'];raw=ROOT/source['raw_path']
        if sha(raw)!=audio:raise ValueError('Audio SHA changed')
        wave=np.frombuffer(subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(raw),'-vn','-f','f32le','-ac','1','-ar','24000','pipe:1'],timeout=30),dtype='<f4').copy()
        if len(wave)!=source['decoded_samples']:raise ValueError('Decoded sample identity')
        tick=time.monotonic();result=encoder.encode(wave);torch.mps.synchronize();elapsed=time.monotonic()-tick
        old=ROOT/full[audio]['path'];times=np.load(old/'times_s.npy')
        if not np.array_equal(times,result['times_s']):raise ValueError('Full/short feature clocks differ')
        folder=ART/'features'/audio;folder.mkdir(parents=True,exist_ok=False)
        for j,layer in enumerate((12,23)):np.save(folder/f'L{layer}.npy',np.ascontiguousarray(result['layers'][j]))
        np.save(folder/'times_s.npy',result['times_s']);np.save(folder/'intervals_s.npy',result['intervals_s'])
        metadata={**result['metadata'],'audio_sha256':audio,'checkpoint_sha256':plan['weight_sha256'],'query_chunk':512,
            'frontend_placement':'explicit_cpu_float32_then_mps','common_mel_reencoded':False}
        write(folder/'metadata.json',metadata)
        x=result['layers'].astype(np.float32)
        oldx=np.stack([np.load(old/f'L{l}.npy',mmap_mode='r') for l in (12,23)]).astype(np.float32)
        cosine=float(np.sum(x*oldx)/max(np.linalg.norm(x)*np.linalg.norm(oldx),1e-30))
        files={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in folder.iterdir()}
        record=dict(audio_sha256=audio,path=str(folder.relative_to(ROOT)),samples=len(wave),frames=len(times),dtype='float16',
            finite=bool(np.isfinite(x).all()),minimum=float(x.min()),maximum=float(x.max()),mean=float(x.mean()),std=float(x.std()),
            files=files,metadata=metadata,encode_s=elapsed,full_feature_cosine=cosine,
            memory=dict(tensor_bytes=torch.mps.current_allocated_memory(),driver_bytes=torch.mps.driver_allocated_memory(),recommended_bytes=torch.mps.recommended_max_memory()))
        if not record['finite']:raise ValueError('Nonfinite short context features')
        records.append(record);write(path,dict(complete=False,records=records,wall_s=time.monotonic()-started))
        print(json.dumps(dict(worker=group,songs=len(records),encode_s=elapsed)),flush=True)
        del wave,result,x,oldx;torch.mps.empty_cache()
    write(path,dict(complete=True,records=records,wall_s=time.monotonic()-started))


def main():
    plan=prepare();started=time.monotonic()
    report=dict(id=plan['id'],complete=False,status='encoding',planned_songs=len(plan['audios']),committed_songs=0,
        records=[],workers=[],optimizer_updates=0,new_cloud_compute_calls=0,plan_sha256=sha(PLAN))
    write(REPORT,report)
    try:
        for group in range((len(plan['audios'])+7)//8):
            remaining=plan['total_wall_s']-(time.monotonic()-started)
            if remaining<=0:raise TimeoutError('Total extraction wall')
            with (ART/f'worker-{group:02d}.log').open('x') as out:
                child=subprocess.Popen([str(ROOT/plan['environment']/'bin/python'),'-u','-m','scripts.cache_short_context_mert','--worker',str(group)],
                    cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                try:code=child.wait(timeout=min(plan['worker_wall_s']+60,remaining))
                except subprocess.TimeoutExpired:
                    child.terminate();child.wait(timeout=30);raise
            if code:raise RuntimeError('Explicit failed worker '+str(group))
            result=json.loads((ART/f'worker-{group:02d}.json').read_text())
            if not result['complete']:raise ValueError('Incomplete worker')
            report['records'].extend(result['records']);report['workers'].append(dict(group=group,wall_s=result['wall_s']))
            report.update(committed_songs=len(report['records']),wall_s=time.monotonic()-started)
            byte_count=sum(v['bytes'] for r in report['records'] for v in r['files'].values())
            if byte_count>plan['cache_byte_cap']:raise ValueError('Cache byte guard')
            report['committed_bytes']=byte_count;write(REPORT,report)
        report.update(status='completed',complete=True,wall_s=time.monotonic()-started);write(REPORT,report)
    except BaseException as exc:
        report.update(status='stopped',failure=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-started);write(REPORT,report);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);args=parser.parse_args()
    main() if args.worker is None else worker(args.worker)
