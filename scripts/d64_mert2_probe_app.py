"""Bounded original-audio MERT2 timing/cache probe. No optimizer updates."""
from pathlib import Path
import hashlib,json,time
import modal
from beatfcn.train.campaign_budget import reserve,update_stage
from beatfcn.train.paid_call import await_bounded_call
ROOT=Path(__file__).resolve().parents[1]
STAGE='d64-frontend-preparation'
REV='7001f7657276238629e61c7746ce70735e0604c0'
app=modal.App('beatfcn-d64-mert2-source-probe')
volume=modal.Volume.from_name('beatfcn-data',create_if_missing=False)
RESOURCES=dict(gpu='L40S',cpu=4,memory_gib=16,timeout_s=600,startup_allowance_s=300,retries=0,max_containers=1)
image=(modal.Image.debian_slim(python_version='3.11').apt_install('ffmpeg')
    .pip_install('torch==2.6.0','torchaudio==2.6.0',index_url='https://download.pytorch.org/whl/cu124')
    .pip_install('numpy==2.2.6','transformers==4.53.2','safetensors==0.5.3')
    .run_commands("python -c \"from huggingface_hub import snapshot_download; snapshot_download('m-a-p/MERT-v2-FullSong',revision='"+REV+"',cache_dir='/weights/huggingface',allow_patterns=['*.py','*.json','*.safetensors'])\"")
    .env({'PYTHONPATH':'/work','HF_HUB_OFFLINE':'1','OMP_NUM_THREADS':'4'})
    .add_local_dir(ROOT/'beatfcn','/work/beatfcn')
    .add_local_dir(ROOT/'inputs/d64_source_audio','/work/audio'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@app.function(image=image,gpu='L40S',cpu=(4,4),memory=(16384,16384),timeout=600,startup_timeout=300,
              retries=0,max_containers=1,scaledown_window=2,volumes={'/data':volume})
def probe(expected):
    import numpy as np
    import torch
    from beatfcn.audio.mert2 import FrozenMert2,fine_mel
    from beatfcn.ingest.pipeline import decode
    for p,h in expected.items():
        if sha('/work/'+p)!=h:raise ValueError('Input identity changed: '+p)
    torch.set_num_threads(4)
    out=Path('/data/evolution/capability-evolution-20261024')/STAGE
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);start=time.perf_counter()
    report=dict(id=STAGE,complete=False,optimizer_updates=0,songs=[],scope='Frozen encoder health/cache timing only; no quality winner')
    def save():
        report['wall_s']=time.perf_counter()-start
        (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    try:
        encoder=FrozenMert2(cache_dir='/weights/huggingface')
        report.update(load_wall_s=time.perf_counter()-start,parameters=sum(p.numel() for p in encoder.model.parameters()))
        for item in json.loads(Path('/work/audio/manifest.json').read_text()):
            source=Path('/work/audio')/Path(item['path']).name
            if sha(source)!=item['sha256']:raise ValueError('Original audio SHA differs')
            wave=decode(str(source));torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
            result=encoder.encode(wave);torch.cuda.synchronize();encoded=time.perf_counter()-t
            mel=fine_mel(wave)
            folder=out/item['sha256'];folder.mkdir()
            for i,l in enumerate(encoder.spec.layers):np.save(folder/f'L{l}.npy',result['layers'][i])
            for name,value in [('times_s',result['times_s']),('intervals_s',result['intervals_s']),('mel10',mel['values']),
                               ('mel_times_s',mel['times_s']),('mel_intervals_s',mel['intervals_s'])]:np.save(folder/(name+'.npy'),value)
            (folder/'metadata.json').write_text(json.dumps(result['metadata'],indent=2)+'\n')
            old=np.load('/data/mel10/'+item['sha256']+'.npy')
            n=min(len(old),len(mel['values']))-8
            difference=float(np.abs(old[:n].astype(np.float32)-mel['values'][:n].astype(np.float32)).max())
            stats={}
            for name,a in [('wave',wave),('mert',result['layers']),('mel',mel['values'])]:
                f=a.astype(np.float64);stats[name]=dict(shape=list(a.shape),dtype=str(a.dtype),nan=int(np.isnan(f).sum()),inf=int(np.isinf(f).sum()),
                    minimum=float(f.min()),maximum=float(f.max()),mean=float(f.mean()),std=float(f.std()))
            if difference>.01:raise ValueError('Source mel does not reproduce existing interior: '+str(difference))
            report['songs'].append(dict(**item,real_samples=len(wave),audio_s=len(wave)/24000,encode_s=encoded,
                rtf=(len(wave)/24000)/encoded,frames=result['layers'].shape[1],mel_frames=len(mel['values']),old_mel_frames=len(old),
                old_interior_max_error=difference,stats=stats,metadata=result['metadata'],
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in folder.iterdir()}))
            print(json.dumps({k:report['songs'][-1][k] for k in ['mode','audio_s','encode_s','rtf','frames','mel_frames','old_mel_frames']}),flush=True)
            save()
        report['complete']=True;save();return report
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}';save();raise
    finally:volume.commit()


@app.local_entrypoint()
def main():
    expected={f'beatfcn/{p.relative_to(ROOT/"beatfcn")}':sha(p) for p in (ROOT/'beatfcn').rglob('*.py')}
    expected.update({'audio/'+p.name:sha(p) for p in (ROOT/'inputs/d64_source_audio').iterdir() if p.is_file()})
    ledger=ROOT/'reports/rhythm_v2_budget.json'
    # Image assembly can take minutes; refresh all read-only state after it, before the GPU reservation.
    from scripts.refresh_rhythm_state import refresh
    connection=refresh()
    identity=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
    reserve(ledger,STAGE,RESOURCES,json.loads((ROOT/'reports/rhythm_v2_rates.json').read_text()),connection,
            app_id=app.app_id,source_sha256=identity)
    try:
        call=probe.spawn(expected);update_stage(ledger,STAGE,status='running',function_call_id=call.object_id)
        print(json.dumps(dict(app=app.app_id,call=call.object_id,stage=STAGE)),flush=True)
        result=await_bounded_call(call,timeout_s=RESOURCES['timeout_s']+RESOURCES.get('startup_allowance_s',120),app_id=app.app_id,modal_command=[str(ROOT/'.venv/bin/modal')]);(ROOT/'reports/d64_mert2_probe.json').write_text(json.dumps(result,indent=2)+'\n')
        update_stage(ledger,STAGE,status='complete',ended_at=time.time(),function_wall_s=result['wall_s'])
    except BaseException as exc:
        update_stage(ledger,STAGE,status='observation_error',error=f'{type(exc).__name__}: {exc}',observed_at=time.time());raise
