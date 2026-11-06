"""Verify the new event adapter against real tensors using the optimized pool/ring."""
from pathlib import Path
import hashlib,json,time
import modal
from beatfcn.train.campaign_budget import reserve,update_stage
from beatfcn.train.paid_call import await_bounded_call
app=modal.App('beatfcn-d64-event-transport')
STAGE='d64-event-transport-v3'
volume=modal.Volume.from_name('beatfcn-data',create_if_missing=False)
if modal.is_local():
    from scripts.d64_backbone_probe_v2_app import image as base_image,ROOT
    image=(base_image.add_local_file(ROOT/'inputs/r08_local_real_probe/pca/i8_9-16-24.npz','/work/scales.npz')
           .add_local_dir(ROOT/'inputs/d64_transport_cache','/work/transport_cache'))
else:
    ROOT=Path('/work')
    image=None
RESOURCES=dict(gpu='L40S',cpu=8,memory_gib=32,timeout_s=300,startup_allowance_s=120,retries=0,max_containers=1)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@app.function(image=image,gpu='L40S',cpu=(8,8),memory=(32768,32768),timeout=300,retries=0,
              max_containers=1,scaledown_window=2,volumes={'/data':volume})
def probe(expected):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from scripts.probe_full_event_model import load_batch
    from beatfcn.model.full_event_model import FullEventModel,FullEventConfig,event_loss
    from beatfcn.train.event_transport import EventTransport,pack_event_batch
    from beatfcn.train.pool import SongPool
    from beatfcn.train.profiling import TraceWindows
    from beatfcn.cloud.gpu_util import GpuUtil
    for path,digest in expected.items():
        if sha('/work/'+path)!=digest:raise ValueError('Input identity changed '+path)
    torch.set_num_threads(8);torch.manual_seed(20261025)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path('/data/evolution/capability-evolution-20261024')/STAGE
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);start=time.perf_counter();report=dict(id=STAGE,complete=False,quality_winner=None,optimizer_updates=0)
    monitor=GpuUtil();pool=None;stream=None
    def save():
        report['wall_s']=time.perf_counter()-start
        (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    try:
        manifest,cpu,seq_cpu=load_batch(Path('/work/fixture'),'cpu')
        _,reference,seq=load_batch(Path('/work/fixture'),'cuda')
        scales=np.load('/work/scales.npz')
        layers=(9,16,24);summaries={}
        for i,r in enumerate(manifest['records']):
            n=r['tensors']['song']['shape'][0]
            p=Path('/tmp')/(r['audio_sha256']+'-summary.npy')
            np.save(p,cpu['song'][i,:n].numpy());summaries[r['audio_sha256']]=p
            for l in layers:
                relative=f'inputs/r09_positive_pairs/taps/v4_i8_L9-16-24/{r["audio_sha256"]}/L{l}.npy'
                cloud=Path('/data/taps/v4_i8_L9-16-24')/r['audio_sha256']/f'L{l}.npy'
                if sha(cloud)!=manifest['input_sha256'][relative]:raise ValueError('Actual cloud feature cache differs')
        class Loader:
            def paths(self,key,mel=True):
                return [Path('/data/taps/v4_i8_L9-16-24')/key/f'L{l}.npy' for l in layers]+[summaries[key],Path('/work/transport_cache')/(key+'.npy')]
        pool=SongPool(Loader(),layers,device='cuda',capacity=8,loaders=2,mel=True,check_finite=True)
        recipe=dict(keys=[r['audio_sha256'] for r in manifest['records']],layers=layers,
            local_slices=[(r['local_cells'][0],r['local_cells'][1]-r['local_cells'][0]) for r in manifest['records']],
            song_slices=[(0,r['tensors']['song']['shape'][0]) for r in manifest['records']],
            mel_slices=[(r['local_cells'][0]*4,(r['local_cells'][1]-r['local_cells'][0])*4) for r in manifest['records']])
        packed=pack_event_batch(cpu,seq_cpu,recipe)
        tape=SimpleNamespace(batches=[(packed,{})]*64,skip_batches=0)
        stream=EventTransport(tape,pool,prefetch=8,scales={l:torch.as_tensor(scales[f'scale_{l}'],device='cuda') for l in layers})
        actual,actual_seq,_=next(stream)
        report['input_max_errors']={k:float((actual[k].float()-v.float()).abs().max()) for k,v in reference.items()}
        report['targets_identical']=all(torch.equal(actual_seq[k],v) for k,v in seq.items())
        if max(report['input_max_errors'].values())!=0 or not report['targets_identical']:raise ValueError('Transport changes actual tensors')
        model=FullEventModel(FullEventConfig(mel_bins=128,dropout=0.)).cuda().eval()
        with torch.no_grad():
            direct=model(reference,seq);attached=model(actual,actual_seq)
        report['logits_max_error']=float((direct-attached).abs().max())
        if report['logits_max_error']>1e-5:raise ValueError('Transport changes logits')
        del direct,attached,reference,seq
        model.train();monitor.take();torch.cuda.synchronize();loop=time.perf_counter()
        traces=TraceWindows(out/'traces',(0,),2,cuda=True,detailed=True)
        for i,(inp,target,_) in enumerate(stream):
            traces.begin(i,time.perf_counter()-loop)
            model.zero_grad(set_to_none=True)
            with traces.scope('event.forward_backward'):
                loss=event_loss(model(inp,target),target);loss.backward()
            if i==0:
                report['gradients_finite']=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                if not report['gradients_finite']:raise ValueError('Nonfinite transport-path gradients')
            record=traces.end(i)
            if record:report['trace']=record
        torch.cuda.synchronize()
        report.update(complete=True,passed=True,loop_s=time.perf_counter()-loop,backward_microbatches=63,
            gpu=monitor.take(),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,transport=stream.telemetry(),
            scope='Identical real tensors/logits; fixed two-song stream. Not full-corpus throughput or 8-minute guard validation.')
        save();return report
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}';save();raise
    finally:
        if stream:stream.close()
        if pool:pool.pool.shutdown(wait=True)
        monitor.close();volume.commit()


@app.local_entrypoint()
def main():
    from scripts.refresh_rhythm_state import refresh
    connection=refresh();ledger=ROOT/'reports/rhythm_v2_budget.json'
    expected={f'beatfcn/{p.relative_to(ROOT/"beatfcn")}':sha(p) for p in (ROOT/'beatfcn').rglob('*.py')}
    for name in ['inputs.npz','manifest.json']:expected['fixture/'+name]=sha(ROOT/'inputs/diagnostic_panels/d64-full-event-mel'/name)
    expected['scales.npz']=sha(ROOT/'inputs/r08_local_real_probe/pca/i8_9-16-24.npz')
    expected.update({'transport_cache/'+p.name:sha(p) for p in (ROOT/'inputs/d64_transport_cache').iterdir() if p.is_file()})
    identity=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
    reserve(ledger,STAGE,RESOURCES,json.loads((ROOT/'reports/rhythm_v2_rates.json').read_text()),connection,app_id=app.app_id,source_sha256=identity)
    try:
        call=probe.spawn(expected);update_stage(ledger,STAGE,status='running',function_call_id=call.object_id)
        result=await_bounded_call(call,timeout_s=RESOURCES['timeout_s']+RESOURCES.get('startup_allowance_s',120),app_id=app.app_id,modal_command=[str(ROOT/'.venv/bin/modal')]);(ROOT/'reports/d64_event_transport_probe_v3.json').write_text(json.dumps(result,indent=2)+'\n')
        update_stage(ledger,STAGE,status='complete',ended_at=time.time(),function_wall_s=result['wall_s'])
        print(json.dumps({k:result[k] for k in ['complete','passed','wall_s','loop_s','gpu']}))
    except BaseException as exc:
        update_stage(ledger,STAGE,status='observation_error',error=f'{type(exc).__name__}: {exc}');raise
