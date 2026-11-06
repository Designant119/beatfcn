"""Zero-update, real-input capacity and SSD backend checks. Not quality evidence."""
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time

import torch

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_typed_canvas import LocalTypedCanvas, canvas_loss
from beatfcn.model.mamba2_reference import Mamba2Reference
from beatfcn.train.relative_canvas import relative_batch
from scripts.run_frontend_canvas_comparison import setup, rows_for


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/local-portfolio-v1/resource-probe'
REPORT = ART / 'report.json'


def save(report):
    ART.mkdir(parents=True,exist_ok=True)
    temp = REPORT.with_suffix('.partial')
    temp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    temp.replace(REPORT)


def memory():
    return dict(mps_allocated_bytes=torch.mps.current_allocated_memory(),
        mps_driver_bytes=torch.mps.driver_allocated_memory(),
        process_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        note='Allocator snapshots and process high-water RSS, not a sampled total-system peak')


def parity(cpu, gpu):
    return max(float((cpu[k]-gpu[k].detach().cpu()).abs().max()) for k in cpu)


def main():
    if REPORT.exists():
        raise FileExistsError('Preserve existing resource probe')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise ValueError('Silent MPS CPU fallback is not permitted')
    if not torch.backends.mps.is_available():
        raise RuntimeError('Native MPS required for this backend probe')
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.65)
    report=dict(status='running',optimizer_updates=0,cloud_compute_calls=0,models={},ssd={},
        wall_limit_s=300,mps_memory_fraction=.65,torch_version=torch.__version__,
        purpose='Actual real-input fwd/back measurements; no optimizer or checkpoint selection')
    paths=['bench/rhythm-v2/local-portfolio-cohort-v1.json',
        'bench/rhythm-v2/local-frontend-comparison-v2.json',
        'beatfcn/model/local_typed_canvas.py','beatfcn/model/mamba2_reference.py',
        'beatfcn/train/relative_canvas.py','scripts/probe_local_portfolio_resources.py']
    report['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    started=time.monotonic()
    def alarm_handler(*_):
        raise TimeoutError('Frozen300s resource probe deadline')
    signal.signal(signal.SIGALRM,alarm_handler)
    signal.alarm(report['wall_limit_s'])
    save(report)
    try:
        p=json.loads((ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text())
        panel,store=setup(p)
        row=rows_for(panel,store,[dict(index=p['health_indices'][0],view='draft')])['mert'][0]
        x,y=relative_batch([row])
        report['index']=row['index']
        report['inputs']={k:dict(shape=list(v.shape),finite=bool(torch.isfinite(v).all())) for k,v in x.items()}
        for label,d,heads,ffn,layers in [('small',128,4,384,1),('capacity',256,8,1024,2)]:
            config={**p['models']['mert'],'d':d,'heads':heads,'kv_heads':2,'ffn':ffn,'encoder_layers':layers}
            torch.manual_seed(20261028)
            model=LocalTypedCanvas(FullEventConfig(**config),density_reference=p['density_reference']).eval()
            entry=dict(config=config,parameters=sum(v.numel() for v in model.parameters()),steps=[])
            report['models'][label]=entry
            t=time.monotonic()
            output=model(x);loss,_=canvas_loss(output,y,x);loss.backward()
            cpu={k:v.detach().clone() for k,v in output.items()}
            cpu_grad={k:v.grad.detach().clone() for k,v in model.named_parameters() if v.grad is not None}
            entry.update(cpu_fwd_backward_s=time.monotonic()-t,cpu_loss=float(loss.detach()))
            del output,loss
            model.zero_grad(set_to_none=True);model.to('mps')
            xx={k:v.to('mps') for k,v in x.items()};yy={k:v.to('mps') for k,v in y.items()}
            for repeat in range(3):
                model.zero_grad(set_to_none=True);torch.mps.synchronize();t=time.monotonic()
                out=model(xx);ll,_=canvas_loss(out,yy,xx);ll.backward();torch.mps.synchronize()
                seconds=time.monotonic()-t
                if repeat==0:
                    error=parity(cpu,out)
                    grad_error=max(float((v.grad.detach().cpu()-cpu_grad[k]).abs().max())
                        for k,v in model.named_parameters() if v.grad is not None)
                    entry.update(cpu_mps_logit_max_error=error,cpu_mps_gradient_max_error=grad_error)
                    if max(error,grad_error)>1e-4:
                        raise ValueError('Real-input CPU/MPS numerical mismatch')
                if not torch.isfinite(ll) or any(not torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None):
                    raise ValueError('Nonfinite real loss or gradient')
                entry['steps'].append(dict(repeat=repeat,fwd_backward_s=seconds,loss=float(ll.detach()),memory=memory()))
                print(json.dumps(dict(stage='capacity',model=label,repeat=repeat,seconds=seconds,parameters=entry['parameters'])),flush=True)
                del out,ll
            del model,xx,yy,cpu,cpu_grad
            gc.collect();torch.mps.empty_cache();save(report)
        # SSD uses actual10ms mel values as a numeric backend input, not a trained
        # model or proof that its later complete candidate can fit chart targets.
        for length in (200,1600):
            real=x['mel'][:,:length]
            u=torch.cat([real,real],-1).contiguous()
            torch.manual_seed(20261028)
            block=Mamba2Reference(256,d_state=128,headdim=64,chunk_size=64).eval()
            t=time.monotonic();out=block(u);out.square().mean().backward()
            reference=out.detach();entry=dict(cpu_fwd_backward_s=time.monotonic()-t,input_shape=list(u.shape))
            block.zero_grad(set_to_none=True);block.to('mps');uu=u.to('mps')
            torch.mps.synchronize();t=time.monotonic();observed=block(uu);observed.square().mean().backward();torch.mps.synchronize()
            entry.update(mps_fwd_backward_s=time.monotonic()-t,
                cpu_mps_max_error=float((reference-observed.detach().cpu()).abs().max()),memory=memory())
            if entry['cpu_mps_max_error']>1e-4:
                raise ValueError('SSD CPU/MPS mismatch')
            report['ssd'][str(length)]=entry
            print(json.dumps(dict(stage='ssd',length=length,**entry)),flush=True)
            del block,out,reference,observed,u,uu
            gc.collect();torch.mps.empty_cache();save(report)
        report['status']='completed'
    except BaseException as exc:
        report.update(status='failed',failure=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        signal.alarm(0);report['wall_s']=time.monotonic()-started;save(report)


if __name__=='__main__':
    main()
