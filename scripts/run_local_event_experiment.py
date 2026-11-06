"""Bounded CPU/MPS event experiments with explicit exposure and memory curves.

The initial fixed-batch mode establishes backend mechanics only. Quality runs
require a separately identified held-out panel; a training fit never fills it in.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

import numpy as np
import torch

from beatfcn.events.canonical import Event, LaneLoc, LineLoc
from beatfcn.model.full_event_codec import EventRequest, EventVocabulary
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.train.event_conditions import encode_controls, training_labels
from scripts.probe_full_event_model import load_batch, metrics

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()


def memory(device):
    result = dict(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if device == 'mps':
        result.update(mps_tensor_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_bytes=torch.mps.driver_allocated_memory(),
                      mps_recommended_bytes=torch.mps.recommended_max_memory())
    return result


def labels(manifest, batch):
    """Training labels are derived only in fixed-batch diagnostic mode."""
    views = []
    for row in manifest['records']:
        request = EventRequest(**row['request'])
        events = [Event(str(i), e['onset_us'], e['end_us'], e['type'],
                  LaneLoc(e['lane']) if request.mode == 'mania_4k' else LineLoc('rhythm', 'above', 0.))
                  for i, e in enumerate(row['target_events'])]
        scope = (request.start_us, request.end_us)
        bands, _ = training_labels(request.mode, events, scope)
        views.append(encode_controls(request.mode, bands, scope, origin_us=request.start_us))
    return {**batch, 'controls': torch.from_numpy(np.stack(views))}


def run(plan_path):
    plan = json.loads(plan_path.read_text())
    output = ROOT/plan['output']
    if output.exists():
        raise FileExistsError(output)
    if plan['kind'] != 'fixed_batch_backend_and_fit':
        raise ValueError('This run kind cannot stand in for a held-out quality experiment')
    device = plan['device']
    if device not in ('cpu', 'mps'):
        raise ValueError('Local experiments cannot allocate cloud resources')
    if os.getenv('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise ValueError('Silent MPS CPU fallback is forbidden for these timings')
    if device == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS is unavailable; do not silently change the experiment backend')
        torch.mps.set_per_process_memory_fraction(plan['mps_memory_fraction'])
    torch.set_num_threads(plan['cpu_threads'])
    torch.manual_seed(plan['seed'])
    if device == 'mps':
        torch.mps.manual_seed(plan['seed'])
    started = time.monotonic()
    report = dict(id=plan['id'], complete=False, passed=False, updates=0,
        kind=plan['kind'], device=device, torch_version=torch.__version__, macos=platform.mac_ver()[0],
        hardware=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
        physical_memory_bytes=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)),
        new_cloud_spend_usd=0, quality_winner=None, heldout_generation=None, curve=[],
        processed_valid_tokens=0, processed_requests=0, flops=None,
        flops_note='Not measured; parameter count is not a complete attention compute estimate',
        sources_sha256={str(plan_path.relative_to(ROOT)):digest(plan_path),
            'beatfcn/model/full_event_model.py':digest(ROOT/'beatfcn/model/full_event_model.py'),
            'scripts/run_local_event_experiment.py':digest(__file__)})

    def save(phase):
        report.update(phase=phase, wall_s=time.monotonic()-started, memory=memory(device))
        output.parent.mkdir(parents=True,exist_ok=True)
        temp=output.with_suffix('.tmp')
        temp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        temp.replace(output)
        print(json.dumps(dict(id=report['id'],phase=phase,updates=report['updates'],wall_s=report['wall_s'])),flush=True)

    def bound():
        if time.monotonic()-started > plan['max_wall_s']:
            raise TimeoutError('Local experiment reached its registered wall-time bound')

    try:
        save('loading_real_inputs')
        folder=ROOT/plan['data']
        manifest,cpu_inputs,cpu_seq=load_batch(folder,'cpu')
        if manifest['tensor_sha256'] != plan['tensor_sha256']:
            raise ValueError('Frozen tensor identity differs from plan')
        cpu_inputs=labels(manifest,cpu_inputs)
        report['input_tensors']={}
        for group,batch in [('input',cpu_inputs),('target',cpu_seq)]:
            for name,tensor in batch.items():
                x=tensor.float()
                report['input_tensors'][group+':'+name]=dict(shape=list(x.shape),nan=int(torch.isnan(x).sum()),
                    inf=int(torch.isinf(x).sum()),minimum=float(x.min()),maximum=float(x.max()),
                    mean=float(x.mean()),std=float(x.std(unbiased=False)))
        if any(x['nan'] or x['inf'] for x in report['input_tensors'].values()):
            raise ValueError('Nonfinite actual tensor')
        config=FullEventConfig(**plan['config'])
        model=FullEventModel(config,EventVocabulary(**manifest['vocabulary'])).eval()
        report.update(config=asdict(config),parameters=sum(p.numel() for p in model.parameters()),
            real_charts=[r['chart_id'] for r in manifest['records']],
            source_data_sha256=manifest['tensor_sha256'],
            condition_provenance='All applicable human target bands within the training request; not a product evaluation')
        with torch.no_grad():
            expected=model(cpu_inputs,cpu_seq)
        expected_loss=event_loss(model(cpu_inputs,cpu_seq),cpu_seq)
        expected_loss.backward()
        expected_grad=model.control_projection.weight.grad.detach().clone()
        model.zero_grad(set_to_none=True)
        inputs={k:v.to(device) for k,v in cpu_inputs.items()}
        seq={k:v.to(device) for k,v in cpu_seq.items()}
        model=model.to(device)
        logits=model(inputs,seq)
        loss=event_loss(logits,seq);loss.backward();sync(device)
        report['backend']=dict(logits_max_error=float((logits.detach().cpu()-expected).abs().max()),
            control_gradient_max_error=float((model.control_projection.weight.grad.cpu()-expected_grad).abs().max()))
        report['initial']=metrics(logits.detach(),seq)
        report['initial']['request_mean_ce']=float(loss.detach())
        if max(report['backend'].values())>plan['max_backend_error']:
            raise ValueError('CPU/backend equivalence failed')
        report['gradients_finite']=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        if not report['gradients_finite']:
            raise ValueError('Absent/nonfinite real gradient')
        model.zero_grad(set_to_none=True)
        del logits,loss,expected,expected_loss,expected_grad
        save('backend_verified')
        optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=0.)
        model.train();sync(device);training_start=time.monotonic()
        tokens=int(cpu_seq['valid'].sum());requests=len(inputs['mode'])
        for step in range(1,plan['steps']+1):
            bound();optimizer.zero_grad(set_to_none=True)
            logits=model(inputs,seq);loss=event_loss(logits,seq)
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),plan['clip'],error_if_nonfinite=True)
            optimizer.step()
            report['updates']=step
            report['processed_valid_tokens']+=tokens;report['processed_requests']+=requests
            if step==1 or step%10==0:
                sync(device)
                report['curve'].append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(grad),
                    valid_tokens=report['processed_valid_tokens'],requests=report['processed_requests'],
                    train_elapsed_s=time.monotonic()-training_start,memory=memory(device)))
                save('fitting_training_batch')
            del logits,loss
        sync(device);report['train_s']=time.monotonic()-training_start
        del optimizer
        model.eval()
        with torch.no_grad():
            final=model(inputs,seq)
            report['final']=metrics(final,seq)
            report['final']['request_mean_ce']=float(event_loss(final,seq))
            n=min(24,seq['tokens'].shape[1]);short={k:v[:,:n] for k,v in seq.items()}
            memory_bank=model.encode(inputs);cache=model.start_cache(memory_bank)
            cached=torch.cat([model.decode(*(short[k][:,i:i+1] for k in ('tokens','fields','times','valid')),
                                memory_bank,cache=cache) for i in range(n)],1)
            report['cache_max_error']=float((cached-final[:,:n]).abs().max())
        del final,cached,cache,memory_bank
        save('checking_training_generation')
        gen_start=time.monotonic()
        requests=[EventRequest(**row['request']) for row in manifest['records']]
        generations=generate_batch(model,inputs,requests,plan['generation_token_cap'],
                deadline=min(started+plan['max_wall_s'],gen_start+plan['generation_max_s']))
        sync(device);report['generation_s']=time.monotonic()-gen_start
        for result,row in zip(generations,manifest['records']):
            result['chart_id']=row['chart_id'];result['target_events']=row['target_events']
            signature=lambda e:(e['onset_us'],e['end_us'],e['type'],e['lane'])
            result['exact_multiset']=Counter(map(signature,result['events']))==Counter(map(signature,row['target_events']))
        report['generations']=generations
        report['passed']=(report['final']['request_mean_ce']<=plan['maximum_ce']
            and min(v['accuracy'] for v in report['final']['fields'].values())>=plan['minimum_field_accuracy']
            and report['cache_max_error']<=plan['max_backend_error']
            and all(g['complete'] and g['exact_multiset'] for g in generations))
        checkpoint=ROOT/plan['checkpoint'];checkpoint.parent.mkdir(exist_ok=True,parents=True)
        if checkpoint.exists():raise FileExistsError(checkpoint)
        torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=asdict(config),
            vocabulary=manifest['vocabulary'],diagnostic_only=True,plan_sha256=digest(plan_path)),checkpoint)
        report['checkpoint_sha256']=digest(checkpoint)
        report['complete']=True;save('complete_health_only')
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}';report['complete']=True
        save('failed');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    run(parser.parse_args().plan.resolve())
