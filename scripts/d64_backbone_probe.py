"""Real CUDA health checks only; cannot select a generalisation winner."""
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.model.full_event_codec import BOS, EventRequest, EventVocabulary
from beatfcn.cloud.gpu_util import GpuUtil
from scripts.probe_full_event_model import load_batch, metrics, generate


def run(plan, folder, output, publish):
    output = Path(output)
    manifest, inputs, sequence = load_batch(Path(folder), 'cuda')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    report = dict(stage=plan['id'], complete=False, candidates=[],
        quality_winner=None, scope='Real fixed-batch implementation probe, not held-out quality',
        plan=plan, torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
        input_identity=hashlib.sha256((Path(folder)/'inputs.npz').read_bytes()).hexdigest(), tensors={})
    for section, batch in [('input', inputs), ('target', sequence)]:
        for key, value in batch.items():
            f = value.float()
            report['tensors'][section+':'+key] = dict(shape=list(value.shape), dtype=str(value.dtype),
                nan=int(torch.isnan(f).sum()), inf=int(torch.isinf(f).sum()),
                minimum=float(f.min()), maximum=float(f.max()), mean=float(f.mean()), std=float(f.std(unbiased=False)))
    if any(v['nan'] or v['inf'] for v in report['tensors'].values()):
        raise ValueError('Non-finite actual inputs')

    def record():
        report['wall_s'] = time.monotonic() - started
        publish(report)

    def deadline():
        if time.monotonic() - started > plan['work_deadline_s']:
            raise TimeoutError('Bounded probe deadline: retain time for evidence flush')

    record()
    for candidate in plan['candidates']:
        row = dict(name=candidate['name'], passed=False, complete=False, updates=0, curve=[])
        report['candidates'].append(row)
        monitor = GpuUtil()
        try:
            deadline()
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(plan['seed'])
            config = FullEventConfig(**candidate['config'])
            model = FullEventModel(config, EventVocabulary(**manifest['vocabulary'])).cuda().eval()
            row.update(config=asdict(config), parameters=sum(p.numel() for p in model.parameters()))
            if not 237_500_000 <= row['parameters'] <= 262_500_000:
                raise ValueError('Candidate capacity outside frozen 0.25B +/-5%')
            # Full-vs-step, future-token causality, padding, and independent-cache reset.
            one = {k:v[:1] for k,v in inputs.items()}
            n = min(32, int(sequence['valid'][0].sum()))
            seq = {k:v[:1, :n] for k,v in sequence.items()}
            with torch.no_grad():
                memory = model.encode(one)
                full = model(one, seq)
                cache = model.start_cache(memory)
                chunks = [model.decode(*(seq[k][:,i:i+1] for k in ('tokens','fields','times','valid')),
                    memory,cache=cache) for i in range(n)]
                delta = float((torch.cat(chunks,1)-full).abs().max())
                altered = {k:v.clone() for k,v in seq.items()}
                altered['tokens'][:,n//2:] = BOS; altered['times'][:,n//2:] += 10.
                causal_error = float((model(one,altered)[:,:n//2]-full[:,:n//2]).abs().max())
                fresh = model.start_cache(memory)
                fresh_first = model.decode(*(seq[k][:,:1] for k in ('tokens','fields','times','valid')),memory,cache=fresh)
                reset_error = float((fresh_first-chunks[0]).abs().max())
                padded = {k:v.clone() for k,v in sequence.items()}
                padded['tokens'][~padded['valid']] = BOS
                padded['times'][~padded['valid']] += 100.
                full_batch = model(inputs,sequence)
                padding_error = float((model(inputs,padded)-full_batch)[sequence['valid']].abs().max())
                row['initial'] = metrics(full_batch,sequence)
                row['initial']['request_mean_ce'] = float(event_loss(full_batch,sequence))
            row['invariants'] = dict(cache_max_error=delta, future_max_error=causal_error,
                                    reset_max_error=reset_error, padding_max_error=padding_error)
            if max(delta,causal_error,reset_error,padding_error)>plan['max_fp32_error']:
                raise ValueError('Causal/cache/padding invariant failed')
            del memory, full, chunks, cache, fresh, fresh_first, full_batch
            record()
            opt = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=0.)
            model.train()
            monitor.take(); train_started=time.monotonic(); intervals=[]
            for step in range(1,plan['steps']+1):
                deadline()
                opt.zero_grad(set_to_none=True)
                logits = model(inputs,sequence)
                loss = event_loss(logits,sequence)
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite supervised loss')
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                if step == 1:
                    row['gradient_paths'] = {}
                    for name,p in model.named_parameters():
                        if p.grad is None or not bool(torch.isfinite(p.grad).all()):
                            raise ValueError('Absent/nonfinite gradient: '+name)
                        group = name.split('.')[0]
                        row['gradient_paths'][group] = row['gradient_paths'].get(group,0.)+float(p.grad.float().square().sum())
                    if any(x == 0 for x in row['gradient_paths'].values()):
                        raise ValueError('Zero gradient branch')
                opt.step();row['updates']=step
                if step == 1 or step % 10 == 0:
                    util=monitor.take();intervals.append(util)
                    means=[v['util_mean'] for v in intervals[-8:] if v['util_mean'] is not None]
                    if time.monotonic()-train_started>=480 and means and sum(means)/len(means)<70:
                        raise RuntimeError('Original 70% utilization guard after 480s')
                    row['curve'].append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(grad),
                        train_wall_s=time.monotonic()-train_started,gpu=util))
                    print(json.dumps(dict(candidate=row['name'],**row['curve'][-1])),flush=True)
                    record()
                del logits,loss
            row['train_wall_s']=time.monotonic()-train_started
            del opt
            model.eval()
            with torch.no_grad():
                final=model(inputs,sequence)
                row['final']=metrics(final,sequence)
                row['final']['request_mean_ce']=float(event_loss(final,sequence))
            del final
            row['generations']=[]
            for i,item in enumerate(manifest['records']):
                deadline()
                pred=generate(model,{k:v[i:i+1] for k,v in inputs.items()},EventRequest(**item['request']),plan['max_generation_tokens'])
                truth=sorted(item['target_events'],key=lambda e:(e['onset_us'],e['end_us'],e['type'],e['lane']))
                pred.update(chart_id=item['chart_id'],exact_multiset=pred['events']==truth,target_events=truth)
                row['generations'].append(pred)
            row['passed']=(row['final']['request_mean_ce']<=plan['max_ce']
                and min(v['accuracy'] for v in row['final']['fields'].values())>=plan['min_field_accuracy']
                and all(g['complete'] and g['exact_multiset'] for g in row['generations']))
            row['complete']=True
            # Save model only as diagnostic evidence, never promote a health fit.
            checkpoint=output/(row['name']+'-diagnostic.pt')
            torch.save(dict(model=model.state_dict(),config=asdict(config),vocabulary=manifest['vocabulary'],
                            diagnostic_only=True,optimizer_updates=row['updates']),checkpoint)
            row['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            del model
            gc.collect()
        except Exception as exc:
            row['error']=f'{type(exc).__name__}: {exc}'
            record()
            # Infrastructure/numerical errors need diagnosis, not blind repetition.
            raise
        finally:
            row['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
            row['peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
            monitor.close();record()
    report['complete']=True
    report['passed']=all(c['passed'] for c in report['candidates'])
    record()
    return report
