"""Freeze and execute a bounded real-input full-stack Mamba-2 health gate."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.fullstack_mamba import FullStackMamba, FullStackMambaConfig
from beatfcn.model.full_event_model import FullEventModel, FullEventConfig, event_loss
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.train.local_request_panel import LocalRequestPanel
from beatfcn.train.local_panel import collate
from beatfcn.train.local_audio_audit import tensor_health
from beatfcn.train.reference_completion import completion_row
from scripts.run_reference_completion import read, write, sha

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'bench/rhythm-v2/fullstack-mamba-health-v1.json'


def make_pair():
    """Fresh backbones, identical compatible input/output parameters, fresh Adam."""
    base = read('bench/rhythm-v2/completion-quality-v1.json')
    old = torch.load(ROOT / base['parent'], map_location='cpu', weights_only=False)
    config = old['config']
    del old
    torch.manual_seed(20261026)
    transformer = FullEventModel(FullEventConfig(**config))
    torch.manual_seed(20261026)
    mamba = FullStackMamba(FullStackMambaConfig())
    prefixes = ('layer_mix', 'local_projection', 'band_embedding', 'song_projection',
                'mel_projection', 'mel_source', 'reference_projection', 'control_projection',
                'control_ids', 'empty_reference', 'source_ids', 'mode_ids', 'token_embedding',
                'field_embedding', 'final_norm', 'output')
    state = mamba.state_dict()
    copied = []
    for name, value in transformer.state_dict().items():
        if name.split('.')[0] in prefixes and name in state and state[name].shape == value.shape:
            state[name] = value.clone(); copied.append(name)
    mamba.load_state_dict(state)
    return transformer, mamba, copied


def prepare():
    if (ROOT / PLAN).exists():
        raise FileExistsError(PLAN)
    previous = read('bench/rhythm-v2/reference-completion-v1.json')
    sources = {k: v for k, v in previous['source_sha256'].items()
               if k != 'scripts/run_reference_completion.py'}
    for path in ('scripts/probe_fullstack_mamba.py', 'beatfcn/model/mamba2_reference.py',
                 'beatfcn/model/fullstack_mamba.py', 'tests/test_mamba2_reference.py',
                 'tests/test_fullstack_mamba.py'):
        sources[path] = sha(path)
    for path in (ROOT / 'inputs/fullstack_mamba_v1/upstream').iterdir():
        sources[str(path.relative_to(ROOT))] = sha(path.relative_to(ROOT))
    baseline, model, copied = make_pair()
    counts = dict(transformer=sum(p.numel() for p in baseline.parameters()),
                  fullstack_mamba=sum(p.numel() for p in model.parameters()))
    if not .95 <= counts['fullstack_mamba'] / counts['transformer'] <= 1.05:
        raise ValueError('Capacity matching outside five percent')
    plan = dict(id='fullstack-mamba-health-v1', tape=previous['tape'], features=previous['features'],
        cases=previous['cases'], source_sha256=sources, seed=20261026,
        config=asdict(model.c), parameters=counts, shared_initial_parameters=copied,
        steps=100, learning_rate=.001, max_request_ce=.05, max_wall_s=1800,
        deadline_unix=1793037775., token_cap=512, max_fp32_error=1e-4,
        report='reports/fullstack_mamba_health_v1.json', artifacts='artifacts/fullstack-mamba-v1/health',
        backend='explicit official-Mamba2Simple/SSD-equation PyTorch reference, FP32; not fused CUDA',
        architecture='Bidirectional Mamba-2 local/song/reference encoders; causal Mamba-2 context-prefix fusion and event decoder. Frozen Pupu/mel remain external.',
        cloud_spend_usd=0, quality_training_started=False)
    write(PLAN, plan)
    print(json.dumps(dict(prepared=True, parameters=counts, cases=len(plan['cases']))))


def run():
    plan = read(PLAN)
    if (ROOT / plan['report']).exists():
        raise FileExistsError(plan['report'])
    started = time.monotonic()
    report = dict(id=plan['id'], complete=False, passed=False, status='verifying',
        optimizer_updates=0, new_cloud_spend_usd=0, quality_passed=False,
        plan_sha256=sha(PLAN), cases=[], curve=[], final=[], generation=[], backend=plan['backend'])
    def save():
        report['wall_s'] = time.monotonic() - started
        write(plan['report'], report)
    def deadline():
        if time.monotonic() - started >= plan['max_wall_s'] or time.time() >= plan['deadline_unix']:
            raise TimeoutError('Frozen full-stack Mamba health bound')
    try:
        for path, expected in plan['source_sha256'].items():
            if sha(path) != expected:
                raise ValueError('Frozen input/code changed: ' + path)
        if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
            raise RuntimeError('Explicit native MPS backend required')
        torch.set_num_threads(2); torch.mps.set_per_process_memory_fraction(.6)
        panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
        cpu, requests, metadata = [], [], []
        for case in plan['cases']:
            value, request, meta = completion_row(panel, case['index'], case['task'], reference=True, seed=plan['seed'])
            pair = collate([value])
            if meta['split'] != 'train' or meta['chart_id'] != case['chart_id']:
                raise ValueError('Health case identity changed')
            report['cases'].append(dict(**case, metadata=meta,
                tensors={s: {k: tensor_health(v.numpy()) for k, v in values.items()}
                         for s, values in (('inputs', pair[0]), ('sequence', pair[1]))}))
            cpu.append(pair); requests.append(request); metadata.append(meta)
        baseline, model, shared = make_pair()
        del baseline
        report['parameters'] = sum(p.numel() for p in model.parameters())
        report['shared_initial_parameters'] = shared
        model.eval()
        inp, seq = cpu[0]
        with torch.no_grad():
            tick = time.monotonic(); cpu_logits = model(inp, seq)[seq['valid']]
            report['first_cpu_forward_s'] = time.monotonic() - tick
            model.to('mps')
            mi, ms = ({k: v.to('mps') for k, v in section.items()} for section in cpu[0])
            tick = time.monotonic(); mps_logits = model(mi, ms).cpu()[seq['valid']]
            report['first_mps_forward_s'] = time.monotonic() - tick
            parity = float((cpu_logits - mps_logits).abs().max())
            report['cpu_mps_valid_logits_max_abs'] = parity
            if parity > plan['max_fp32_error']:
                raise ValueError('Real full-stack CPU/MPS numerical parity failed')
            # Cached prefix prefill must match the real whole forward, not a short empty prefix.
            n = min(32, int(seq['valid'].sum()))
            memory = model.encode(mi); cache = model.start_cache(memory)
            steps = [model.decode(*(ms[k][:, i:i + 1] for k in ('tokens', 'fields', 'times', 'valid')),
                                  memory, cache=cache).cpu() for i in range(n)]
            step_error = float((torch.cat(steps, 1)[0] - mps_logits[:n]).abs().max())
            report['actual_prefix_cache_max_abs'] = step_error
            if step_error > plan['max_fp32_error']:
                raise ValueError('Real context-prefix full/cache parity failed')
            del memory, cache, steps, mi, ms, cpu_logits, mps_logits
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=0.)
        model.train()
        report['status'] = 'real_input_and_backend_passed'; save()
        print(json.dumps({k: report[k] for k in ('status', 'first_cpu_forward_s', 'first_mps_forward_s', 'cpu_mps_valid_logits_max_abs', 'actual_prefix_cache_max_abs')}), flush=True)
        for step in range(1, plan['steps'] + 1):
            deadline(); optimizer.zero_grad(set_to_none=True)
            total = 0.
            for pair in cpu:
                inp, seq = ({k: v.to('mps') for k, v in section.items()} for section in pair)
                loss = event_loss(model(inp, seq), seq) / len(cpu)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite actual supervised loss')
                loss.backward(); total += float(loss.detach())
                del inp, seq, loss
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if step == 1:
                gradients = Counter()
                for name, p in model.named_parameters():
                    if p.grad is None or not torch.isfinite(p.grad).all():
                        raise ValueError('Missing/nonfinite gradient: ' + name)
                    gradients[name.split('.')[0]] += float(p.grad.square().sum())
                report['gradient_norms'] = {k: v**.5 for k, v in gradients.items()}
                for key in ('local_projection', 'song_projection', 'mel_projection', 'reference_projection',
                            'time_projection', 'encoders', 'decoder', 'output'):
                    if gradients[key] <= 0:
                        raise ValueError('Zero gradient branch: ' + key)
            optimizer.step(); report['optimizer_updates'] = step
            if step == 1 or step % 10 == 0:
                report['curve'].append(dict(step=step, ce=total, preclip_norm=float(norm)))
                report['status'] = 'fixed_batch_training'; save()
                print(json.dumps(dict(step=step, ce=total, wall_s=report['wall_s'])), flush=True)
        model.eval()
        with torch.no_grad():
            for case, pair in zip(plan['cases'], cpu):
                inp, seq = ({k: v.to('mps') for k, v in section.items()} for section in pair)
                logits = model(inp, seq)
                legal = logits.masked_fill(~seq['allowed'], -1e9)
                heads = {}
                for field in seq['fields'][seq['valid']].unique().tolist():
                    mask = seq['valid'] & (seq['fields'] == field)
                    heads[str(field)] = dict(count=int(mask.sum()), accuracy=float((legal.argmax(-1)[mask] == seq['targets'][mask]).float().mean()))
                report['final'].append(dict(index=case['index'], task=case['task'],
                                           ce=float(event_loss(logits, seq)), heads=heads))
        del optimizer
        model.to('cpu')
        for case, pair, request, meta in zip(plan['cases'], cpu, requests, metadata):
            deadline()
            pred = generate_batch(model, pair[0], [request], plan['token_cap'], deadline=started + plan['max_wall_s'])[0]
            exact = pred['complete'] and pred['tokens'] == pair[1]['targets'][0][pair[1]['valid'][0]].tolist()
            artifact = f"{plan['artifacts']}/{case['index']}-{case['task']}.json"
            write(artifact, dict(metadata=meta, prediction=pred, exact=exact))
            report['generation'].append(dict(index=case['index'], task=case['task'], complete=pred['complete'],
                                             exact=exact, artifact=artifact,
                                             types=dict(Counter(e['type'] for e in pred['events']))))
            save()
        report.update(complete=True, status='completed', passed=all(x['ce'] <= plan['max_request_ce'] for x in report['final'])
                      and len(report['generation']) == 6 and all(x['exact'] for x in report['generation']))
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
    print(json.dumps(dict(passed=report['passed'], updates=report['optimizer_updates'], wall_s=report['wall_s'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()
