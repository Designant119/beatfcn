"""Bounded real-input health/fit probe of the full-sized event mainline."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beatfcn.model.full_event_codec import BOS, FIELDS, EventRequest, EventState, EventVocabulary
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_batch(folder, device):
    manifest = json.loads((folder/'manifest.json').read_text())
    if sha(folder/'inputs.npz') != manifest['tensor_sha256']:
        raise ValueError('Prepared tensor identity changed')
    arrays = np.load(folder/'inputs.npz')
    n = len(manifest['records'])
    batches = {}
    for section in ('input', 'sequence'):
        batch = {}
        for key in [k.split(':', 2)[2] for k in arrays.files if k.startswith(f'0:{section}:')]:
            values = [arrays[f'{i}:{section}:{key}'] for i in range(n)]
            if key in ('mode', 'controls'):
                joined = np.stack(values)
            elif key == 'local':
                joined = np.zeros((n, values[0].shape[0], max(v.shape[1] for v in values), *values[0].shape[2:]), values[0].dtype)
                for i, v in enumerate(values):
                    joined[i, :, :v.shape[1]] = v
            else:
                joined = np.zeros((n, max(len(v) for v in values), *values[0].shape[1:]), values[0].dtype)
                for i, v in enumerate(values):
                    joined[i, :len(v)] = v
            batch[key] = torch.from_numpy(joined).to(device)
        if section == 'sequence':
            batch['allowed'][..., 0] |= ~batch['valid']
            if not bool((batch['allowed'].gather(-1, batch['targets'][..., None]).squeeze(-1) | ~batch['valid']).all()):
                raise ValueError('A real target is outside the prefix-only legal support')
        batches[section] = batch
    return manifest, batches['input'], batches['sequence']


def metrics(logits, seq):
    scores = logits.float().masked_fill(~seq['allowed'], -1e9)
    losses = F.cross_entropy(scores.flatten(0, 1), seq['targets'].flatten(), reduction='none').view_as(seq['targets'])
    correct = scores.argmax(-1) == seq['targets']
    result = dict(ce=float((losses * seq['valid']).sum() / seq['valid'].sum()),
                  token_accuracy=float((correct & seq['valid']).sum() / seq['valid'].sum()), fields={})
    for i, name in enumerate(FIELDS):
        mask = (seq['fields'] == i) & seq['valid']
        if mask.any():
            result['fields'][name] = dict(count=int(mask.sum()), ce=float(losses[mask].mean()),
                                          accuracy=float(correct[mask].float().mean()))
    return result


@torch.no_grad()
def generate(model, inputs, request, max_tokens):
    state = EventState(request, model.vocab)
    memory = model.encode(inputs)
    cache = model.start_cache(memory)
    device = inputs['local'].device
    for _ in range(max_tokens):
        field, t = state.query()
        logits = model.decode(torch.tensor([[state.tokens[-1] if state.tokens else BOS]], device=device),
            torch.tensor([[field]], device=device), torch.tensor([[t]], device=device),
            torch.ones(1, 1, dtype=torch.bool, device=device), memory, cache=cache)[0, 0]
        support = torch.tensor(state.allowed_tokens(), device=device)
        if not len(support):
            raise ValueError('Grammar reached a dead end')
        token = int(support[logits[support].argmax()])
        state.apply(token)
        if state.done:
            break
    return dict(complete=state.done, events=[asdict(e) for e in sorted(state.events)], tokens=state.tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    result_path = ROOT / plan['result']
    if result_path.exists():
        raise FileExistsError(result_path)
    for name, expected in plan['input_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Input changed: {name}')
    torch.set_num_threads(plan['threads'])
    torch.manual_seed(plan['seed'])
    started = time.perf_counter()
    result = dict(id=plan['id'], complete=False, passed=False, optimizer_updates=0,
        phase='start', plan_sha256=sha(args.plan), new_cloud_spend_usd=0,
        formal_training=False, quality_improvement_proven=False)

    def save(phase):
        result.update(phase=phase, wall_s=time.perf_counter()-started,
                      peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(phase=phase, updates=result['optimizer_updates'], wall_s=result['wall_s'])), flush=True)

    save('loading')
    try:
        manifest, inputs, seq = load_batch(ROOT/plan['data'], plan['device'])
        config = FullEventConfig(**plan['config'])
        model = FullEventModel(config, EventVocabulary(**manifest['vocabulary'])).to(plan['device'])
        result.update(config=asdict(config), parameters=sum(p.numel() for p in model.parameters()),
                      real_charts=[r['chart_id'] for r in manifest['records']],
                      input_shapes={k:list(v.shape) for k,v in inputs.items()},
                      sequence_shapes={k:list(v.shape) for k,v in seq.items()},
                      torch_version=torch.__version__, seed=plan['seed'], device=plan['device'])
        model.eval()
        t0 = time.perf_counter()
        logits = model(inputs, seq)
        result['initial'] = metrics(logits.detach(), seq)
        loss = event_loss(logits, seq)
        loss.backward()
        result['initial_forward_backward_s'] = time.perf_counter()-t0
        result['gradient_summary'] = {}
        for name, module in [('local_projection',model.local_projection), ('song_projection',model.song_projection),
                             ('reference_projection',model.reference_projection), ('controls',model.control_projection),
                             *[(f'encoder_{i}',m) for i,m in enumerate(model.encoders)], ('decoder',model.decoder), ('output',model.output)]:
            parameters = list(module.parameters())
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError(f'Absent or nonfinite gradient: {name}')
            norm = sum(float(p.grad.square().sum()) for p in parameters) ** .5
            result['gradient_summary'][name] = dict(norm=norm, finite=True)
            if norm == 0:
                raise ValueError(f'Zero gradient path: {name}')
        model.zero_grad(set_to_none=True)
        del logits, loss
        save('health_checked')
        if plan['steps']:
            optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=0.)
            model.train()
            result['curve'] = []
            for step in range(1, plan['steps']+1):
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs, seq)
                loss = event_loss(logits, seq)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), plan['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                result['optimizer_updates'] = step
                if step == 1 or step % 10 == 0:
                    result['curve'].append(dict(step=step, loss=float(loss.detach()), gradient_norm=float(gradient)))
                    save('fitting')
                del logits, loss
            del optimizer
            model.eval()
            with torch.no_grad():
                logits = model(inputs, seq)
                result['final'] = metrics(logits, seq)
                result['teacher_logits_sha256'] = hashlib.sha256(logits.cpu().numpy().tobytes()).hexdigest()
            del logits
            result['generations'] = []
            for i, row in enumerate(manifest['records']):
                request = EventRequest(**row['request'])
                prediction = generate(model, {k:v[i:i+1] for k,v in inputs.items()}, request, plan['generation_token_cap'])
                prediction['chart_id'] = row['chart_id']
                target = sorted(row['target_events'], key=lambda e:(e['onset_us'],e['end_us'],e['type'],e['lane']))
                prediction['target_events'] = target
                prediction['exact_multiset'] = prediction['events'] == target
                result['generations'].append(prediction)
                save('generation_checked')
            artifact = ROOT/plan['checkpoint']
            if artifact.exists():
                raise FileExistsError(artifact)
            artifact.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(model=model.state_dict(),config=asdict(config),vocabulary=manifest['vocabulary'],
                optimizer_updates=result['optimizer_updates'],diagnostic_only=True,plan_sha256=result['plan_sha256']),artifact)
            result['checkpoint_sha256'] = sha(artifact)
            result['passed'] = (result['final']['ce'] <= plan['maximum_ce']
                and min(m['accuracy'] for m in result['final']['fields'].values()) >= plan['minimum_field_accuracy']
                and all(g['complete'] and g['exact_multiset'] for g in result['generations']))
        else:
            result['passed'] = True
        result['complete'] = True
        save('complete')
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        save('failed')
        raise


if __name__ == '__main__':
    main()
