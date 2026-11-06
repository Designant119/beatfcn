"""D88 real tensor, cache and fixed-batch health; never a quality promotion."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import torch

from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.model.relative_event_generate import generate_batch
from beatfcn.train.local_audio_audit import tensor_health
from beatfcn.train.local_request_panel import LocalRequestPanel, event_request
from beatfcn.train.local_shapes import pad_input_shapes
from beatfcn.train.relative_request_panel import RelativeRequestPanel
from scripts.probe_local_sequence_likelihood import device_batch, unified_metrics
from scripts.run_local_event_experiment import memory
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def cache_proof(model, inp, seq, deadline):
    valid = seq['valid']
    if valid.shape[0] != 1 or not bool(valid[0, 0]) or bool((valid[:, 1:] & ~valid[:, :-1]).any()):
        raise ValueError('Cache proof requires one nonempty right-padded request')
    count = int(valid.sum())
    encoded = model.encode(inp)
    keys = ['tokens', 'fields', 'times', 'valid']
    full = model.decode(*(seq[k] for k in keys), encoded)
    cache = model.start_cache(encoded)
    pieces = []
    # Full training forward retains collator padding. Incremental inference
    # never appends padding to the cache; compare every actual target position.
    for i in range(count):
        deadline()
        pieces.append(model.decode(*(seq[k][:, i:i+1] for k in keys), encoded, cache=cache))
    cached = torch.cat(pieces, 1)
    padded = full.shape[1]-count
    full = full[:, :count]
    allowed = seq['allowed'][:, :count]
    difference = float((full-cached).abs().max())
    prediction_equal = bool(torch.equal(full.masked_fill(~allowed, -1e9).argmax(-1),
                                        cached.masked_fill(~allowed, -1e9).argmax(-1)))
    return dict(tokens=count, padding_not_queried=padded, maximum_logit_error=difference,
                legal_argmax_equal=prediction_equal)


def run():
    plan_path = ROOT/'bench/rhythm-v2/d88-relative-head-probe-v3.json'
    plan = json.loads(plan_path.read_text())
    output = ROOT/plan['output']
    if output.exists():
        raise FileExistsError('Probe already has evidence; do not overwrite or blind-retry')
    for path, expected in plan['source_sha256'].items():
        if sha(ROOT/path) != expected:
            raise ValueError('Frozen source changed: '+path)
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise ValueError('Native MPS required')
    torch.set_num_threads(2)
    torch.mps.set_per_process_memory_fraction(.6)
    started = time.monotonic()
    report = dict(id=plan['id'], plan_sha256=sha(plan_path), complete=False, passed=False,
        optimizer_updates=0, new_cloud_compute_calls=0, status='preparing', tensors=[],
        cache_proofs=[], initial=[], curve=[], final=[], generation=[],
        scope='Real fixed-batch implementation health only; no held-out quality or condition claim')

    def save():
        report.update(wall_s=time.monotonic()-started, memory=memory('mps'))
        write_json(output, report)
        print(json.dumps(dict(status=report['status'], updates=report['optimizer_updates'],
                             wall_s=report['wall_s'])), flush=True)

    def deadline():
        if time.monotonic()-started > plan['max_wall_s']:
            raise TimeoutError('Frozen local probe wall limit')

    try:
        old = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
        panel = RelativeRequestPanel(ROOT, plan['tape'], plan['features'])
        cpu, requests, targets = [], [], []
        for case in plan['cases']:
            deadline()
            index = case['record_index']
            row = panel.records[index]
            if row['split'] != 'train' or row['chart_id'] != case['chart_id']:
                raise ValueError('Frozen train-only request identity differs')
            controls = torch.tensor(row['controls']).numpy().copy()
            controls[[i for i in range(len(controls)) if i not in case['specified_fields']], :3] = 0
            inp, seq = panel.batch([index], controls=[controls])
            old_inp, old_seq = old.batch([index], controls=[controls])
            if inp.keys() != old_inp.keys() or any(not torch.equal(v, old_inp[k]) for k, v in inp.items()):
                raise ValueError('Coordinate intervention changed audio, reference or condition inputs')
            if any(not torch.equal(seq[k], old_seq[k]) for k in ('valid', 'fields')):
                raise ValueError('Coordinate intervention changed token count or grammar fields')
            if not bool(seq['allowed'].gather(2, seq['targets'][..., None]).squeeze(-1)[seq['valid']].all()):
                raise ValueError('Actual teacher target is illegal')
            inp = pad_input_shapes(inp, plan['input_shapes'])
            cpu.append((inp, seq))
            requests.append(event_request(row['request']))
            targets.append(sorted(panel.targets(index)))
            report['tensors'].append(dict(index=index, chart_id=row['chart_id'],
                audio_reference_condition_inputs_unchanged=True, grammar_fields_and_length_unchanged=True,
                target_count=len(targets[-1]),
                tensors={section:{k:tensor_health(v.numpy()) for k,v in values.items()}
                         for section,values in [('inputs', inp), ('sequence', seq)]}))
        del old, panel
        saved = torch.load(ROOT/plan['parent'], map_location='cpu', weights_only=False)
        if saved['updates'] != 2 or saved['config'] != plan['config']:
            raise ValueError('Parent differs from common step2')
        model = FullEventModel(FullEventConfig(**plan['config']))
        model.load_state_dict(saved['model'])
        model.eval()
        # The new coordinate state uses the same network; check the real complete
        # teacher sequences on both backends, not just tiny synthetic tensors.
        for backend in ['cpu', 'mps']:
            model.to(backend)
            for case, pair in zip(plan['cases'], cpu, strict=True):
                deadline()
                inp, seq = device_batch(pair, backend)
                proof = cache_proof(model, inp, seq, deadline)
                proof.update(backend=backend, record_index=case['record_index'])
                proof['passed'] = (proof['maximum_logit_error'] <= plan['cache_atol']
                                   and proof['legal_argmax_equal'])
                report['cache_proofs'].append(proof)
                save()
                if not proof['passed']:
                    raise ValueError('Real cached/full parity failed')
                del inp, seq
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=0.)
        optimizer.load_state_dict(deepcopy(saved['optimizer']))
        if any(float(state['step']) != 2 for state in optimizer.state.values()):
            raise ValueError('Optimizer parent did not restore step2')
        if any(group['lr'] != plan['learning_rate'] for group in optimizer.param_groups):
            raise ValueError('Parent optimizer learning rate differs')
        with torch.no_grad():
            for pair in cpu:
                inp, seq = device_batch(pair, 'mps')
                report['initial'].append(unified_metrics(model(inp, seq), seq))
                del inp, seq
        torch.set_rng_state(saved['cpu_rng'])
        torch.mps.set_rng_state(saved['mps_rng'])
        del saved
        report['status'] = 'zero_update_passed'
        save()
        for step in range(1, plan['steps']+1):
            deadline()
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for pair in cpu:
                inp, seq = device_batch(pair, 'mps')
                value = event_loss(model(inp, seq), seq)/len(cpu)
                value.backward()
                total += float(value.detach())
                del inp, seq, value
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if step == 1:
                gradients = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if not bool(torch.isfinite(param.grad).all()):
                            raise ValueError('Nonfinite parameter gradient')
                        group = name.split('.')[0]
                        gradients[group] = gradients.get(group, 0.)+float(param.grad.square().sum())
                report['initial_gradient_norm_by_module'] = {k:v**.5 for k,v in gradients.items()}
                for key in ['local_projection', 'mel_projection', 'token_embedding', 'decoder', 'output']:
                    if gradients.get(key, 0.) <= 0:
                        raise ValueError('No gradient reaches required module '+key)
            optimizer.step()
            report['optimizer_updates'] = step
            if step % 10 == 0:
                report['curve'].append(dict(step=step, request_mean_ce=total, preclip_gradient_norm=float(norm)))
                report['status'] = 'fixed_batch_training'
                save()
        model.eval()
        with torch.no_grad():
            for pair in cpu:
                inp, seq = device_batch(pair, 'mps')
                report['final'].append(unified_metrics(model(inp, seq), seq))
                del inp, seq
        checkpoint = ROOT/plan['checkpoint']
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open('xb') as stream:
            torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                config=asdict(model.c), parent=plan['parent'], diagnostic_only=True, auto_resume=False,
                probe_updates=plan['steps'], plan_sha256=sha(plan_path)), stream)
        report['checkpoint'] = dict(path=plan['checkpoint'], sha256=sha(checkpoint))
        del optimizer
        model.to('cpu')
        for case, pair, request, target in zip(plan['cases'], cpu, requests, targets, strict=True):
            deadline()
            generated = generate_batch(model, pair[0], [request], plan['generation_token_cap'],
                deadline=started+plan['max_wall_s'])[0]
            target_tokens = pair[1]['targets'][0][pair[1]['valid'][0]].tolist()
            generated.update(record_index=case['record_index'],
                exact_tokens=generated['complete'] and generated['tokens'] == target_tokens,
                exact_events=generated['events'] == [asdict(e) for e in target])
            report['generation'].append(generated)
        report.update(complete=True, status='completed',
            passed=(all(r['request_mean_ce'] <= plan['max_request_mean_ce'] and r['bos_correct'] for r in report['final'])
                    and all(r['exact_tokens'] and r['exact_events'] for r in report['generation'])),
            parent_unchanged=sha(ROOT/plan['parent']) == plan['source_sha256'][plan['parent']])
        if not report['parent_unchanged']:
            raise ValueError('Common parent changed')
        save()
    except Exception as exc:
        report.update(status='failed', passed=False, error=f'{type(exc).__name__}: {exc}')
        save()
        raise


if __name__ == '__main__':
    run()
