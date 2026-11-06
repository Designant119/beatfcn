"""Paired local implementation probe for one frozen sequence-loss intervention."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.full_event_generate import generate_batch
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.train.local_audio_audit import tensor_health
from beatfcn.train.local_request_panel import LocalRequestPanel, event_request
from beatfcn.train.local_shapes import pad_input_shapes
from beatfcn.train.sequence_likelihood import ModeSequenceNLL
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def device_batch(pair, device):
    return tuple({k:v.to(device) for k,v in section.items()} for section in pair)


def unified_metrics(logits, sequence):
    selected = logits.float().masked_fill(~sequence['allowed'], -1e9)
    predictions = selected.argmax(-1)
    valid = sequence['valid']
    fields = sequence['fields']
    from beatfcn.model.full_event_codec import FIELDS
    return dict(request_mean_ce=float(event_loss(logits, sequence).detach()),
        bos_correct=bool(predictions[0, 0] == sequence['targets'][0, 0]),
        bos_end_probability=float(torch.softmax(selected[0, 0], -1)[2].detach()),
        token_accuracy=float((predictions[valid] == sequence['targets'][valid]).float().mean()),
        fields={name: dict(count=int((valid & (fields == i)).sum()),
            correct=int(((predictions == sequence['targets']) & valid & (fields == i)).sum()))
            for i, name in enumerate(FIELDS) if bool((valid & (fields == i)).any())})


def run():
    plan_path = ROOT/'bench/rhythm-v2/d86-sequence-likelihood-probe-v2.json'
    plan = json.loads(plan_path.read_text())
    output = ROOT/plan['output']
    if output.exists():
        raise FileExistsError(output)
    for path, expected in plan['source_sha256'].items():
        if sha(ROOT/path) != expected:
            raise ValueError('Frozen input changed: '+path)
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise ValueError('Native MPS required')
    torch.set_num_threads(2)
    torch.mps.set_per_process_memory_fraction(.6)
    started = time.monotonic()
    report = dict(id=plan['id'], complete=False, passed=False, plan_sha256=sha(plan_path),
        optimizer_updates=0, new_cloud_compute_calls=0, tensors=[], zero_update_proofs=[], arms={},
        status='preparing', scope='Real fixed batch implementation health only; never a quality winner')

    def save():
        report['elapsed_s'] = time.monotonic()-started
        write_json(output, report)
        print(json.dumps(dict(status=report['status'], optimizer_updates=report['optimizer_updates'],
                             elapsed_s=report['elapsed_s'])), flush=True)

    def deadline():
        if time.monotonic()-started > plan['max_wall_s']:
            raise TimeoutError('Frozen probe wall-clock limit')

    try:
        panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
        cpu = []
        requests = []
        for case in plan['cases']:
            row = panel.records[case['record_index']]
            if row['split'] != 'train' or row['chart_id'] != case['chart_id']:
                raise ValueError('Probe request identity mismatch')
            control = np.asarray(row['controls'], np.float32).copy()
            control[[i for i in range(len(control)) if i not in case['specified_fields']], :3] = 0
            inp, seq = panel.batch([case['record_index']], controls=[control])
            inp = pad_input_shapes(inp, plan['input_shapes'])
            cpu.append((inp, seq)); requests.append(event_request(row['request']))
            report['tensors'].append(dict(index=case['record_index'],
                tensors={section:{k:tensor_health(v.numpy()) for k,v in values.items()}
                         for section,values in [('inputs', inp), ('sequence', seq)]}))
        objective = ModeSequenceNLL(**plan['normalizers']).to('mps')

        def fresh():
            saved = torch.load(ROOT/plan['parent'], map_location='cpu', weights_only=False)
            if saved['updates'] != 2 or saved['config'] != plan['config']:
                raise ValueError('Common parent differs from frozen step 2')
            model = FullEventModel(FullEventConfig(**plan['config'])).to('mps')
            model.load_state_dict(saved['model'])
            opt = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=0.)
            opt.load_state_dict(deepcopy(saved['optimizer']))
            if any(float(state['step']) != 2 for state in opt.state.values()):
                raise ValueError('Optimizer did not start at common step 2')
            if any(group['lr'] != plan['learning_rate'] for group in opt.param_groups):
                raise ValueError('Parent optimizer learning rate differs from plan')
            torch.set_rng_state(saved['cpu_rng']); torch.mps.set_rng_state(saved['mps_rng'])
            return model, opt

        # Compare exact intended per-request scaling and every parameter gradient
        # on real MPS tensors before any optimizer update.
        model, optimizer = fresh()
        model.train()
        for case, pair in zip(plan['cases'], cpu, strict=True):
            inp, seq = device_batch(pair, 'mps')
            logits = model(inp, seq)
            direct = objective(logits, seq, inp['mode'])
            scale = seq['valid'].sum()/objective.mean_tokens.index_select(0, inp['mode'])[0]
            expected = event_loss(logits, seq)*scale
            params = tuple(model.parameters())
            gd = torch.autograd.grad(direct, params, retain_graph=True, allow_unused=True)
            ge = torch.autograd.grad(expected, params, allow_unused=True)
            delta = max(float((a-b).abs().max()) for a,b in zip(gd,ge) if a is not None and b is not None)
            norm = sum(float(b.square().sum()) for b in ge if b is not None)**.5
            errnorm = sum(float((a-b).square().sum()) for a,b in zip(gd,ge) if a is not None and b is not None)**.5
            proof = dict(record_index=case['record_index'],
                loss_difference=abs(float(direct.detach())-float(expected.detach())),
                gradient_max_error=delta, gradient_relative_error=errnorm/max(norm, 1e-12),
                reference_gradient_norm=norm, coefficient_scale=float(scale),
                same_unused_parameters=all((a is None) == (b is None) for a,b in zip(gd,ge)))
            proof['passed'] = (proof['same_unused_parameters'] and np.isfinite(norm) and norm > 0
                and proof['loss_difference'] <= 1e-5 and delta <= 1e-5 and proof['gradient_relative_error'] <= 1e-4)
            report['zero_update_proofs'].append(proof)
            if not proof['passed']:
                raise ValueError('Real sequence objective equivalence gate failed')
            del inp, seq, logits, direct, expected, gd, ge
        del model, optimizer
        report['status'] = 'zero_update_passed'
        save()

        for arm in ('request_mean_control', 'constant_mode_candidate'):
            deadline()
            model, optimizer = fresh()
            model.train()
            record = dict(updates=0, initial=[], curve=[], final=[], generation=[], clip_steps=0)
            report['arms'][arm] = record
            with torch.no_grad():
                for pair in cpu:
                    inp, seq = device_batch(pair, 'mps')
                    record['initial'].append(unified_metrics(model(inp, seq), seq))
                    del inp, seq
            if arm == 'constant_mode_candidate':
                control_initial = report['arms']['request_mean_control']['initial']
                if any(abs(a['request_mean_ce']-b['request_mean_ce']) > 1e-7 for a,b in zip(record['initial'],control_initial)):
                    raise ValueError('Paired arms did not start with identical predictions')
            # Diagnostic forwards use no dropout (frozen config=0); resetting RNG
            # again also prevents audit ordering from changing training RNG.
            saved_rng = torch.load(ROOT/plan['parent'], map_location='cpu', weights_only=False)
            torch.set_rng_state(saved_rng['cpu_rng']); torch.mps.set_rng_state(saved_rng['mps_rng'])
            del saved_rng
            for step in range(1, plan['steps']+1):
                deadline()
                optimizer.zero_grad(set_to_none=True)
                objective_total = 0.
                for pair in cpu:
                    inp, seq = device_batch(pair, 'mps')
                    logits = model(inp, seq)
                    value = (event_loss(logits, seq) if arm == 'request_mean_control'
                             else objective(logits, seq, inp['mode']))/len(cpu)
                    value.backward(); objective_total += float(value.detach())
                    del inp, seq, logits, value
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                record['clip_steps'] += float(norm) > 1.
                optimizer.step()
                record['updates'] = step
                report['optimizer_updates'] += 1
                if step % 10 == 0:
                    record['curve'].append(dict(step=step, objective=objective_total, preclip_gradient_norm=float(norm)))
                    report['status'] = arm
                    save()
            model.eval()
            with torch.no_grad():
                for pair in cpu:
                    inp, seq = device_batch(pair, 'mps')
                    record['final'].append(unified_metrics(model(inp, seq), seq))
                    del inp, seq
            terminal = ROOT/plan['checkpoint_dir']/(arm+'.pt')
            terminal.parent.mkdir(parents=True, exist_ok=True)
            with terminal.open('xb') as stream:
                torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    config=asdict(model.c), diagnostic_only=True, auto_resume=False,
                    parent=plan['parent'], probe_updates=plan['steps'], plan_sha256=sha(plan_path)), stream)
            record['checkpoint'] = dict(path=str(terminal.relative_to(ROOT)), sha256=sha(terminal))
            del optimizer
            model.to('cpu')
            for case, pair, request in zip(plan['cases'], cpu, requests, strict=True):
                deadline()
                generated = generate_batch(model, pair[0], [request], plan['generation_token_cap'],
                    deadline=started+plan['max_wall_s'])[0]
                seq = pair[1]
                target = seq['targets'][0][seq['valid'][0]].tolist()
                generated.update(record_index=case['record_index'],
                    exact_tokens=generated['complete'] and generated['tokens'] == target)
                record['generation'].append(generated)
            record['passed'] = (all(r['request_mean_ce'] <= plan['max_request_mean_ce'] and r['bos_correct'] for r in record['final'])
                and all(r['exact_tokens'] for r in record['generation']))
            del model
            save()
        report.update(complete=True, status='completed',
            passed=report['arms']['constant_mode_candidate']['passed'],
            parent_unchanged=sha(ROOT/plan['parent']) == plan['source_sha256'][plan['parent']])
        if not report['parent_unchanged']:
            raise ValueError('Parent changed during probe')
        save()
    except Exception as exc:
        report.update(status='failed', passed=False, error=f'{type(exc).__name__}: {exc}')
        save()
        raise


if __name__ == '__main__':
    run()
