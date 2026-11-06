"""Real new-song fixed-batch gate for D94, using the unchanged absolute codec."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.full_event_codec import encode_events
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.train.local_audio_audit import tensor_health
from beatfcn.train.local_request_panel import LocalRequestPanel, event_request
from beatfcn.train.local_shapes import pad_input_shapes
from scripts.prepare_local_data_scale_probe import padding_shapes, sha
from scripts.probe_local_sequence_likelihood import device_batch, unified_metrics
from scripts.run_local_event_experiment import memory
from scripts.run_local_song_baseline import write_json

ROOT = Path(__file__).resolve().parents[1]


def padding_proof(model, raw, padded, sequence, device, deadline):
    versions = []
    for inputs in (raw, padded):
        deadline()
        model.zero_grad(set_to_none=True)
        inp, seq = device_batch((inputs, sequence), device)
        logits = model(inp, seq)
        loss = event_loss(logits, seq)
        loss.backward()
        versions.append(dict(loss=float(loss.detach()),
            logits=logits.detach().cpu()[sequence['valid']],
            gradients={k: None if p.grad is None else p.grad.detach().cpu().clone()
                       for k, p in model.named_parameters()}))
        del inp, seq, logits, loss
    a, b = versions
    if any((a['gradients'][k] is None) != (b['gradients'][k] is None) for k in a['gradients']):
        raise ValueError('Padding changed gradient connectivity')
    return dict(loss_max_abs=abs(a['loss'] - b['loss']),
        valid_logits_max_abs=float((a['logits'] - b['logits']).abs().max()),
        gradient_max_abs=max(float((v - b['gradients'][k]).abs().max())
            for k, v in a['gradients'].items() if v is not None))


def run(plan_path):
    plan = json.loads(plan_path.read_text())
    output = ROOT / plan['output']
    if output.exists():
        raise FileExistsError('Probe already has evidence; inspect it before any new attempt')
    for path, expected in plan['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Frozen source or asset changed: ' + path)
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise ValueError('Native MPS required')
    if plan['steps'] != 100 or plan['microbatch_requests'] != 1 or len(plan['cases']) != 8:
        raise ValueError('The registered fixed batch is eight requests, microbatch one, 100 updates')
    torch.set_num_threads(2)
    torch.mps.set_per_process_memory_fraction(.6)
    started = time.monotonic()
    report = dict(id=plan['id'], plan_sha256=sha(plan_path), complete=False, passed=False,
        optimizer_updates=0, new_cloud_compute_calls=0, quality_passed=False,
        status='preparing', tensors=[], padding_proofs=[], initial=[], curve=[], final=[], generation=[],
        scope=plan['scope'])

    def save():
        report.update(wall_s=time.monotonic() - started, memory=memory('mps'))
        write_json(output, report)
        print(json.dumps(dict(status=report['status'], updates=report['optimizer_updates'],
                             wall_s=report['wall_s'])), flush=True)

    def deadline():
        if time.monotonic() - started > plan['max_wall_s']:
            raise TimeoutError('Frozen new-data probe wall limit')

    try:
        panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
        old = json.loads((ROOT / 'reports/d67_local_request_tape_v1.json').read_text())
        old_components = {c['component'] for c in old['charts']}
        cpu, raw_inputs, requests, targets = [], [], [], []
        for case in plan['cases']:
            deadline()
            index = case['record_index']
            row = panel.records[index]
            if (row['split'] != 'train' or row['chart_id'] != case['chart_id']
                    or row['component'] in old_components or row['request'] != case['request']):
                raise ValueError('Frozen new training request identity differs')
            controls = np.asarray(row['controls'], np.float32).copy()
            controls[[i for i in range(len(controls)) if i not in case['specified_fields']], :3] = 0
            inp, seq = panel.batch([index], controls=[controls])
            request, target = event_request(row['request']), sorted(panel.targets(index))
            actual = seq['targets'][0][seq['valid'][0]].tolist()
            if actual != list(encode_events(target, request)) or len(actual) != case['tokens']:
                raise ValueError('Actual targets and full event sequence do not align')
            if not bool(seq['allowed'].gather(2, seq['targets'][..., None]).squeeze(-1)[seq['valid']].all()):
                raise ValueError('Actual teacher target is illegal')
            if case['input_shapes'] != padding_shapes(inp['song'].shape[1]):
                raise ValueError('Full song context differs from the frozen padding bucket')
            raw_inputs.append(inp)
            inp = pad_input_shapes(inp, case['input_shapes'])
            cpu.append((inp, seq)); requests.append(request); targets.append(target)
            report['tensors'].append(dict(index=index, chart_id=row['chart_id'],
                request=row['request'], human_events=[asdict(e) for e in target],
                specified_fields=case['specified_fields'], target_count=len(target),
                target_alignment_exact=True, full_song_summary_kept=True,
                tensors={section: {k: tensor_health(v.numpy()) for k, v in values.items()}
                         for section, values in [('inputs', inp), ('sequence', seq)]}))
        del panel
        saved = torch.load(ROOT / plan['parent'], map_location='cpu', weights_only=False)
        if saved['updates'] != 2 or saved['config'] != plan['config']:
            raise ValueError('Common step2 state differs')
        model = FullEventModel(FullEventConfig(**plan['config'])).to('mps')
        model.load_state_dict(saved['model'])
        model.train()
        # Only the newly needed long-song bucket requires a new parity check.
        # Old 896-position and full/cache equivalence evidence is retained.
        for case, raw, pair in zip(plan['cases'], raw_inputs, cpu, strict=True):
            if case['input_shapes']['song'] == 1024:
                proof = padding_proof(model, raw, pair[0], pair[1], 'mps', deadline)
                proof['passed'] = all(np.isfinite(v) and v <= plan['padding_parity_atol'] for v in proof.values())
                proof.update(record_index=case['record_index'], backend='mps', optimizer_updates=0)
                report['padding_proofs'].append(proof)
                save()
                if not proof['passed']:
                    raise ValueError('Full long-song masked padding parity failed')
        if not report['padding_proofs']:
            raise ValueError('Missing the newly required long-song padding proof')
        del raw_inputs
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=0.)
        optimizer.load_state_dict(deepcopy(saved['optimizer']))
        if any(float(state['step']) != 2 for state in optimizer.state.values()):
            raise ValueError('Optimizer parent did not restore step2')
        if any(group['lr'] != plan['learning_rate'] or group['weight_decay'] != 0
               for group in optimizer.param_groups):
            raise ValueError('Parent optimizer settings differ')
        with torch.no_grad():
            for pair in cpu:
                inp, seq = device_batch(pair, 'mps')
                report['initial'].append(unified_metrics(model(inp, seq), seq))
                del inp, seq
        torch.set_rng_state(saved['cpu_rng']); torch.mps.set_rng_state(saved['mps_rng'])
        del saved
        report.update(status='zero_update_passed', trainable_parameters=sum(p.numel() for p in model.parameters()))
        save()
        for step in range(1, plan['steps'] + 1):
            deadline()
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for pair in cpu:
                inp, seq = device_batch(pair, 'mps')
                value = event_loss(model(inp, seq), seq) / len(cpu)
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
                        gradients[group] = gradients.get(group, 0.) + float(param.grad.square().sum())
                report['initial_gradient_norm_by_module'] = {k: v**.5 for k, v in gradients.items()}
                for key in ('local_projection', 'mel_projection', 'token_embedding', 'decoder', 'output'):
                    if gradients.get(key, 0.) <= 0:
                        raise ValueError('No gradient reaches required module ' + key)
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
        checkpoint = ROOT / plan['checkpoint']
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open('xb') as stream:
            torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                config=asdict(model.c), parent=plan['parent'], diagnostic_only=True, auto_resume=False,
                probe_updates=plan['steps'], plan_sha256=sha(plan_path)), stream)
        report['checkpoint'] = dict(path=plan['checkpoint'], sha256=sha(checkpoint))
        del optimizer
        model.to('cpu')
        for case, pair, request, target in zip(plan['cases'], cpu, requests, targets, strict=True):
            deadline()
            generated = generate_batch(model, pair[0], [request], plan['generation_token_cap'],
                deadline=started + plan['max_wall_s'])[0]
            target_tokens = pair[1]['targets'][0][pair[1]['valid'][0]].tolist()
            generated.update(record_index=case['record_index'],
                exact_tokens=generated['complete'] and generated['tokens'] == target_tokens,
                exact_events=generated['events'] == [asdict(e) for e in target])
            report['generation'].append(generated)
        report.update(complete=True, status='completed',
            passed=(all(r['request_mean_ce'] <= plan['max_request_mean_ce'] and r['bos_correct'] for r in report['final'])
                    and all(r['exact_tokens'] and r['exact_events'] for r in report['generation'])),
            parent_unchanged=sha(plan['parent']) == plan['source_sha256'][plan['parent']])
        if not report['parent_unchanged']:
            raise ValueError('Common parent changed')
        save()
    except Exception as exc:
        report.update(status='failed', passed=False, error=f'{type(exc).__name__}: {exc}')
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan)
