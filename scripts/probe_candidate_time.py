"""One-request comparison of shared and per-candidate millisecond queries."""
import json
import os
from pathlib import Path
import time

import torch

from beatfcn.model.candidate_time import head_ms_accuracy, head_ms_logits
from beatfcn.model.full_event_codec import FIELDS
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.train.local_request_panel import LocalRequestPanel

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'bench/rhythm-v2/candidate-time-query-v1.json'


def write(path, payload):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def shift_content(inputs, fraction):
    shifted = {k: v.clone() if torch.is_tensor(v) else v for k, v in inputs.items()}
    for key, dim in (('local', 2), ('mel', 1), ('song', 1)):
        amount = int(shifted[key].shape[dim] * fraction)
        shifted[key] = torch.roll(shifted[key], shifts=amount, dims=dim)
    return shifted


def choose(panel):
    field = FIELDS.index('head_ms')
    for index, row in enumerate(panel.records):
        if row['split'] != 'train' or row['mode'] != 'phira':
            continue
        sequence = panel.sequence(index)
        count = int((sequence['fields'] == field).sum())
        if 24 <= count <= 80:
            return index, count
    raise ValueError('No train Phira request has between 24 and 80 head_ms decisions')


def measure(model, inputs, sequence, per_candidate):
    was = model.training
    model.eval()
    with torch.no_grad():
        logits = head_ms_logits(model, inputs, sequence, per_candidate_time=per_candidate)
        loss = event_loss(logits, sequence)
        report = head_ms_accuracy(logits, sequence, model.vocab)
    if was:
        model.train()
    report = {k: v for k, v in report.items() if k not in ('chosen', 'target')}
    report['loss'] = float(loss)
    if not torch.isfinite(loss):
        raise ValueError('Non-finite candidate-time loss')
    return report


def arm(model, init, inputs, sequence, shifted, per_candidate, plan, deadline, report):
    model.load_state_dict(init)
    model.train()
    torch.manual_seed(plan['seed'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'],
                                  weight_decay=plan['weight_decay'])
    name = 'candidate' if per_candidate else 'shared'
    curve = [dict(update=0, **measure(model, inputs, sequence, per_candidate))]
    report['arms'][name] = curve
    write(ROOT / plan['output'], report)
    print(json.dumps(dict(arm=name, update=0, accuracy=curve[0]['accuracy'], loss=curve[0]['loss'])),
          flush=True)
    for step in range(1, plan['steps'] + 1):
        deadline()
        optimizer.zero_grad(set_to_none=True)
        loss = event_loss(head_ms_logits(model, inputs, sequence, per_candidate_time=per_candidate), sequence)
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Non-finite gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(), plan['gradient_clip_norm'])
        optimizer.step()
        if step % 5 == 0 or step == plan['steps']:
            point = dict(update=step, **measure(model, inputs, sequence, per_candidate))
            curve.append(point)
            write(ROOT / plan['output'], report)
            print(json.dumps(dict(arm=name, update=step, accuracy=point['accuracy'], loss=point['loss'])),
                  flush=True)
    final = measure(model, shifted, sequence, per_candidate)
    report['shifted'][name] = {k: v for k, v in final.items()}
    write(ROOT / plan['output'], report)
    return curve[-1], final


def main():
    plan = json.loads(PLAN.read_text())
    output = ROOT / plan['output']
    if output.exists():
        raise FileExistsError(output)
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise ValueError('Native MPS required')
    torch.set_num_threads(2)
    torch.mps.set_per_process_memory_fraction(.6)
    started = time.monotonic()
    report = dict(id=plan['id'], complete=False, passed=False, inconclusive=False,
        optimizer_updates=0, quality_updates=0, new_cloud_compute_calls=0,
        dimension=plan['dimension'], hypothesis=plan['hypothesis'], arms={}, shifted={},
        parent=plan['parent'])

    def deadline():
        if time.monotonic() - started > plan['max_wall_s']:
            raise TimeoutError('Candidate-time probe wall limit')

    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    index, count = choose(panel)
    row = panel.records[index]
    inputs, sequence = panel.batch([index], device='mps')
    report['request'] = dict(record_index=index, chart_id=row['chart_id'], mode=row['mode'],
        split=row['split'], head_ms=count, tokens=int(sequence['valid'].sum()),
        audio_sha256=row['audio_sha256'])
    saved = torch.load(ROOT / plan['parent'], map_location='cpu', weights_only=False)
    if saved['updates'] != 2:
        raise ValueError('Parent is not the verified step-0002 state')
    model = FullEventModel(FullEventConfig(**saved['config'])).to('mps')
    init = {k: v.detach().cpu().clone() for k, v in saved['model'].items()}
    del saved
    model.load_state_dict(init)
    report['trainable_parameters'] = sum(p.numel() for p in model.parameters())
    shifted = shift_content(inputs, plan['shift_fraction'])
    try:
        shared, shared_shift = arm(model, init, inputs, sequence, shifted, False, plan, deadline, report)
        candidate, candidate_shift = arm(model, init, inputs, sequence, shifted, True, plan, deadline, report)
        gain = candidate['accuracy'] - shared['accuracy']
        drop = candidate['accuracy'] - candidate_shift['accuracy']
        saturated = shared['accuracy'] >= plan['control_saturation']
        passed = (gain >= plan['minimum_accuracy_gain'] and drop >= plan['minimum_shift_drop']
                  and not saturated)
        report.update(complete=True, passed=passed, inconclusive=saturated and not passed,
            gain=gain, shift_drop=drop, shared_accuracy=shared['accuracy'],
            candidate_accuracy=candidate['accuracy'],
            shifted_shared_accuracy=shared_shift['accuracy'],
            shifted_candidate_accuracy=candidate_shift['accuracy'],
            optimizer_updates=plan['steps'] * 2, wall_s=time.monotonic() - started)
    except Exception as exc:
        report.update(complete=False, error=f'{type(exc).__name__}: {exc}',
            wall_s=time.monotonic() - started)
        write(output, report)
        raise
    write(output, report)
    print(json.dumps(dict(passed=report['passed'], inconclusive=report['inconclusive'],
        gain=report['gain'], shift_drop=report['shift_drop'])), flush=True)


if __name__ == '__main__':
    main()
