"""Real paired-chart diagnostic: identical audio/context, different true labels.

Both routes use the original CE. Every probe weight is discarded. Memorized
paired generation is an implementation gate, never held-out controllability.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.full_event_codec import EventRequest
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.train.event_conditions import CONTROL_FIELDS
from beatfcn.train.local_audio_audit import tensor_health
from beatfcn.train.local_request_panel import LocalRequestPanel
from beatfcn.train.local_shapes import pad_input_shapes
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def run():
    path = ROOT / 'bench/rhythm-v2/d70-condition-route-probe-v1.json'
    plan = json.loads(path.read_text()); output = ROOT / plan['output']
    if output.exists():
        raise FileExistsError(output)
    for source, expected in plan['inputs_sha256'].items():
        if sha(ROOT / source) != expected:
            raise ValueError('Frozen probe input changed: ' + source)
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise ValueError('Native MPS required')
    torch.set_num_threads(2); torch.mps.set_per_process_memory_fraction(.6)
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    cases, indices, controls = [], [], []
    for pair in plan['pairs']:
        prepared = []
        for index in pair['indices']:
            row = panel.records[index]
            if row['split'] != 'train' or row['mode'] != 'phira':
                raise ValueError('Probe must use actual Phira training requests')
            control = np.asarray(row['controls'], np.float32).copy()
            field = CONTROL_FIELDS.index(pair['field'])
            control[[j for j in range(len(control)) if j != field], :3] = 0
            inp, seq = panel.batch([index], controls=[control])
            prepared.append((inp, seq)); indices.append(index); controls.append(control)
            cases.append(dict(index=index, chart_id=row['chart_id'], component=row['component'],
                              field=pair['field'], band=row['bands'][pair['field']],
                              request=row['request'], actual_events=row['event_count'], types=row['types']))
        if any(not torch.equal(value, prepared[1][0][key]) for key, value in prepared[0][0].items() if key != 'controls'):
            raise ValueError('Pair has different audio or visible context')
        ta, tb = [values[1]['targets'][0].tolist() for values in prepared]
        divergence = next(i for i, (a, b) in enumerate(zip(ta, tb)) if a != b)
        for key in ('tokens', 'fields', 'times', 'valid', 'allowed'):
            if not torch.equal(prepared[0][1][key][:, :divergence+1], prepared[1][1][key][:, :divergence+1]):
                raise ValueError('First branch has a non-condition discriminator')
        for case in cases[-2:]:
            case['divergence'] = divergence
    cpu_inp, cpu_seq = panel.batch(indices, controls=controls)
    cpu_inp = pad_input_shapes(cpu_inp, plan['input_shapes'])
    tensor_stats = {section: {k: tensor_health(v.numpy()) for k, v in arrays.items()}
                    for section, arrays in [('inputs', cpu_inp), ('sequence', cpu_seq)]}
    batch = [(pad_input_shapes(panel.batch([i], controls=[c])[0], plan['input_shapes']),
              panel.batch([i], controls=[c])[1]) for i, c in zip(indices, controls)]
    device = [({k: v.to('mps') for k, v in inp.items()}, {k: v.to('mps') for k, v in seq.items()})
              for inp, seq in batch]
    saved = torch.load(ROOT / plan['checkpoint'], map_location='cpu', weights_only=False)
    report = dict(id=plan['id'], complete=False, passed=False, plan_sha256=sha(path), cases=cases,
                  tensor_stats=tensor_stats, routes=[], optimizer_updates=0, quality_updates=0,
                  new_cloud_compute_calls=0, weights_discarded=True,
                  scope='Two real training song pairs, CE only, 100 updates per route; not heldout generation quality')
    started = time.monotonic()
    try:
        for gain in plan['gains']:
            config = FullEventConfig(**{**saved['config'], 'condition_residual_gain': gain})
            model = FullEventModel(config).to('mps').train(); model.load_state_dict(saved['model'])
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
            optimizer.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['cpu_rng']); torch.mps.set_rng_state(saved['mps_rng'])
            route = dict(gain=gain, config=asdict(config), parameters=sum(p.numel() for p in model.parameters()),
                         curve=[], updates=0, learning_rate=optimizer.param_groups[0]['lr'])
            report['routes'].append(route)

            def measure(step):
                model.eval(); rows = []
                with torch.no_grad():
                    for case, (inp, seq) in zip(cases, device):
                        logits = model(inp, seq).masked_fill(~seq['allowed'], -1e9)
                        branch = case['divergence']
                        rows.append(dict(ce=float(event_loss(logits, seq)),
                            accuracy=float((logits.argmax(-1)[seq['valid']] == seq['targets'][seq['valid']]).float().mean()),
                            branch_correct=bool(logits.argmax(-1)[0, branch] == seq['targets'][0, branch]),
                            branch_target=int(seq['targets'][0, branch]), branch_prediction=int(logits.argmax(-1)[0, branch])))
                route['curve'].append(dict(step=step, rows=rows)); model.train()
                report.update(wall_s=time.monotonic()-started)
                write_json(output, report)

            measure(0)
            for step in range(1, plan['steps']+1):
                if time.monotonic()-started > plan['max_wall_s']:
                    raise TimeoutError('Paired condition health time cap')
                optimizer.zero_grad(set_to_none=True)
                for inp, seq in device:
                    loss = event_loss(model(inp, seq), seq) / len(device)
                    loss.backward(); del loss
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                control_grad = float(model.control_projection.weight.grad.norm())
                optimizer.step()
                route.update(updates=step, gradient_norm=float(grad), control_gradient_norm=control_grad)
                report['optimizer_updates'] += 1
                if step in (25, 50, 100):
                    measure(step)
            model = model.to('cpu').eval(); del optimizer
            torch.mps.empty_cache()
            with torch.no_grad():
                requests = [EventRequest(**{**c['request'], 'copy_times': tuple(c['request']['copy_times']),
                                            'known_carry': tuple(c['request']['known_carry'])}) for c in cases]
                predictions = generate_batch(model, cpu_inp, requests, plan['token_cap'],
                                             deadline=started+plan['max_wall_s'])
            expected = [seq['targets'][0][seq['valid'][0]].tolist() for _, seq in batch]
            route['generation'] = predictions
            route['exact_memorized_outputs'] = [p['complete'] and p['tokens'] == t for p, t in zip(predictions, expected)]
            route['passed'] = (all(r['ce'] <= plan['maximum_ce'] and r['branch_correct'] for r in route['curve'][-1]['rows'])
                               and all(route['exact_memorized_outputs']))
            route['driver_bytes_end'] = torch.mps.driver_allocated_memory()
            del model
            write_json(output, report)
        report.update(complete=True, passed=report['routes'][-1]['passed'], wall_s=time.monotonic()-started)
        write_json(output, report)
        print(json.dumps(dict(passed=report['passed'], wall_s=report['wall_s'], routes=[
            {k: r[k] for k in ['gain', 'passed', 'exact_memorized_outputs', 'control_gradient_norm']} for r in report['routes']])))
    except Exception as exc:
        report.update(complete=True, error=f'{type(exc).__name__}: {exc}', wall_s=time.monotonic()-started)
        write_json(output, report)
        raise


if __name__ == '__main__':
    run()
