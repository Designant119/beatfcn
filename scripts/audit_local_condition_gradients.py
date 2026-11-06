"""Zero-update, actual paired-prefix audit; no optimizer or model modification."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.train.event_conditions import CONTROL_FIELDS
from beatfcn.train.local_audio_audit import tensor_health
from beatfcn.train.local_request_panel import LocalRequestPanel
from beatfcn.train.local_shapes import pad_input_shapes
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def gradient_relation(a, b):
    a, b = a.double(), b.double()
    na, nb = float(a.norm()), float(b.norm())
    dot = float(a @ b)
    return dict(norm_a=na, norm_b=nb, dot=dot,
                cosine=dot / (na * nb) if na and nb else None,
                cancellation_ratio=float((a + b).norm()) / (na + nb) if na + nb else None)


def state_sha(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().numpy().tobytes())
    return h.hexdigest()


def run():
    plan_path = ROOT / 'bench/rhythm-v2/d73-condition-gradients-v1.json'
    plan = json.loads(plan_path.read_text()); output = ROOT / plan['output']
    if output.exists():
        raise FileExistsError(output)
    for name, expected in plan['inputs_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen diagnostic source changed: ' + name)
    torch.set_num_threads(2)
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    batch, cases = [], []
    for pair in plan['pairs']:
        prepared = []
        for index in pair['indices']:
            row = panel.records[index]
            if row['split'] != 'train' or row['mode'] != 'phira':
                raise ValueError('Expected actual Phira training requests')
            control = np.asarray(row['controls'], np.float32).copy()
            field = CONTROL_FIELDS.index(pair['field'])
            control[[i for i in range(len(control)) if i != field], :3] = 0
            inp, seq = panel.batch([index], controls=[control])
            inp = pad_input_shapes(inp, plan['input_shapes'])
            for tensors in (inp, seq):
                if any(not bool(torch.isfinite(v).all()) for v in tensors.values()):
                    raise ValueError('Nonfinite diagnostic input')
            prepared.append((inp, seq))
            cases.append(dict(index=index, chart_id=row['chart_id'], component=row['component'],
                              field=pair['field'], band=row['bands'][pair['field']],
                              request=row['request'], actual_events=row['event_count'], types=row['types']))
        if any(not torch.equal(v, prepared[1][0][k]) for k, v in prepared[0][0].items() if k != 'controls'):
            raise ValueError('Pair has different noncondition input')
        targets = [s['targets'][0][s['valid'][0]].tolist() for _, s in prepared]
        branch = next(i for i, (a, b) in enumerate(zip(*targets)) if a != b)
        for k in ('tokens', 'fields', 'times', 'allowed', 'valid'):
            if not torch.equal(prepared[0][1][k][:, :branch+1], prepared[1][1][k][:, :branch+1]):
                raise ValueError('Pair first fork has another discriminator')
        for case in cases[-2:]:
            case.update(branch=branch, target_a=targets[0][branch], target_b=targets[1][branch])
        batch.extend(prepared)
    report = dict(id=plan['id'], complete=False, passed=False, plan_sha256=sha(plan_path),
                  optimizer_updates=0, new_cloud_compute_calls=0, cases=cases, checkpoints=[],
                  tensor_stats=[{part: {k: tensor_health(v.numpy()) for k, v in values.items()}
                                 for part, values in [('inputs', inp), ('sequence', seq)]} for inp, seq in batch])
    started = time.monotonic()
    try:
        for checkpoint in plan['checkpoints']:
            saved = torch.load(ROOT / checkpoint, map_location='cpu', weights_only=False)
            model = FullEventModel(FullEventConfig(**saved['config'])).eval()
            model.load_state_dict(saved['model']); del saved
            before = state_sha(model)
            selected = {n: p for n, p in model.named_parameters() if n.startswith(('control_projection.', 'control_ids.', 'output.'))
                        or (n.startswith('decoder.') and '.cross_attn.' in n)}
            names, params = list(selected), list(selected.values())
            arm = dict(checkpoint=checkpoint, rows=[], pairs=[])
            report['checkpoints'].append(arm)
            gradients = []
            for i, (case, (inp, seq)) in enumerate(zip(cases, batch)):
                if time.monotonic() - started > plan['max_wall_s']:
                    raise TimeoutError('Diagnostic deadline reached')
                inp = {**inp, 'controls': inp['controls'].clone().requires_grad_(True)}
                capture = []
                hook = model.control_projection.register_forward_hook(lambda _m, _a, result: capture.append(result))
                try:
                    logits = model(inp, seq)
                finally:
                    hook.remove()
                branch, a, b = case['branch'], case['target_a'], case['target_b']
                if not bool(seq['allowed'][0, branch, [a, b]].all()):
                    raise ValueError('Both pair targets must be legal at the same fork')
                margin = logits[0, branch, a] - logits[0, branch, b]
                input_j, projected_j = torch.autograd.grad(margin, [inp['controls'], capture[0]], retain_graph=True)
                change = batch[i - i % 2][0]['controls'] - batch[i - i % 2 + 1][0]['controls']
                projected_change = model.control_projection(change) - model.control_projection(torch.zeros_like(change))
                local_direction = float((input_j * change).sum())
                projected_direction = float((projected_j * projected_change).sum().detach())
                legal = logits.masked_fill(~seq['allowed'], -1e9)
                losses = F.cross_entropy(legal.flatten(0, 1), seq['targets'].flatten(), reduction='none').view_as(seq['targets'])
                count = int(seq['valid'].sum())
                full = (losses * seq['valid']).sum() / (4 * count)
                fork = losses[0, branch] / (4 * count)
                rest_mask = seq['valid'].clone(); rest_mask[0, branch] = False
                rest = (losses * rest_mask).sum() / (4 * count)
                gf = torch.autograd.grad(full, params, retain_graph=True)
                gb = torch.autograd.grad(fork, params, retain_graph=True)
                gr = torch.autograd.grad(rest, params)
                gf, gb, gr = [{n: g.detach().flatten() for n, g in zip(names, gs)} for gs in (gf, gb, gr)]
                reconstruction = max(float((gf[n] - gb[n] - gr[n]).abs().max()) for n in names)
                with torch.no_grad():
                    prefix = {k: v[:, :branch+1] for k, v in seq.items()}
                    prefix_logits = model(inp, prefix)
                    prefix_delta = float((prefix_logits - logits[:, :branch+1]).abs().max())
                    probabilities = legal[0, branch].softmax(-1)
                if reconstruction > plan['max_gradient_reconstruction_error'] or prefix_delta > plan['max_prefix_error']:
                    raise ValueError('Gradient decomposition or prefix equivalence failed')
                arm['rows'].append(dict(chart_id=case['chart_id'], tokens=count,
                    margin_a_minus_b=float(margin.detach()), target_a_probability=float(probabilities[a]),
                    target_b_probability=float(probabilities[b]), branch_target=int(seq['targets'][0, branch]),
                    branch_prediction=int(legal[0, branch].argmax()),
                    full_ce=float(full.detach()) * 4, fork_ce=float(losses[0, branch].detach()),
                    input_margin_jacobian=input_j.tolist(), input_jacobian_norm=float(input_j.norm()),
                    projected_jacobian_norm=float(projected_j.norm()),
                    direction_for_a_minus_b=local_direction, projected_direction_for_a_minus_b=projected_direction,
                    gradient_reconstruction_error=reconstruction, prefix_max_logit_difference=prefix_delta,
                    fork_rest_gradients={n: gradient_relation(gb[n], gr[n]) for n in names}))
                gradients.append((gf, gb, gr))
                del logits, legal, losses, full, fork, rest, capture, margin, input_j, projected_j, inp
                write_json(output, report)
            for i in (0, 2):
                ra, rb = arm['rows'][i:i+2]
                arm['pairs'].append(dict(field=cases[i]['field'], indices=[cases[i]['index'], cases[i+1]['index']],
                    margin_a=ra['margin_a_minus_b'], margin_b=rb['margin_a_minus_b'],
                    exact_margin_difference_a_minus_b=ra['margin_a_minus_b'] - rb['margin_a_minus_b'],
                    both_correct=ra['branch_prediction'] == ra['branch_target'] and rb['branch_prediction'] == rb['branch_target'],
                    gradients={part: {n: gradient_relation(gradients[i][j][n], gradients[i+1][j][n]) for n in names}
                               for j, part in enumerate(('full', 'fork', 'rest'))}))
            arm['weights_unchanged'] = before == state_sha(model)
            if not arm['weights_unchanged']:
                raise ValueError('Diagnostic altered model weights')
            del gradients, params, selected, model
        report.update(complete=True, passed=True, wall_s=time.monotonic()-started,
                      scope='Mechanism observations on two preexisting checkpoints; not D70-D72 discarded terminal weights or heldout quality.')
        write_json(output, report)
        print(json.dumps(dict(passed=True, wall_s=report['wall_s'],
                              pairs=[[{k:v for k,v in p.items() if k!='gradients'} for p in c['pairs']] for c in report['checkpoints']])))
    except Exception as exc:
        report.update(complete=True, passed=False, error=f'{type(exc).__name__}: {exc}', wall_s=time.monotonic()-started)
        write_json(output, report)
        raise


if __name__ == '__main__':
    run()
