"""Exact archived R19 replay and isolated one-step optimizer sensitivity."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import types

import numpy as np
import torch

from scripts.matched_acoustic_data import make_example
from scripts.r09_local_data import file_sha, verify_files

ROOT = Path(__file__).resolve().parents[1]


def tree_sha(value):
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def hypothetical_step(model, optimizer, multiplier):
    """Deep-copy both parameters and optimizer moments before a diagnostic step."""
    candidate = deepcopy(model)
    trial = torch.optim.AdamW(candidate.parameters(), lr=.0003, weight_decay=0.)
    trial.load_state_dict(deepcopy(optimizer.state_dict()))
    for group in trial.param_groups:
        group['lr'] *= multiplier
    for src, dst in zip(model.parameters(), candidate.parameters()):
        dst.grad = None if src.grad is None else src.grad.detach().clone()
    trial.step()
    candidate.zero_grad(set_to_none=True)
    return candidate


def update_summary(before, after):
    accum = {}
    for (name, src), (other, dst) in zip(before.named_parameters(), after.named_parameters()):
        if name != other or not torch.isfinite(dst).all():
            raise ValueError('parameter identity or finiteness failure')
        prefix = name.split('.')[0]
        if prefix == 'outputs':
            prefix += '.' + name.split('.')[1]
        row = accum.setdefault(prefix, dict(elements=0, parameter_square=0., update_square=0.))
        row['elements'] += src.numel()
        row['parameter_square'] += float(src.detach().double().square().sum())
        row['update_square'] += float((dst.detach().double() - src.detach().double()).square().sum())
    for row in accum.values():
        row['parameter_rms'] = (row['parameter_square'] / row['elements']) ** .5
        row['update_rms'] = (row['update_square'] / row['elements']) ** .5
        row['relative_update_l2'] = (row['update_square'] / max(row['parameter_square'], 1e-24)) ** .5
    return accum


def loss_report(model, examples, queries, frozen, *, backward=False):
    rows = []
    for (inputs, _, identity), query in zip(examples, queries):
        output = model(inputs, query)
        loss, terms, accuracy = frozen.supervised_loss(output, query)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite diagnostic loss')
        if backward:
            (loss / len(examples)).backward()
        rows.append(dict(identity=identity, loss=float(loss.detach()),
            terms={k:float(v.detach()) for k, v in terms.items()}, accuracy=accuracy))
    return dict(mean_loss=sum(r['loss'] for r in rows) / len(rows), cores=rows)


def load_archived_model(path):
    module = types.ModuleType('d39_frozen_r19')
    module.__file__ = str(ROOT / 'scripts/r19_count_candidate.py')
    sys.modules[module.__name__] = module
    exec(compile(path.read_bytes(), module.__file__, 'exec'), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_text())
    verify_files(plan)
    destination = ROOT / plan['result']
    if destination.exists():
        raise FileExistsError('preserve previous diagnostic')
    torch.set_num_threads(plan['threads'])
    torch.set_num_interop_threads(1)
    frozen = load_archived_model(ROOT / plan['frozen_model'])
    data_plan = json.loads((ROOT / plan['panel_plan']).read_text())
    manifest = json.loads((ROOT / plan['panel_manifest']).read_text())
    baseline = json.loads((ROOT / plan['baseline']).read_text())
    examples = [make_example(item, data_plan) for item in manifest['items']]
    queries = [frozen.teacher_queries(inputs, labels)[0] for inputs, labels, _ in examples]
    model = frozen.SequenceModel(history_enabled=True, config=plan['model'], seed=plan['seed']).eval()
    if frozen.state_sha(model) != baseline['initial_state_sha256']:
        raise ValueError('initial state mismatch')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=0.)
    rng = np.random.default_rng(plan['seed'])
    report = dict(id='D39', complete=False, optimizer_updates=0, diagnostic_optimizer_updates=0,
        hypothetical_optimizer_steps=0, diagnostic_forwards=0, snapshots=[], history=[],
        model_promotion=False, checkpoint_saved=False, new_paid_compute_usd=0,
        plan_sha256=file_sha(plan_path), initial_state_sha256=frozen.state_sha(model),
        evidence_scope='exact rejected-R19 replay; local train-loss line probes, not quality trials')
    started = time.perf_counter()

    def save():
        report.update(wall_s=time.perf_counter() - started,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

    def inspect(step):
        model.zero_grad(set_to_none=True)
        model_hash = frozen.state_sha(model)
        optimizer_hash = tree_sha(optimizer.state_dict())
        torch_rng, numpy_rng = torch.random.get_rng_state().clone(), deepcopy(rng.bit_generator.state)
        current = loss_report(model, examples, queries, frozen, backward=True)
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        snapshot = dict(step=step, current=current, all_mask_grad_norm=float(grad), proposals=[])
        report['diagnostic_forwards'] += len(examples)
        for factor in plan['lr_multipliers']:
            if time.perf_counter() - started > plan['fit_deadline_s']:
                raise TimeoutError('diagnostic deadline')
            candidate = hypothetical_step(model, optimizer, factor)
            with torch.no_grad():
                result = loss_report(candidate, examples, queries, frozen)
            snapshot['proposals'].append(dict(lr_multiplier=factor, effective_lr=.0003 * factor,
                loss_delta=result['mean_loss'] - current['mean_loss'], result=result,
                parameter_updates=update_summary(model, candidate)))
            report['hypothetical_optimizer_steps'] += 1
            report['diagnostic_forwards'] += len(examples)
            del candidate
            if frozen.state_sha(model) != model_hash or tree_sha(optimizer.state_dict()) != optimizer_hash:
                raise ValueError('hypothetical step mutated the original state')
        model.zero_grad(set_to_none=True)
        if not torch.equal(torch_rng, torch.random.get_rng_state()) or numpy_rng != rng.bit_generator.state:
            raise ValueError('inspection changed training RNG')
        snapshot.update(model_state_sha256=model_hash, optimizer_state_sha256=optimizer_hash,
            original_model_optimizer_and_rng_unchanged=True)
        report['snapshots'].append(snapshot)
        save()

    save()
    try:
        for step in range(100):
            if time.perf_counter() - started > plan['fit_deadline_s']:
                raise TimeoutError('diagnostic deadline')
            optimizer.zero_grad(set_to_none=True)
            losses, views = [], []
            for (inputs, labels, _), query in zip(examples, queries):
                draft, view = frozen.corruption(labels, inputs, step, rng)
                loss, _, _ = frozen.supervised_loss(model(inputs, query, draft), query)
                (loss / len(examples)).backward()
                losses.append(float(loss.detach()))
                views.append(view)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            expected = baseline['history'][step]
            error = max(abs(a - b) for a, b in zip(losses, expected['core_losses']))
            if error > plan['baseline_reproduction_atol'] or views != expected['views']:
                raise ValueError(f'baseline mismatch at {step + 1}: {error}')
            report.update(optimizer_updates=step + 1, diagnostic_optimizer_updates=step + 1)
            report['history'].append(dict(step=step + 1, loss=sum(losses) / len(losses),
                grad_norm=float(grad), max_baseline_loss_error=error))
            if step + 1 in plan['inspection_steps']:
                inspect(step + 1)
            save()
        final = report['snapshots'][-1]['current']
        for actual, expected in zip(final['cores'], baseline['final']['cores']):
            if abs(actual['loss'] - expected['loss']) > plan['baseline_reproduction_atol'] or \
                    actual['accuracy'] != expected['accuracy'] or any(
                        abs(v - expected['terms'][k]) > plan['baseline_reproduction_atol']
                        for k, v in actual['terms'].items()):
                raise ValueError('final baseline mismatch')
        report.update(complete=True, probe_passed=True, exact_baseline_reproduced=True,
            training_forwards=800, training_field_queries=sum(len(q.fields) for q in queries) * 100,
            final_state_sha256=frozen.state_sha(model), final=final, candidate_quality_passed=False)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        save()
        raise
    save()
    print(json.dumps({k:report[k] for k in ('complete', 'optimizer_updates', 'hypothetical_optimizer_steps',
        'diagnostic_forwards', 'wall_s', 'exact_baseline_reproduced')}))


if __name__ == '__main__':
    main()
