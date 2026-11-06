"""D37 exact rejected-baseline reproduction with read-only signal/gradient probes."""
import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import types

import numpy as np
import torch

from beatfcn.model.event_sequence import FIELDS
from scripts.matched_acoustic_data import make_example
from scripts.r09_local_data import verify_files, file_sha

ROOT = Path(__file__).resolve().parents[1]


def frozen_module():
    path = ROOT / 'inputs/code_baselines/r17-complete-event-state/rejected/scripts/r17_sequence_candidate.py'
    module = types.ModuleType('d37_frozen_r17')
    # Preserve the archived module's original relative dependency root without
    # restoring an executable entry into scripts/ or changing its source bytes.
    module.__file__ = str(ROOT / 'scripts/r17_sequence_candidate.py')
    sys.modules[module.__name__] = module
    exec(compile(path.read_bytes(), module.__file__, 'exec'), module.__dict__)
    return module


def count_reference(queries):
    mass = defaultdict(lambda: np.zeros(2, np.float64))
    raw = defaultdict(lambda: np.zeros(2, np.int64))
    for q in queries:
        use = q.fields == FIELDS.index('member')
        counts, target = q.syntax[use, 9].tolist(), q.targets[use].tolist()
        for count, y in zip(counts, target):
            mass[count][y] += 1 / len(counts) / len(queries)
            raw[count][y] += 1
    entropy = 0.
    for values in mass.values():
        positive = values[values > 0]
        entropy -= float(np.sum(positive * np.log(positive / values.sum())))
    return dict(equal_core_conditional_entropy=entropy,
        equal_core_majority_accuracy=sum(float(v.max()) for v in mass.values()),
        counts={str(c):v.tolist() for c, v in raw.items()},
        mass={str(c):v.tolist() for c, v in mass.items()}, diagnostic_not_used_for_generation=True)


@torch.no_grad()
def signal_trace(model, inputs, query):
    selected = (query.fields == FIELDS.index('member')) & (query.syntax[:, 9] < 3)
    changed_syntax = query.syntax.clone()
    changed_syntax[selected, 9] += 1
    altered = replace(query, syntax=changed_syntax)
    prepared = model.prepare(inputs)
    history = torch.cat((model.null_history, model.event_tokens(query.events, inputs)), 0)
    allow = torch.arange(len(history))[None] <= query.syntax[:, 10:11]
    x, alternative = model.query_tokens(query, inputs), model.query_tokens(altered, inputs)
    rows = []

    def record(stage):
        difference = (alternative[selected] - x[selected]).norm(dim=-1)
        original_norm = x[selected].norm(dim=-1)
        before = model.outputs['member'](model.norm(x))[selected].softmax(-1)[:, 0]
        after = model.outputs['member'](model.norm(alternative))[selected].softmax(-1)[:, 0]
        rows.append(dict(stage=stage, mean_signal_norm=float(difference.mean()),
            mean_state_norm=float(original_norm.mean()),
            mean_relative_signal=float((difference / original_norm.clamp_min(1e-12)).mean()),
            mean_end_probability_change=float((after - before).mean()),
            positive_end_probability_fraction=float((after > before).float().mean())))

    record('query')
    for i, (layer, static) in enumerate(zip(model.layers, prepared['static'])):
        kv = layer.attentions[0].prepare(history)
        x = layer(x, kv, static, allow)
        alternative = layer(alternative, kv, static, allow)
        record(f'layer_{i + 1}')
    output = model.readout(x, prepared)
    ordinary = model(inputs, query, prepared=prepared)
    if any(not torch.equal(output[k], ordinary[k]) for k in output):
        raise ValueError('trace changed the original forward')
    member = query.fields == FIELDS.index('member')
    truth = query.targets[member]
    logits = output['member'][member]
    predictions = logits.argmax(-1)
    count = query.syntax[member, 9]
    groups = {}
    for value in sorted(set(count.tolist())):
        use = count == value
        confusion = torch.bincount(truth[use] * 2 + predictions[use], minlength=4).reshape(2, 2)
        groups[str(value)] = dict(support=int(use.sum()), confusion_true_rows=confusion.tolist(),
            mean_end_probability=float(logits[use].softmax(-1)[:, 0].mean()))
    return dict(stages=rows, actual_predictions_by_member_count=groups,
        raw_member_decisions=dict(count=count.tolist(), target=truth.tolist(), pred=predictions.tolist(),
            end_probability=logits.softmax(-1)[:, 0].tolist()),
        count_column_weight_norm=float(model.query_counts.weight[:, -1].norm()),
        raw_plus_one_count_feature_norm=float((model.query_counts.weight[:, -1] / 8).norm()),
        counterfactual_keeps_history_fixed_and_may_be_unreachable=True)


def gradient_trace(model, inputs, query, frozen):
    outputs = model(inputs, query)
    total, terms, _ = frozen.supervised_loss(outputs, query)
    names = ['query_counts.weight', 'query_time.weight', 'phase.weight', 'occupied.weight',
             'layers.0.attentions.0.q.weight', 'layers.0.ffn.0.weight']
    lookup = dict(model.named_parameters())
    params = [lookup[n] for n in names]
    a = torch.autograd.grad(terms['member'], params, retain_graph=True)
    b = torch.autograd.grad(total - terms['member'], params)
    rows = {}
    for name, first, second in zip(names, a, b):
        if not torch.isfinite(first).all() or not torch.isfinite(second).all():
            raise ValueError('nonfinite diagnostic gradient')
        an, bn = float(first.norm()), float(second.norm())
        row = dict(member_norm=an, other_fields_norm=bn,
            cosine=float((first * second).sum()) / (an * bn) if an and bn else None)
        if name == 'query_counts.weight':
            am, bm = first[:, -1], second[:, -1]
            amn, bmn = float(am.norm()), float(bm.norm())
            row['member_count_column'] = dict(member_norm=amn, other_fields_norm=bmn,
                cosine=float((am * bm).sum()) / (amn * bmn) if amn and bmn else None)
        rows[name] = row
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('read-only gradient trace accumulated training gradients')
    return rows


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
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    frozen = frozen_module()
    data = json.loads((ROOT / plan['panel_plan']).read_text())
    manifest = json.loads((ROOT / plan['panel_manifest']).read_text())
    baseline = json.loads((ROOT / plan['baseline']).read_text())
    examples = [make_example(item, data) for item in manifest['items']]
    queries = [frozen.teacher_queries(i, labels)[0] for i, labels, _ in examples]
    model = frozen.SequenceModel(history_enabled=True, config=plan['model'], seed=plan['seed']).eval()
    if frozen.state_sha(model) != baseline['initial_state_sha256']:
        raise ValueError('wrong baseline initialization')
    report = dict(id='D37', complete=False, probe_passed=False, optimizer_updates=0,
        diagnostic_optimizer_updates=0, model_promotion=False, new_paid_compute_usd=0,
        checkpoint_saved=False, plan_sha256=file_sha(plan_path), initial_state_sha256=frozen.state_sha(model),
        count_reference=count_reference(queries), history=[], snapshots=[],
        quality_scope='instrumented reproduction of rejected R17 B, not a candidate or new quality evidence')
    started = time.perf_counter()

    def save():
        report.update(wall_s=time.perf_counter() - started,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

    def inspect(step):
        state, rng_state = frozen.state_sha(model), torch.random.get_rng_state().clone()
        rows = []
        for (inputs, _, identity), query in zip(examples, queries):
            rows.append(dict(identity=identity, signal=signal_trace(model, inputs, query),
                gradients=gradient_trace(model, inputs, query, frozen)))
        if state != frozen.state_sha(model) or not torch.equal(rng_state, torch.random.get_rng_state()):
            raise ValueError('inspection changed model or RNG')
        report['snapshots'].append(dict(step=step, model_state_sha256=state, state_and_rng_unchanged=True, cores=rows))
        save()

    save()
    try:
        inspect(0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=0.)
        rng = np.random.default_rng(plan['seed'])
        for step in range(100):
            if time.perf_counter() - started > plan['fit_deadline_s']:
                raise TimeoutError('diagnostic deadline')
            optimizer.zero_grad(set_to_none=True)
            losses, kinds = [], []
            for (inputs, labels, _), query in zip(examples, queries):
                draft, kind = frozen.corruption(labels, inputs, step, rng)
                outputs = model(inputs, query, draft)
                loss, _, _ = frozen.supervised_loss(outputs, query)
                (loss / len(examples)).backward()
                losses.append(float(loss.detach()))
                kinds.append(kind)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            reference = baseline['history'][step]
            error = max(abs(a - b) for a, b in zip(losses, reference['core_losses']))
            if error > 1e-6 or kinds != reference['views']:
                raise ValueError(f'reproduction mismatch at step {step + 1}: {error}')
            report['optimizer_updates'] = step + 1
            report['diagnostic_optimizer_updates'] = step + 1
            report['history'].append(dict(step=step + 1, loss=sum(losses) / len(losses),
                grad_norm=float(grad), max_baseline_loss_error=error))
            optimizer.zero_grad(set_to_none=True)
            if step + 1 in (10, 50, 100):
                inspect(step + 1)
            save()
        final = []
        with torch.no_grad():
            for (inputs, _, identity), query, expected in zip(examples, queries, baseline['final']['cores']):
                loss, terms, accuracy = frozen.supervised_loss(model(inputs, query), query)
                actual = {k:float(v) for k, v in terms.items()}
                if abs(float(loss) - expected['loss']) > 1e-6 or accuracy != expected['accuracy'] or \
                        any(abs(v - expected['terms'][k]) > 1e-6 for k, v in actual.items()):
                    raise ValueError('final unweighted loss or accuracy differs from baseline')
                final.append(dict(identity=identity, loss=float(loss), terms=actual, accuracy=accuracy))
        report.update(final=dict(mean_loss=sum(c['loss'] for c in final) / len(final), cores=final),
            training_forwards=800, training_field_queries=sum(len(q.fields) for q in queries) * 100,
            final_state_sha256=frozen.state_sha(model), complete=True, probe_passed=True,
            exact_baseline_reproduced=True, candidate_quality_passed=False)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        save()
        raise
    save()
    print(json.dumps({k:report[k] for k in ('complete', 'optimizer_updates', 'wall_s',
        'exact_baseline_reproduced', 'count_reference')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
