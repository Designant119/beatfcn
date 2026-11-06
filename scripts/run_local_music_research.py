"""Finish the two previously proposed, independently testable research pairs."""
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_music_research import BudgetAllocationCanvas, ContextMatchingCanvas
from beatfcn.model.local_typed_canvas import canvas_loss
from beatfcn.train.canonical_probability import explicit_row
from beatfcn.train.relative_canvas import relative_batch, generate_relative
from scripts.confirm_canvas_finalists import gates as screen_gates
from scripts.replicate_canvas_finalist import evaluate
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_group_diffusion_health import rate
from scripts.run_local_portfolio_scale import row_factory, weight_digest, sha, write


ROOT = Path(__file__).resolve().parents[1]


def audio_descriptor(example):
    x = np.asarray(example['inputs']['mel'])[np.asarray(example['inputs']['mel_valid'])].mean(-1)
    y = x - x.mean()
    scale = float(y @ y) + 1e-8
    correlations = [float(y[:-lag] @ y[lag:] / scale) if len(y) > lag else 0.
                    for lag in (25, 33, 50, 67, 100)]
    return np.array([x.mean(), x.std(), np.maximum(np.diff(x), 0).mean(), *correlations])


def negative_map(panel, row, indices):
    values = np.stack([audio_descriptor(row(i, 'draft')) for i in indices])
    if not np.isfinite(values).all():
        raise ValueError('Audio-only descriptor must be finite')
    values = (values - values.mean(0)) / np.maximum(values.std(0), 1e-5)
    result = {}
    for j, index in enumerate(indices):
        distance = np.square(values - values[j]).sum(-1)
        candidates = [k for k in np.argsort(distance, kind='stable')
                      if panel.records[indices[int(k)]]['component'] != panel.records[index]['component']]
        result[str(index)] = [indices[int(k)] for k in candidates[:3]]
    return result


def make_model(plan, arm, device):
    torch.manual_seed(plan['seed'])
    config = FullEventConfig(**plan['model'])
    if plan['kind'] == 'depth':
        width, depth = plan['allocations'][arm]
        model = BudgetAllocationCanvas(config, density_reference=plan['density_reference'], width=width, depth=depth)
    else:
        model = ContextMatchingCanvas(config, density_reference=plan['density_reference'])
    return model.to(device)


def input_batch(example, device):
    inp, tgt = relative_batch([example], device)
    return {k: v for k, v in inp.items() if not k.startswith(('local', 'song'))}, tgt


def loss_for(model, plan, arm, example, inp, tgt, row, device):
    main, parts = canvas_loss(model(inp), tgt, inp)
    if plan['kind'] == 'ssl':
        negatives = [input_batch(row(i, 'draft'), device)[0]
                     for i in plan['negative_map'][str(example['index'])]]
        auxiliary, _ = model.matching_loss(inp, negatives)
        parts = dict(parts, matching=auxiliary)
        main = main + plan['auxiliary_weight'][arm] * auxiliary
    return main, parts


@torch.no_grad()
def diagnostics(model, plan, arm, batches, row, device):
    model.eval()
    result = []
    for e, inp, tgt in batches:
        loss, parts = loss_for(model, plan, arm, e, inp, tgt, row, device)
        result.append(dict(index=e['index'], loss=float(loss), parts={k: float(v) for k, v in parts.items()}))
    return result


@torch.no_grad()
def coarse_counterfactual(model, plan, row, panel, art, deadline, *, coarse_only=True):
    from beatfcn.model.event_canvas import RhythmEvent
    from beatfcn.model.sample import match_f1
    from scripts.run_local_typed_canvas import score
    from scripts.review_frontend_canvas_comparison import metrics
    records = []
    indices = plan['indices']['dev']
    for j, index in enumerate(indices):
        if time.monotonic() > deadline:
            raise TimeoutError('Coarse counterfactual wall')
        e = row(index, 'draft')
        donor = indices[(j + 2) % len(indices)]
        if panel.records[index]['component'] == panel.records[donor]['component']:
            raise ValueError('Coarse donor must differ by component')
        inp, _ = input_batch({**e, 'targets': []}, 'mps')
        other, _ = input_batch(row(donor, 'draft'), 'mps')
        if coarse_only:
            inp['coarse_mel'] = other['mel']
        else:
            inp['mel'] = other['mel']
        pred = generate_relative(model, inp, e['request'], deadline=deadline)
        events = [RhythmEvent(**v) for v in pred['events']]
        start = e['request'].start_us
        width = e['request'].end_us - start
        shifts = [match_f1([start + (v.onset_us - start + d) % width for v in events],
                          [v.onset_us for v in e['targets']], 10000)[3] for d in plan['shifts_us']]
        records.append(dict(index=index, component=panel.records[index]['component'], view='draft',
            request=asdict(e['request']), prediction=pred, target=[asdict(v) for v in e['targets']],
            scores=score(events, e['targets']), shifted_onset10=shifts, donor_index=donor))
        write(art / ('coarse-swapped.json' if coarse_only else 'all-audio-swapped.json'), records)
    return metrics(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=('depth', 'ssl'), required=True)
    kind = parser.parse_args().kind
    ident = f'local-music-{kind}-pair-v1'
    plan_path = ROOT / f'bench/rhythm-v2/{ident}.json'
    report_path = ROOT / f'reports/local_music_{kind}_pair_v1.json'
    art = ROOT / 'artifacts' / ident
    if plan_path.exists() or report_path.exists():
        raise FileExistsError('No automatic research retry')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise ValueError('No silent fallback')
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.5)
    parent = json.loads((ROOT / 'bench/rhythm-v2/local-portfolio-scale-v1.json').read_text())
    panel, features = setup(json.loads((ROOT / 'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()))
    base = row_factory(panel, features)
    def row(*args, **kwargs):
        return explicit_row(base(*args, **kwargs))
    arms = ['wide8', 'deep16'] if kind == 'depth' else ['supervised', 'matching']
    plan = dict(id=ident, kind=kind, arms=arms, seed=20261074 if kind == 'depth' else 20261075,
        model=parent['models']['small64'], density_reference=parent['density_reference'],
        indices=parent['indices'], stream=parent['sample_stream']['train64'][:1000],
        health_indices=parent['health_indices'], empty_health_index=parent['empty_health_index'],
        health_steps=100, quality_steps=1000, effective_batch=2,
        allocations=dict(wide8=[192, 8], deep16=[144, 16]), auxiliary_weight=dict(supervised=0., matching=.1),
        learning_rate=.0005, minimum_learning_rate=.00005, warmup=50, weight_decay=0.,
        per_arm_total_wall_s=7200, health_wall_s=1200, evaluation_wall_s=600,
        shifts_us=parent['shifts_us'], matched_time_s=600,
        scope='Actual research note proposals2/3, separate from the two exhausted council rounds. No text/pretrained model weights. Nativeevent/COPY contract unchanged; no emotion or human-motif annotation claim.',
        hypothesis=('At near-equal full-model parameter budget, deeper/narrower temporal allocation improves complete events and placement at equal updates and within a fixed600second training envelope.' if kind == 'depth' else
                    'Adding disjoint2second-local versus symmetric12second-surround audio matching, with3other-component energy/tempo-proxy hard negatives, improves supervised chart generation and useful coarse-context benefit.'),
        hypothesis_gate=dict(full_gain=.015 if kind == 'depth' else .01, onset_max_regression=.01,
                             component_wins=10, useful_context_gain=.01, acoustic_onset_gain=.05),
        local_screen=dict(complete=32, min_nonempty=24, count_ratio=[.5, 1.75], full_note_f1=.20,
                          onset10=.20, onset10_minus_shift=.05, min_positive_components=10),
        depth_adjustment='136/4=34 violates the existing rotary head_dim%4 contract. Recalculated144/head_dim36/FFN384; wide192/FFN576. Actual full-model parameter gap must remain<=5%; no width grid or quality search.',
        auxiliary='InfoNCE T=.1, weight0/.1. Both arms compute identical audio views/negative sets and contain identical projection heads. Center2s is excluded before encoding from a symmetric12s available surround; local/context clocks are center-relative for positives and negatives to prevent request/view timing identifiers. No dev/confirm audio in auxiliary training.',
        same_time='Report latest frozen0/250/500/750/1000 checkpoint whose actual training wall<=600s; never the best score. A coarse resource-envelope diagnostic, not exact matched-step or throughput proof.',
        source_sha256={n: sha(ROOT / n) for n in ('scripts/run_local_music_research.py', 'beatfcn/model/local_music_research.py',
            'beatfcn/model/local_access_canvas.py', 'beatfcn/model/mel_only_canvas.py', 'beatfcn/model/local_typed_canvas.py',
            'beatfcn/model/full_event_model.py', 'beatfcn/train/relative_canvas.py', 'scripts/replicate_canvas_finalist.py')},
        research_report_sha256=sha(ROOT / 'research_notes/LFM 3 技术架构/official_architecture_and_beatfcn.md'),
        cloud_compute_calls=0, confirmation_scored=False)
    if kind == 'ssl':
        plan['negative_map'] = negative_map(panel, row, parent['indices']['train64'])
    models = [make_model(plan, a, 'cpu') for a in arms]
    plan['parameters'] = {a: sum(p.numel() for p in m.parameters()) for a, m in zip(arms, models)}
    plan['initial_weight_sha256'] = {a: weight_digest(m) for a, m in zip(arms, models)}
    if max(plan['parameters'].values()) > 12_000_000:
        raise ValueError('Research parameter cap')
    if kind == 'depth' and max(plan['parameters'].values()) / min(plan['parameters'].values()) > 1.05:
        raise ValueError('Depth allocation parameter mismatch')
    if kind == 'ssl' and len(set(plan['initial_weight_sha256'].values())) != 1:
        raise ValueError('SSL arms must share all initial tensors')
    if kind == 'depth':
        left, right = [m.state_dict() for m in models]
        if any(not torch.equal(value, right[name]) for name, value in left.items() if not name.startswith('temporal.')):
            raise ValueError('Shared nontrunk initialization changed')
        plan['shared_boundary_initialization_equal'] = True
    del models
    write(plan_path, plan)
    for name in plan['source_sha256']:
        dest = art / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    shutil.copy2(plan_path, art / 'source/frozen-plan.json')
    report = dict(id=ident, status='running', arms={}, plan_sha256=sha(plan_path), cloud_compute_calls=0, confirmation_scored=False)
    write(report_path, report)
    def stop(*_):
        raise KeyboardInterrupt('External interruption')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    current = None
    try:
        for arm in arms:
            start = time.monotonic()
            model = make_model(plan, arm, 'mps')
            torch.mps.manual_seed(plan['seed'])
            opt = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=0.)
            current = dict(status='health_running', health_updates=0, quality_updates=0,
                           parameters=plan['parameters'][arm], curves=[], failures=[], health_weights_used=False,
                           initial_weight_sha256=weight_digest(model))
            report['arms'][arm] = current
            folder = art / arm
            folder.mkdir(parents=True, exist_ok=True)
            def checkpoint(stage, step):
                path = folder / f'{stage}-{step:04d}.pt'
                torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), stage=stage, updates=step,
                                plan_sha256=sha(plan_path)), path)
                return path
            specs = list(zip(plan['health_indices'], ('draft', 'single_condition', 'random_groups', 'middle')))
            specs.append((plan['empty_health_index'], 'draft'))
            batches = [(e, *input_batch(e, 'mps')) for i, view in specs for e in [row(i, view)]]
            current['input_audit'] = [dict(index=e['index'], tensors={k: dict(shape=list(v.shape),
                finite=bool(torch.isfinite(v).all()), min=float(v.min()), max=float(v.max()), mean=float(v.float().mean()),
                std=float(v.float().std()) if v.numel() > 1 else 0.) for k, v in inp.items()}) for e, inp, _ in batches]
            current['initial_diagnostic'] = diagnostics(model, plan, arm, batches, row, 'mps')
            cpu = make_model(plan, arm, 'cpu').eval()
            e, inp, tgt = batches[0]
            ci, ct = ({k: v.cpu() for k, v in d.items()} for d in (inp, tgt))
            model.eval()
            co, mo = cpu(ci), model(inp)
            error = max(float((co[k] - mo[k].cpu()).abs().max()) for k in co)
            cl, _ = loss_for(cpu, plan, arm, e, ci, ct, row, 'cpu')
            ml, _ = loss_for(model, plan, arm, e, inp, tgt, row, 'mps')
            cl.backward(); ml.backward()
            cp = dict(cpu.named_parameters()); gradient = 0.
            for name, p in model.named_parameters():
                if (p.grad is None) != (cp[name].grad is None):
                    raise ValueError('Gradient presence mismatch')
                if p.grad is not None:
                    gradient = max(gradient, float((p.grad.cpu() - cp[name].grad).abs().max()))
            current['parity'] = dict(forward=error, gradient=gradient)
            if error > 1e-4 or gradient > .003:
                raise ValueError('CPU/MPS parity')
            del cpu, co, mo, cl, ml, cp
            for step in range(1, 101):
                if time.monotonic() - start > plan['health_wall_s']:
                    raise TimeoutError('Health wall')
                model.train(); opt.zero_grad(set_to_none=True)
                for group in opt.param_groups: group['lr'] = rate(plan, step)
                total = 0.
                for e, inp, tgt in batches:
                    loss, _ = loss_for(model, plan, arm, e, inp, tgt, row, 'mps')
                    (loss / 5).backward(); total += float(loss.detach()) / 5
                grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)); opt.step()
                current['health_updates'] = step
                if step % 10 == 0:
                    current['curves'].append(dict(stage='health', step=step, loss=total, gradient=grad, wall_s=time.monotonic() - start))
                    write(report_path, report)
            final = diagnostics(model, plan, arm, batches, row, 'mps')
            current['final_diagnostic'] = final
            predictions = [generate_relative(model, inp, e['request'], deadline=start + plan['health_wall_s']) for e, inp, _ in batches]
            write(folder / 'health-predictions.json', predictions)
            gates = dict(loss=np.mean([v['loss'] for v in final]) <= .9 * np.mean([v['loss'] for v in current['initial_diagnostic']]),
                         complete=all(v['complete'] for v in predictions))
            current['health_gates'] = {k: bool(v) for k, v in gates.items()}
            checkpoint('health', 100)
            if not all(gates.values()):
                current['status'] = 'health_failed'; write(report_path, report)
                raise ValueError('Health fitting gate failed; no quality interpretation')
            current['health_wall_s'] = time.monotonic() - start
            del model, opt, batches
            gc.collect(); torch.mps.empty_cache()
            model = make_model(plan, arm, 'mps'); torch.mps.manual_seed(plan['seed'])
            if weight_digest(model) != current['initial_weight_sha256']:
                raise ValueError('Health weights not discarded')
            opt = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=0.)
            current.update(status='quality_running', checkpoint_training_wall_s={'0': 0.})
            checkpoint('quality', 0); quality_start = time.monotonic()
            for step, specs in enumerate(plan['stream'], 1):
                if time.monotonic() - start > plan['per_arm_total_wall_s'] - plan['evaluation_wall_s']:
                    raise TimeoutError('Full arm wall reserve evaluation')
                model.train(); opt.zero_grad(set_to_none=True)
                for group in opt.param_groups: group['lr'] = rate(plan, step)
                total = 0.
                for spec in specs:
                    e = row(spec['index'], spec['view'], spec['field_index'])
                    inp, tgt = input_batch(e, 'mps')
                    loss, _ = loss_for(model, plan, arm, e, inp, tgt, row, 'mps')
                    (loss / 2).backward(); total += float(loss.detach()) / 2
                grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)); opt.step()
                current['quality_updates'] = step
                if step % 25 == 0 or step == 1:
                    torch.mps.synchronize()
                    point = dict(stage='quality', step=step, loss=total, gradient=grad, wall_s=time.monotonic() - quality_start,
                        memory=dict(allocated=torch.mps.current_allocated_memory(), driver=torch.mps.driver_allocated_memory(), recommended=torch.mps.recommended_max_memory()))
                    current['curves'].append(point); write(report_path, report)
                    print(json.dumps(dict(arm=arm, **point)), flush=True)
                if step % 250 == 0:
                    checkpoint('quality', step)
                    current['checkpoint_training_wall_s'][str(step)] = time.monotonic() - quality_start
            current.update(status='evaluating', training_wall_s=time.monotonic() - quality_start,
                           final_weight_sha256=sha(folder / 'quality-1000.pt'))
            write(report_path, report)
            deadline = min(start + plan['per_arm_total_wall_s'], time.monotonic() + plan['evaluation_wall_s'])
            entry = dict(factory='mel')
            current['evaluations'] = {}
            tasks = dict(draft=(plan['indices']['dev'], {}), empty=(plan['indices']['empty_dev'], {}),
                completion=(plan['indices']['dev'][::2], dict(view='middle')),
                reference_masked=(plan['indices']['dev'][::2], dict(view='middle', masked=True)),
                density_low=(plan['indices']['dev'][::8], dict(bands=(2., 3.))),
                density_high=(plan['indices']['dev'][::8], dict(bands=(6., 8.))))
            for task, (indices, kwargs) in tasks.items():
                current['evaluations'][task] = evaluate(model, entry, plan, row, panel, indices, folder, task, deadline, **kwargs)
                write(report_path, report)
            current['coarse_swapped'] = coarse_counterfactual(model, plan, row, panel, folder, deadline)
            current['all_audio_swapped'] = coarse_counterfactual(model, plan, row, panel, folder, deadline, coarse_only=False)
            within = max(int(s) for s, wall in current['checkpoint_training_wall_s'].items() if wall <= plan['matched_time_s'])
            current['matched_time_checkpoint'] = within
            if within == 1000:
                current['within_600s'] = current['evaluations']['draft']
            else:
                saved = torch.load(folder / f'quality-{within:04d}.pt', map_location='cpu', weights_only=False)
                model.load_state_dict(saved['model']); del saved
                current['within_600s'] = evaluate(model, entry, plan, row, panel, plan['indices']['dev'], folder, 'within-600s', deadline)
            current['local_screen'] = screen_gates(current['evaluations']['draft'], plan['local_screen'])
            current.update(status='completed', total_wall_s=time.monotonic() - start)
            write(report_path, report)
            del model, opt
            gc.collect(); torch.mps.empty_cache()
        a, b = [report['arms'][arm]['evaluations']['draft'] for arm in arms]
        g = plan['hypothesis_gate']
        delta = b['full_note_f1'] - a['full_note_f1']
        onset = b['onset10'] - a['onset10']
        wins = sum(b['components'][c]['full_note_f1'] > v['full_note_f1'] for c, v in a['components'].items())
        within = report['arms'][arms[1]]['within_600s']['full_note_f1'] - report['arms'][arms[0]]['within_600s']['full_note_f1']
        context_gain = (b['full_note_f1'] - report['arms'][arms[1]]['coarse_swapped']['full_note_f1']) - (a['full_note_f1'] - report['arms'][arms[0]]['coarse_swapped']['full_note_f1'])
        support = delta >= g['full_gain'] and onset >= -g['onset_max_regression'] and wins >= g['component_wins']
        acoustic_gain = b['onset10'] - report['arms'][arms[1]]['all_audio_swapped']['onset10']
        support = support and acoustic_gain >= g['acoustic_onset_gain']
        support = support and (within > 0 if kind == 'depth' else context_gain >= g['useful_context_gain'])
        report.update(status='completed', comparison=dict(full_gain=delta, onset_gain=onset, component_wins=wins,
            within_600s_gain=within, useful_context_gain=context_gain, acoustic_onset_gain=acoustic_gain, hypothesis_supported=bool(support)))
        write(report_path, report)
    except BaseException as exc:
        if current is not None:
            failed_state = current['status']
            current['failures'].append(f'{type(exc).__name__}: {exc}')
            current['status'] = 'stopped'
            if 'opt' in locals():
                stage = 'quality' if failed_state in ('quality_running', 'evaluating', 'completed') else 'health'
                checkpoint(stage + '-interrupted', current[stage + '_updates'])
        report.update(status='stopped', failure=f'{type(exc).__name__}: {exc}')
        write(report_path, report)
        raise


if __name__ == '__main__':
    main()
