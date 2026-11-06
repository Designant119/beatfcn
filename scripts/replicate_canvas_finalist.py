"""One frozen finalist's independent initialization, then the sealed panel.

Supports the completed canvas/SSD/access recipes without changing their
schedule horizon, stream, objective, input contract or original readout.
Other method families require their own explicit adapter, never a fallback.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from beatfcn.model.direct_song_canvas import DirectSongCanvas
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_access_canvas import AccessCanvas
from beatfcn.model.local_mamba_canvas import MambaCanvas
from beatfcn.model.local_typed_canvas import LocalTypedCanvas, canvas_loss
from beatfcn.model.mel_only_canvas import MelOnlyCanvas
from beatfcn.model.local_music_research import BudgetAllocationCanvas, ContextMatchingCanvas
from beatfcn.model.sample import match_f1
from beatfcn.train.canonical_probability import explicit_row
from beatfcn.train.mert_canvas_features import MertCanvasFeatures
from beatfcn.train.relative_canvas import relative_batch, generate_relative
from scripts.portfolio_terminal_catalog import eligible
from scripts.review_frontend_canvas_comparison import metrics
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_group_diffusion_health import optimizer_for, rate
from scripts.run_local_portfolio_scale import row_factory, weight_digest, sha, write
from scripts.run_local_typed_canvas import score


ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / 'reports/local_portfolio_finalist_selection_v1.json'
POLICY = ROOT / 'bench/rhythm-v2/local-portfolio-final-selection-v1.json'


def make_model(entry, original, seed, device):
    torch.manual_seed(seed)
    config = original.get('model', original.get('models', {}).get(entry['arm']))
    config = FullEventConfig(**config)
    kwargs = dict(density_reference=original['density_reference'])
    family = entry['factory']
    if family == 'access':
        model = AccessCanvas(config, access=entry['arm'], radius=original['window_radius'], **kwargs)
    elif family == 'depth_budget':
        width, depth = original['allocations'][entry['arm']]
        model = BudgetAllocationCanvas(config, width=width, depth=depth, **kwargs)
    elif family == 'ssl_context':
        model = ContextMatchingCanvas(config, **kwargs)
    elif family in ('mamba', 'direct_song', 'canvas', 'mel'):
        cls = dict(mamba=MambaCanvas, direct_song=DirectSongCanvas,
                   canvas=LocalTypedCanvas, mel=MelOnlyCanvas)[family]
        model = cls(config, **kwargs)
    else:
        raise ValueError('No scientifically equivalent replication adapter: ' + family)
    return model.to(device)


def is_explicit(entry):
    return entry['id'].startswith(('direct-song-', 'local-attention-', 'local-concat-', 'music-'))


def batch(row, entry, device, masked=False):
    inputs, target = relative_batch([row], device, mask_reference=masked)
    if entry['factory'] in ('access', 'mel', 'depth_budget', 'ssl_context'):
        inputs = {k: v for k, v in inputs.items() if not k.startswith(('local', 'song'))}
    return inputs, target


@torch.no_grad()
def diagnostics(model, batches):
    model.eval()
    return [dict(index=e['index'], loss=float(loss),
                 parts={k: float(v) for k, v in parts.items()})
            for e, inp, tgt in batches
            for loss, parts in [canvas_loss(model(inp), tgt, inp)]]


@torch.no_grad()
def evaluate(model, entry, original, row, panel, indices, art, tag, deadline,
             *, view='draft', masked=False, bands=None):
    records = []
    for index in indices:
        if time.monotonic() > deadline:
            raise TimeoutError('Fixed finalist evaluation wall')
        e = row(index, view, 0, bands)
        # The model forward receives request/audio/reference inputs only.
        inp, _ = batch({**e, 'targets': []}, entry, 'mps', masked)
        prediction = generate_relative(model, inp, e['request'], deadline=deadline)
        pred = [RhythmEvent(**v) for v in prediction['events']]
        start = e['request'].start_us
        width = e['request'].end_us - start
        shifts = [match_f1([start + (v.onset_us - start + d) % width for v in pred],
                          [v.onset_us for v in e['targets']], 10000)[3]
                  for d in original['shifts_us']]
        records.append(dict(index=index, component=panel.records[index]['component'],
            view=view, request=asdict(e['request']), prediction=prediction,
            target=[asdict(v) for v in e['targets']], scores=score(pred, e['targets']),
            shifted_onset10=shifts))
        write(art / f'{tag}.json', records)
    return metrics(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True, choices=(1, 2))
    rank = parser.parse_args().rank
    selection = json.loads(SELECTION.read_text())
    policy = json.loads(POLICY.read_text())
    if sha(POLICY) != selection['policy_sha256']:
        raise ValueError('Frozen selection policy changed')
    entry = selection['finalist_identities'][rank - 1]
    seed = selection['second_seed_by_rank'][rank - 1]
    ident = f'local-finalist-r{rank}-seed2-v1'
    art = ROOT / 'artifacts' / ident
    report_path = ROOT / f'reports/local_finalist_r{rank}_seed2_v1.json'
    plan_path = ROOT / f'bench/rhythm-v2/{ident}.json'
    if report_path.exists() or plan_path.exists():
        raise FileExistsError('No retry/reselection of a finalist')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise ValueError('No silent backend fallback')
    for key in ('plan', 'checkpoint', 'report'):
        if sha(ROOT / entry[key]) != entry[key + '_sha256']:
            raise ValueError('Selected identity drift: ' + key)
    original = json.loads((ROOT / entry['plan']).read_text())
    for name, expected in original.get('source_sha256', {}).items():
        if name.startswith('beatfcn/') and sha(ROOT / name) != expected:
            raise ValueError('Original model/input implementation drift: ' + name)
    source_model = original.get('model', original.get('models', {}).get(entry['arm']))
    schedule_steps = original.get('quality_steps', original.get('steps'))
    stream = original.get('sample_stream', original.get('stream'))
    parent = json.loads((ROOT / 'bench/rhythm-v2/local-portfolio-scale-v1.json').read_text())
    if stream is None:
        stream = parent['sample_stream'][entry['arm']]
    stream = stream[:entry['steps']]
    plan = dict(id=ident, selected=entry, seed=seed, model=source_model,
                selection_sha256=sha(SELECTION), policy_sha256=sha(POLICY),
                original_schedule_horizon=schedule_steps, quality_steps=schedule_steps,
                original_terminal_updates=entry['steps'], stream=stream,
                health_updates=100, health_weights_used=False,
                learning_rate=original['learning_rate'], minimum_learning_rate=original['minimum_learning_rate'],
                warmup=original['warmup'], weight_decay=original['weight_decay'],
                health_wall_s=min(1500, original.get('health_wall_s', original.get('health_wall_per_arm_s', 1500))),
                quality_wall_s=min(7200, original.get('quality_wall_s', original.get('quality_wall_per_arm_s', original.get('training_wall_s', 7200)))),
                evaluation_wall_s=min(1200, original.get('evaluation_wall_s', original.get('evaluation_wall_per_arm_s', 1200))),
                meaning='Independent initialization of the exact selected recipe; old SSD1000 retains its original2000step cosine horizon, not a new1000step decay.',
                cloud_compute_calls=0)
    sources = ('scripts/replicate_canvas_finalist.py', 'scripts/portfolio_terminal_catalog.py',
               'beatfcn/model/local_mamba_canvas.py', 'beatfcn/model/direct_song_canvas.py',
               'beatfcn/model/local_access_canvas.py', 'beatfcn/model/mel_only_canvas.py',
               'beatfcn/model/local_music_research.py',
               'beatfcn/model/local_typed_canvas.py', 'beatfcn/model/full_event_model.py',
               'beatfcn/model/mamba2_reference.py', 'beatfcn/model/fullstack_mamba.py',
               'beatfcn/train/relative_canvas.py', 'beatfcn/train/local_typed_canvas.py',
               'beatfcn/train/local_bundle.py', 'beatfcn/train/canonical_probability.py')
    plan['source_sha256'] = {n: sha(ROOT / n) for n in sources}
    write(plan_path, plan)
    for name in sources:
        dest = art / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    shutil.copy2(plan_path, art / 'source/frozen-plan.json')
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.5)
    panel, store = setup(json.loads((ROOT / 'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()))
    if entry['arm'] == 'short32':
        store = MertCanvasFeatures(ROOT, store.pupu, 'reports/local_mert_context_cache_v1.json')
    base_row = row_factory(panel, store)

    def row(*args, **kwargs):
        value = base_row(*args, **kwargs)
        return explicit_row(value) if is_explicit(entry) else value

    def training_loss(current, example, inputs, target, device):
        loss, parts = canvas_loss(current(inputs), target, inputs)
        if entry['factory'] == 'ssl_context':
            negatives = [batch(row(i, 'draft'), entry, device)[0]
                         for i in original['negative_map'][str(example['index'])]]
            auxiliary, _ = current.matching_loss(inputs, negatives)
            loss = loss + original['auxiliary_weight'][entry['arm']] * auxiliary
            parts = dict(parts, matching=auxiliary)
        return loss, parts

    @torch.no_grad()
    def diagnostic_batches(current, values):
        current.eval()
        return [dict(index=e['index'], loss=float(loss), parts={k: float(v) for k, v in parts.items()})
                for e, inp, tgt in values
                for loss, parts in [training_loss(current, e, inp, tgt, 'mps')]]

    health_indices = original.get('health_indices', parent['health_indices'])
    health_specs = list(zip(health_indices, ('draft', 'single_condition', 'random_groups', 'middle')))
    health_specs.append((original.get('empty_health_index', parent['empty_health_index']), 'draft'))
    model = make_model(entry, original, seed, 'mps')
    initial = weight_digest(model)
    if sum(p.numel() for p in model.parameters()) != entry['parameters']:
        raise ValueError('Exact parameter count changed')
    report = dict(id=ident, candidate=entry['id'], status='health_running', parameters=entry['parameters'],
                  health_updates=0, quality_updates=0, curves=[], failures=[],
                  initial_weight_sha256=initial, plan_sha256=sha(plan_path),
                  health_weights_used=False, confirmation_scored=False, cloud_compute_calls=0)
    write(report_path, report)

    def stop(*_):
        raise KeyboardInterrupt('External stop')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    stage = 'health'
    started = time.monotonic()

    def checkpoint(opt, step):
        dest = art / f'{stage}-{step:04d}.pt'
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), updates=step,
                        plan_sha256=sha(plan_path), cpu_rng=torch.get_rng_state(),
                        mps_rng=torch.mps.get_rng_state()), dest)
        return dest

    def optimizer():
        if original['weight_decay'] == 0:
            return torch.optim.AdamW(model.parameters(), lr=original['learning_rate'], weight_decay=0.)
        return optimizer_for(model, original)

    try:
        batches = [(e, *batch(e, entry, 'mps')) for index, view in health_specs
                   for e in [row(index, view)]]
        report['input_audit'] = [dict(index=e['index'], tensors={k: dict(shape=list(v.shape),
                finite=bool(torch.isfinite(v).all()), min=float(v.min()), max=float(v.max()),
                mean=float(v.float().mean()), std=float(v.float().std()) if v.numel() > 1 else 0.)
                for k, v in inp.items()}) for e, inp, _ in batches]
        report['initial_diagnostic'] = diagnostic_batches(model, batches)
        cpu = make_model(entry, original, seed, 'cpu').eval()
        _, inp, tgt = batches[0]
        ci, ct = ({k: v.cpu() for k, v in d.items()} for d in (inp, tgt))
        model.eval()
        co = cpu(ci)
        mo = model(inp)
        forward = max(float((co[k] - mo[k].cpu()).abs().max()) for k in co)
        cl, _ = training_loss(cpu, batches[0][0], ci, ct, 'cpu')
        ml, _ = training_loss(model, batches[0][0], inp, tgt, 'mps')
        cl.backward()
        ml.backward()
        cp = dict(cpu.named_parameters())
        gradient = 0.
        for name, param in model.named_parameters():
            if (param.grad is None) != (cp[name].grad is None):
                raise ValueError('Gradient presence parity')
            if param.grad is not None:
                gradient = max(gradient, float((param.grad.cpu() - cp[name].grad).abs().max()))
        report['parity'] = dict(forward=forward, gradient=gradient)
        if forward > 1e-4 or gradient > .003:
            raise ValueError('CPU/MPS parity')
        del cpu, co, mo, cl, ml, cp
        opt = optimizer()
        for step in range(1, 101):
            if time.monotonic() - started > plan['health_wall_s']:
                raise TimeoutError('Health wall')
            model.train()
            opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:
                group['lr'] = rate(plan, step)
            total = 0.
            for e, inp, tgt in batches:
                loss, _ = training_loss(model, e, inp, tgt, 'mps')
                (loss / len(batches)).backward()
                total += float(loss.detach()) / len(batches)
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
            opt.step()
            report['health_updates'] = step
            if step == 1 or step % 10 == 0:
                torch.mps.synchronize()
                point = dict(stage=stage, step=step, loss=total, gradient=grad, wall_s=time.monotonic() - started)
                report['curves'].append(point)
                write(report_path, report)
                print(json.dumps(point), flush=True)
        final = diagnostic_batches(model, batches)
        report['final_diagnostic'] = final
        preds = []
        for e, inp, _ in batches:
            pred = generate_relative(model, inp, e['request'], deadline=started + plan['health_wall_s'])
            preds.append(dict(index=e['index'], prediction=pred, target=[asdict(v) for v in e['targets']]))
        write(art / 'health-predictions.json', preds)
        first = report['initial_diagnostic']
        gates = dict(loss=np.mean([v['loss'] for v in final]) <= .9 * np.mean([v['loss'] for v in first]),
                     complete=all(v['prediction']['complete'] for v in preds),
                     count=np.mean([v['parts']['count'] for v in final]) <= 1.05 * np.mean([v['parts']['count'] for v in first]),
                     offset=np.mean([v['parts']['offset'] for v in final]) <= 1.05 * np.mean([v['parts']['offset'] for v in first]))
        report['health_gates'] = {k: bool(v) for k, v in gates.items()}
        report['health_wall_s'] = time.monotonic() - started
        checkpoint(opt, 100)
        if not all(gates.values()):
            report['status'] = 'health_failed'
            write(report_path, report)
            return
        del model, opt, batches, preds
        gc.collect()
        torch.mps.empty_cache()
        model = make_model(entry, original, seed, 'mps')
        torch.mps.manual_seed(seed)
        if weight_digest(model) != initial:
            raise ValueError('Health weights were not discarded')
        opt = optimizer()
        stage = 'quality'
        started = time.monotonic()
        report['status'] = 'quality_running'
        write(report_path, report)
        stream_digest = hashlib.sha256()
        target_digest = hashlib.sha256()
        for step, specs in enumerate(stream, 1):
            if time.monotonic() - started > plan['quality_wall_s']:
                raise TimeoutError('Quality wall')
            model.train()
            opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:
                group['lr'] = rate(plan, step)
            total = 0.
            for spec in specs:
                e = row(spec['index'], spec['view'], spec['field_index'])
                inp, tgt = batch(e, entry, 'mps')
                stream_digest.update(json.dumps(spec, sort_keys=True).encode())
                target_digest.update(json.dumps([asdict(v) for v in e['targets']], sort_keys=True).encode())
                target_digest.update(inp['controls'].detach().cpu().numpy().tobytes())
                loss, _ = training_loss(model, e, inp, tgt, 'mps')
                (loss / len(specs)).backward()
                total += float(loss.detach()) / len(specs)
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
            opt.step()
            report['quality_updates'] = step
            if step == 1 or step % 25 == 0:
                torch.mps.synchronize()
                point = dict(stage=stage, step=step, loss=total, gradient=grad, wall_s=time.monotonic() - started,
                             mps_allocated=torch.mps.current_allocated_memory(), mps_driver=torch.mps.driver_allocated_memory())
                report['curves'].append(point)
                write(report_path, report)
                print(json.dumps(point), flush=True)
            if step % 250 == 0:
                checkpoint(opt, step)
        weight = checkpoint(opt, entry['steps'])
        report.update(status='evaluating', final_weight_sha256=sha(weight), training_wall_s=time.monotonic() - started,
                      supervision_stream_sha256=stream_digest.hexdigest(), actual_target_controls_sha256=target_digest.hexdigest())
        write(report_path, report)
        report['development'] = evaluate(model, entry, original, row, panel, original['indices']['dev'], art,
            'seed2-development', time.monotonic() + plan['evaluation_wall_s'])
        report['reproducibility_gates'] = eligible(report['development'], policy)
        report['reproduced'] = all(report['reproducibility_gates'].values())
        # Confirmation is deliberately a separate operation after all finalist
        # identities and second-seed outcomes are recorded; no adaptive opening.
        report.update(status='completed', quality_wall_s=time.monotonic() - started)
        write(report_path, report)
    except BaseException as exc:
        report.update(status='stopped', stopped_stage=stage, wall_s=time.monotonic() - started)
        report['failures'].append(f'{type(exc).__name__}: {exc}')
        if 'opt' in locals():
            checkpoint(opt, report[f'{stage}_updates'])
        write(report_path, report)
        raise


if __name__ == '__main__':
    main()
