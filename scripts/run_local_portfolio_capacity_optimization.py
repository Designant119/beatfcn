"""Bounded optimization diagnostic after the 10M canvas 100-step underfit."""
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.local_typed_canvas import canvas_loss
from beatfcn.train.relative_canvas import generate_relative, relative_batch
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_scale import fresh, row_factory, weight_digest
from scripts.run_local_typed_canvas import score


ROOT = Path(__file__).resolve().parents[1]
PARENT_PLAN = ROOT / 'bench/rhythm-v2/local-portfolio-scale-v1.json'
PARENT_REPORT = ROOT / 'reports/local_portfolio_scale_v1.json'
PLAN = ROOT / 'bench/rhythm-v2/local-portfolio-capacity-optimization-v1.json'
REPORT = ROOT / 'reports/local_portfolio_capacity_optimization_v1.json'
ART = ROOT / 'artifacts/local-portfolio-v1/capacity-optimization'
LIVE = {}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def journal(message):
    with (ROOT / 'EXPERIMENT_LOG.md').open('a') as handle:
        handle.write('\n\nlocal-portfolio-capacity-optimization-v1 ' + message + '\n')


def evaluate(model, batches, started, wall):
    rows = []
    losses = []
    with torch.no_grad():
        for example, inputs, target in batches:
            loss, detail = canvas_loss(model(inputs), target, inputs)
            losses.append(dict(index=example['index'], loss=float(loss),
                **{key: float(value) for key, value in detail.items()}))
    for example, inputs, target in batches:
        prediction = generate_relative(model, inputs, example['request'], deadline=started + wall)
        events = [RhythmEvent(**event) for event in prediction['events']]
        rows.append(dict(index=example['index'], prediction=prediction,
            target=[asdict(event) for event in example['targets']], scores=score(events, example['targets'])))
    positive = [row['scores']['full_note_f1'] for row in rows if row['target']]
    gates = []
    for row in rows:
        prediction, target, scores = row['prediction'], row['target'], row['scores']
        events = prediction['events']
        if target:
            onset = (scores.get('onset_10ms') or {}).get('f1') or 0
            ratio = len(events) / len(target)
            gates.append(bool(prediction['complete'] and scores['full_note_f1'] >= .75 and onset >= .8
                and .5 <= ratio <= 2))
        else:
            gates.append(bool(prediction['complete'] and not events))
    gates.append(bool(positive) and sum(positive) / len(positive) >= .9)
    return dict(rows=rows, losses=losses, original_health_gates=gates,
        mean_loss=sum(row['loss'] for row in losses) / len(losses),
        positive_mean_full_f1=sum(positive) / len(positive) if positive else 0.)


def main():
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise ValueError('No silent backend fallback')
    if PLAN.exists() or REPORT.exists():
        raise FileExistsError('Preserve the capacity optimization diagnostic')
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.5)
    parent = json.loads(PARENT_PLAN.read_text())
    parent_report = json.loads(PARENT_REPORT.read_text())
    if parent_report['health_updates'] != 100 or parent_report['updates']['capacity64'] != 0:
        raise ValueError('Unexpected capacity parent state')
    plan = dict(id='local-portfolio-capacity-optimization-v1', seed=parent['seed'], steps=300,
        models={'capacity64': parent['models']['capacity64']}, density_reference=parent['density_reference'],
        health_indices=parent['health_indices'], empty_health_index=parent['empty_health_index'],
        learning_rate=.001, minimum_learning_rate=.0001, warmup=50, weight_decay=0., clip=1.,
        diagnostic_checkpoints=[100, 200, 300], health_wall_s=1800, cloud_compute_calls=0,
        hypothesis='TrainingDynamics package: warmup plus cosine schedule and a bounded 300-step diagnostic can remove the previously observed 100-step underfit.',
        fixed='Same capacity64 tensor initialization, five real windows, targets, canvas loss and expected-count readout. No development or confirmation scoring.',
        final_gate='The original scale-v1 per-window and positive-mean real-decode health gate at step 300; no retroactive pass at step 100.',
        source_sha256={name: sha(ROOT / name) for name in (
            'scripts/run_local_portfolio_capacity_optimization.py', 'scripts/run_local_portfolio_scale.py',
            'beatfcn/model/local_typed_canvas.py', 'beatfcn/train/relative_canvas.py',
            str(PARENT_PLAN.relative_to(ROOT)))})
    write(PLAN, plan)
    comparison = json.loads((ROOT / 'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text())
    panel, store = setup(comparison)
    row = row_factory(panel, store)
    model = fresh(plan, 'capacity64')
    initial_digest = weight_digest(model)
    if initial_digest != parent_report['initial_weight_sha256']['capacity64']:
        raise ValueError('Capacity initial weights no longer match scale-v1')
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=0.)
    LIVE.update(model=model, optimizer=optimizer, stage='optimization_diagnostic')
    specs = list(zip(plan['health_indices'], ('draft', 'single_condition', 'random_groups', 'middle')))
    specs.append((plan['empty_health_index'], 'draft'))
    batches = [(row(index, view),) for index, view in specs]
    batches = [(example, *relative_batch([example], 'mps')) for (example,) in batches]
    report = dict(id=plan['id'], status='prepared', plan_sha256=sha(PLAN),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        initial_weight_sha256=initial_digest, diagnostic_updates=0, quality_updates=0,
        curves=[], checkpoints={}, failures=[], wall_s={}, cloud_compute_calls=0,
        development_scored=False, confirmation_scored=False)
    write(REPORT, report)
    journal('冻结：TrainingDynamics单一配方包；同10.14M初值/5窗/损失/读出，lr.001+warmup50+300步余弦，100/200/300真实解码；300仍用原健康门，不追认旧100步，不评分开发/确认，质量未派发，云0。')

    def stop(*_):
        raise KeyboardInterrupt('External stop')

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    try:
        report['status'] = 'running'
        for step in range(1, plan['steps'] + 1):
            if time.monotonic() - started > plan['health_wall_s']:
                raise TimeoutError('Frozen capacity optimization wall')
            if step <= plan['warmup']:
                lr = plan['learning_rate'] * step / plan['warmup']
            else:
                progress = (step - plan['warmup']) / (plan['steps'] - plan['warmup'])
                lr = plan['minimum_learning_rate'] + (plan['learning_rate'] - plan['minimum_learning_rate']) * (1 + math.cos(math.pi * progress)) / 2
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr'] = lr
            total = 0.
            for example, inputs, target in batches:
                loss, _ = canvas_loss(model(inputs), target, inputs)
                (loss / len(batches)).backward()
                total += float(loss.detach()) / len(batches)
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), plan['clip'], error_if_nonfinite=True))
            optimizer.step()
            report['diagnostic_updates'] = step
            if step == 1 or step % 25 == 0:
                point = dict(step=step, learning_rate=lr, loss=total, preclip_gradient_norm=gradient,
                    wall_s=time.monotonic() - started)
                report['curves'].append(point)
                write(REPORT, report)
                print(json.dumps(dict(stage='capacity-optimization', **point)), flush=True)
            if step in plan['diagnostic_checkpoints']:
                model.eval()
                result = evaluate(model, batches, started, plan['health_wall_s'])
                report['checkpoints'][str(step)] = result
                write(ART / f'diagnostic-{step:04d}.json', result)
                path = ART / f'diagnostic-{step:04d}.pt'
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    config=plan['models']['capacity64'], plan_sha256=sha(PLAN), updates=step), path)
                write(REPORT, report)
        final = report['checkpoints']['300']
        report['health_passed'] = all(final['original_health_gates'])
        report['status'] = 'health_passed' if report['health_passed'] else 'health_failed'
        report['final_weight_sha256'] = weight_digest(model)
        report['wall_s']['diagnostic'] = time.monotonic() - started
        write(REPORT, report)
        journal(f'终态：{report["status"]}，300步mean_loss={final["mean_loss"]:.6f}，positive_fullF1={final["positive_mean_full_f1"]:.6f}；只约束本优化配方，不形成开发集宽度结论，质量未自动派发，云0。')
    except BaseException as exc:
        report.setdefault('failures', []).append(dict(stage=LIVE.get('stage'),
            error=f'{type(exc).__name__}: {exc}', diagnostic_updates=report['diagnostic_updates']))
        report['status'] = 'stopped'
        report['wall_s']['diagnostic'] = time.monotonic() - started
        write(REPORT, report)
        journal(f'停止：{report["failures"][-1]}；保留现场，不自动延长，云0。')
        raise


if __name__ == '__main__':
    main()
