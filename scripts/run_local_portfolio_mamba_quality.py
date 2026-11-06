"""Fresh train64 quality curve for the healthy attention-free Mamba canvas."""
import json
import math
import os
from pathlib import Path
import signal
import time

import torch

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_mamba_canvas import MambaCanvas
from beatfcn.model.local_typed_canvas import canvas_loss
from beatfcn.train.relative_canvas import relative_batch
from scripts.run_frontend_canvas_comparison import setup
import scripts.run_local_portfolio_data_scale as evaluator
from scripts.run_local_portfolio_mamba_health import sha, write
from scripts.run_local_portfolio_scale import row_factory, weight_digest


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'bench/rhythm-v2/local-portfolio-scale-v1.json'
HEALTH_PLAN = ROOT / 'bench/rhythm-v2/local-portfolio-mamba-health-v1.json'
HEALTH_REPORT = ROOT / 'reports/local_portfolio_mamba_health_v1.json'
PLAN = ROOT / 'bench/rhythm-v2/local-portfolio-mamba-quality-v1.json'
REPORT = ROOT / 'reports/local_portfolio_mamba_quality_v1.json'
ART = ROOT / 'artifacts/local-portfolio-v1/mamba-quality'
LIVE = {}


def journal(message):
    with (ROOT / 'EXPERIMENT_LOG.md').open('a') as handle:
        handle.write('\n\nlocal-portfolio-mamba-quality-v1 ' + message + '\n')


def checkpoint(model, optimizer, plan, step, stage='quality'):
    path = ART / 'mamba' / f'{stage}-{step:04d}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=plan['model'],
        plan_sha256=sha(PLAN), updates=step, cpu_rng=torch.get_rng_state(), mps_rng=torch.mps.get_rng_state()), path)


def main():
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise ValueError('No silent backend fallback')
    if PLAN.exists() or REPORT.exists():
        raise FileExistsError('Preserve the Mamba quality run')
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.5)
    parent = json.loads(PARENT.read_text())
    health_plan = json.loads(HEALTH_PLAN.read_text())
    health = json.loads(HEALTH_REPORT.read_text())
    if not health['health_passed'] or health['health_updates'] != 100:
        raise ValueError('Mamba quality requires the passed independent health probe')
    plan = dict(id='local-portfolio-mamba-quality-v1', seed=health_plan['seed'], steps=2000,
        effective_batch=2, model=health_plan['model'], density_reference=health_plan['density_reference'],
        indices=parent['indices'], learning_rate=.001, minimum_learning_rate=.0001, warmup=50,
        weight_decay=0., clip=1., quality_checkpoints=[1000, 2000], diagnostic_checkpoints=[250, 500, 1500],
        training_wall_s=7200, evaluation_wall_s=1200, cloud_compute_calls=0,
        shifts_us=parent['shifts_us'], views=parent['views'], sample_stream=parent['sample_stream']['train64'],
        predecessor='local-portfolio-mamba-health-v1', health_plan_sha256=sha(HEALTH_PLAN),
        health_checkpoint_sha256=sha(ROOT / 'artifacts/local-portfolio-v1/mamba-health/health-0100.pt'),
        health_weights_used=False, confirmation_policy='confirm indices stay sealed and are not scored.',
        comparison_policy='Report matched-update small64 data-scale results as context, not as a single-module causal ablation. No early F1 stop; bounded 1000/2000 curve.',
        source_sha256={name: sha(ROOT / name) for name in (
            'scripts/run_local_portfolio_mamba_quality.py', 'scripts/run_local_portfolio_data_scale.py',
            'beatfcn/model/local_mamba_canvas.py', 'beatfcn/model/mamba2_reference.py',
            'beatfcn/model/local_typed_canvas.py', 'beatfcn/train/relative_canvas.py',
            str(PARENT.relative_to(ROOT)), str(HEALTH_PLAN.relative_to(ROOT)))})
    write(PLAN, plan)
    comparison = json.loads((ROOT / 'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text())
    panel, store = setup(comparison)
    row = row_factory(panel, store)
    evaluator.ART = ART
    evaluator.LIVE['panel'] = panel
    torch.manual_seed(plan['seed'])
    model = MambaCanvas(FullEventConfig(**plan['model']), density_reference=plan['density_reference']).to('mps')
    initial_digest = weight_digest(model)
    if initial_digest != health['initial_weight_sha256']:
        raise ValueError('Fresh quality initialization differs from the health initialization')
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=plan['weight_decay'])
    LIVE.update(model=model, optimizer=optimizer, stage='quality')
    report = dict(id=plan['id'], status='prepared', plan_sha256=sha(PLAN), parameters=health['parameters'],
        initial_weight_sha256=initial_digest, health_weights_used=False, updates=0, curves=[], evaluations={},
        failures=[], wall_s={}, cloud_compute_calls=0, confirmation_scored=False,
        exposure=dict(windows=0, events=0, indices={}, types={}))
    write(REPORT, report)
    journal(f'冻结：Mamba健康通过后fresh同初值train64质量曲线，参数{report["parameters"]}，健康权重不加载；2000步/1000复盘，开发原面板，确认密封，无早期F1停止，云0。')

    def stop(*_):
        raise KeyboardInterrupt('External stop')

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    evaluation_s = 0.
    try:
        report['status'] = 'training'
        for step in range(1, plan['steps'] + 1):
            if time.monotonic() - started > plan['training_wall_s']:
                raise TimeoutError('Frozen Mamba quality wall includes evaluations')
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
            for spec in plan['sample_stream'][step - 1]:
                example = row(spec['index'], spec['view'], spec['field_index'])
                inputs, target = relative_batch([example], 'mps')
                loss, _ = canvas_loss(model(inputs), target, inputs)
                (loss / 2).backward()
                total += float(loss.detach()) / 2
                exposure = report['exposure']
                exposure['windows'] += 1
                exposure['events'] += len(example['targets'])
                key = str(spec['index'])
                exposure['indices'][key] = exposure['indices'].get(key, 0) + 1
                for event in example['targets']:
                    exposure['types'][event.type] = exposure['types'].get(event.type, 0) + 1
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), plan['clip'], error_if_nonfinite=True))
            optimizer.step()
            report['updates'] = step
            if step == 1 or step % 25 == 0:
                point = dict(step=step, learning_rate=lr, loss=total, preclip_gradient_norm=gradient,
                    wall_s=time.monotonic() - started)
                report['curves'].append(point)
                report['wall_s']['training'] = point['wall_s']
                write(REPORT, report)
                print(json.dumps(dict(stage='mamba-quality', **point)), flush=True)
            if step in plan['quality_checkpoints'] + plan['diagnostic_checkpoints']:
                checkpoint(model, optimizer, plan, step)
            if step in plan['quality_checkpoints']:
                moment = time.monotonic()
                deadline = min(started + plan['training_wall_s'], moment + plan['evaluation_wall_s'] - evaluation_s)
                report['evaluations'][str(step)] = evaluator.evaluate({'mamba': model}, plan, row,
                    f'quality-{step:04d}', deadline, full=step == plan['steps'])
                evaluation_s += time.monotonic() - moment
                write(REPORT, report)
        report.update(status='completed', complete=True, confirmation_scored=False)
        report['wall_s'].update(training=time.monotonic() - started, evaluation=evaluation_s)
        report['final_weight_sha256'] = sha(ART / 'mamba' / 'quality-2000.pt')
        write(REPORT, report)
        journal(f'终态：updates={report["updates"]}，1000/2000开发评价完成，确认未评分，云0；结果{REPORT.relative_to(ROOT)}。')
    except BaseException as exc:
        report.setdefault('failures', []).append(dict(stage=LIVE.get('stage'),
            error=f'{type(exc).__name__}: {exc}', updates=report['updates']))
        report['status'] = 'stopped'
        checkpoint(model, optimizer, plan, report['updates'], 'partial-quality')
        write(REPORT, report)
        journal(f'停止：{report["failures"][-1]}；保留现场，不自动延长，云0。')
        raise


if __name__ == '__main__':
    main()
