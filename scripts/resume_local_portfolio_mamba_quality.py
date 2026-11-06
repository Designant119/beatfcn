"""Resume the interrupted Mamba quality curve from its exact 1500-step checkpoint."""
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
from scripts.run_local_portfolio_scale import row_factory


ROOT = Path(__file__).resolve().parents[1]
PARENT_PLAN = ROOT / 'bench/rhythm-v2/local-portfolio-mamba-quality-v1.json'
PARENT_REPORT = ROOT / 'reports/local_portfolio_mamba_quality_v1.json'
RESUME_CHECKPOINT = ROOT / 'artifacts/local-portfolio-v1/mamba-quality/mamba/quality-1500.pt'
PLAN = ROOT / 'bench/rhythm-v2/local-portfolio-mamba-quality-resume-v1.json'
REPORT = ROOT / 'reports/local_portfolio_mamba_quality_resume_v1.json'
ART = ROOT / 'artifacts/local-portfolio-v1/mamba-quality-resume'
LIVE = {}


def journal(message):
    with (ROOT / 'EXPERIMENT_LOG.md').open('a') as handle:
        handle.write('\n\nlocal-portfolio-mamba-quality-resume-v1 ' + message + '\n')


def checkpoint(model, optimizer, plan, step):
    path = ART / 'mamba' / f'quality-{step:04d}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=plan['model'],
        plan_sha256=sha(PLAN), updates=step, cpu_rng=torch.get_rng_state(), mps_rng=torch.mps.get_rng_state()), path)


def main():
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise ValueError('No silent backend fallback')
    if PLAN.exists() or REPORT.exists():
        raise FileExistsError('Preserve the Mamba quality resume run')
    parent = json.loads(PARENT_PLAN.read_text())
    parent_report = json.loads(PARENT_REPORT.read_text())
    if parent_report['status'] != 'training' or parent_report['updates'] != 1775:
        raise ValueError('Expected the interrupted 1775-step parent record')
    saved = torch.load(RESUME_CHECKPOINT, map_location='cpu', weights_only=False)
    if saved['plan_sha256'] != sha(PARENT_PLAN) or saved['updates'] != 1500:
        raise ValueError('Unexpected Mamba resume checkpoint')
    plan = dict(id='local-portfolio-mamba-quality-resume-v1', seed=parent['seed'], steps=2000,
        start_step=1500, model=parent['model'], density_reference=parent['density_reference'],
        indices=parent['indices'], learning_rate=parent['learning_rate'],
        minimum_learning_rate=parent['minimum_learning_rate'], warmup=parent['warmup'],
        weight_decay=parent['weight_decay'], clip=parent['clip'], quality_checkpoints=[2000],
        training_wall_s=1800, evaluation_wall_s=1200, cloud_compute_calls=0,
        shifts_us=parent['shifts_us'], sample_stream=parent['sample_stream'],
        parent_plan_sha256=sha(PARENT_PLAN), parent_report_sha256=sha(PARENT_REPORT),
        resume_checkpoint_sha256=sha(RESUME_CHECKPOINT),
        interruption='Process was absent at inspection; parent last persisted step 1775 without a terminal record. Resume from the last exact checkpoint at 1500 and re-execute 1501–1775.',
        confirmation_policy='confirm indices stay sealed and are not scored.',
        source_sha256={name: sha(ROOT / name) for name in (
            'scripts/resume_local_portfolio_mamba_quality.py', 'scripts/run_local_portfolio_data_scale.py',
            'beatfcn/model/local_mamba_canvas.py', 'beatfcn/model/mamba2_reference.py',
            'beatfcn/model/local_typed_canvas.py', 'beatfcn/train/relative_canvas.py',
            str(PARENT_PLAN.relative_to(ROOT)))})
    write(PLAN, plan)
    comparison = json.loads((ROOT / 'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text())
    panel, store = setup(comparison)
    row = row_factory(panel, store)
    evaluator.ART = ART
    evaluator.LIVE['panel'] = panel
    model = MambaCanvas(FullEventConfig(**plan['model']), density_reference=plan['density_reference'])
    model.load_state_dict(saved['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=plan['weight_decay'])
    optimizer.load_state_dict(saved['optimizer'])
    model = model.to('mps')
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to('mps')
    torch.set_rng_state(saved['cpu_rng'])
    torch.mps.set_rng_state(saved['mps_rng'])
    LIVE.update(model=model, optimizer=optimizer, stage='quality_resume')
    report = dict(id=plan['id'], status='prepared', plan_sha256=sha(PLAN),
        parent_updates_at_interruption=1775, resume_from=1500, updates=1500,
        reexecuted_updates=0, curves=[], evaluations={}, failures=[], wall_s={}, cloud_compute_calls=0,
        confirmation_scored=False, exposure=dict(windows=0, events=0, indices={}, types={}))
    write(REPORT, report)
    journal('恢复：父进程消失、最后落盘1775但无终态；从精确1500检查点恢复Adam/RNG，重跑1501–2000（含此前未 checkpoint 的275步），只评2000，确认密封，云0。')

    def stop(*_):
        raise KeyboardInterrupt('External stop')

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    evaluation_s = 0.
    try:
        report['status'] = 'training'
        torch.set_num_threads(4)
        torch.mps.set_per_process_memory_fraction(.5)
        for step in range(plan['start_step'] + 1, plan['steps'] + 1):
            if time.monotonic() - started > plan['training_wall_s']:
                raise TimeoutError('Frozen Mamba resume wall includes evaluation')
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
            report['reexecuted_updates'] = step - plan['start_step']
            if step % 25 == 0:
                point = dict(step=step, learning_rate=lr, loss=total, preclip_gradient_norm=gradient,
                    wall_s=time.monotonic() - started)
                report['curves'].append(point)
                report['wall_s']['training'] = point['wall_s']
                write(REPORT, report)
                print(json.dumps(dict(stage='mamba-quality-resume', **point)), flush=True)
        checkpoint(model, optimizer, plan, 2000)
        moment = time.monotonic()
        deadline = min(started + plan['training_wall_s'], moment + plan['evaluation_wall_s'])
        report['evaluations']['2000'] = evaluator.evaluate({'mamba': model}, plan, row,
            'quality-2000', deadline, full=True)
        evaluation_s = time.monotonic() - moment
        report.update(status='completed', complete=True, confirmation_scored=False)
        report['wall_s'].update(training=time.monotonic() - started, evaluation=evaluation_s)
        report['final_weight_sha256'] = sha(ART / 'mamba' / 'quality-2000.pt')
        write(REPORT, report)
        journal('终态：从1500恢复并完成2000与完整开发评价，确认未评分，云0。')
    except BaseException as exc:
        report.setdefault('failures', []).append(dict(stage=LIVE.get('stage'),
            error=f'{type(exc).__name__}: {exc}', updates=report['updates']))
        report['status'] = 'stopped'
        checkpoint(model, optimizer, plan, report['updates'])
        write(REPORT, report)
        journal(f'停止：{report["failures"][-1]}；保留现场，不自动延长，云0。')
        raise


if __name__ == '__main__':
    main()
