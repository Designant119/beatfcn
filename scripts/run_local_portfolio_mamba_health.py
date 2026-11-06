"""Real fixed-batch health probe for the attention-free Mamba-2 canvas."""
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import time

import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_mamba_canvas import MambaCanvas
from beatfcn.model.local_typed_canvas import canvas_loss
from beatfcn.train.relative_canvas import generate_relative, relative_batch
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_scale import row_factory, weight_digest
from scripts.run_local_typed_canvas import score


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'bench/rhythm-v2/local-portfolio-scale-v1.json'
PLAN = ROOT / 'bench/rhythm-v2/local-portfolio-mamba-health-v1.json'
REPORT = ROOT / 'reports/local_portfolio_mamba_health_v1.json'
ART = ROOT / 'artifacts/local-portfolio-v1/mamba-health'
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
        handle.write('\n\nlocal-portfolio-mamba-health-v1 ' + message + '\n')


def diagnostic(model, batches):
    rows = []
    with torch.no_grad():
        for example, inputs, target in batches:
            loss, detail = canvas_loss(model(inputs), target, inputs)
            rows.append(dict(index=example['index'], loss=float(loss),
                **{key: float(value) for key, value in detail.items()}))
    keys = ('loss', 'count', 'existence', 'offset', 'tail', 'total', 'interval')
    return dict(rows=rows, mean={key: sum(row[key] for row in rows) / len(rows) for key in keys})


def parity(model, optimizer, batch, plan):
    example, inputs, target = batch
    cpu_model = MambaCanvas(FullEventConfig(**plan['model']), density_reference=plan['density_reference'])
    cpu_model.load_state_dict({key: value.detach().cpu() for key, value in model.state_dict().items()})
    cpu_inputs = {key: value.cpu() for key, value in inputs.items()}
    cpu_target = {key: value.cpu() for key, value in target.items()}
    started = time.monotonic()
    cpu_output = cpu_model(cpu_inputs)
    cpu_loss, _ = canvas_loss(cpu_output, cpu_target, cpu_inputs)
    cpu_loss.backward()
    cpu_wall = time.monotonic() - started
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    mps_output = model(inputs)
    mps_loss, _ = canvas_loss(mps_output, target, inputs)
    mps_loss.backward()
    torch.mps.synchronize()
    mps_wall = time.monotonic() - started
    forward_error = max(float((cpu_output[key] - mps_output[key].cpu()).abs().max()) for key in cpu_output)
    gradient_error = 0.
    cpu_parameters = dict(cpu_model.named_parameters())
    for name, parameter in model.named_parameters():
        cpu_gradient = cpu_parameters[name].grad
        if cpu_gradient is None or parameter.grad is None:
            if (cpu_gradient is None) != (parameter.grad is None):
                raise ValueError(f'CPU/MPS gradient presence differs: {name}')
            continue
        gradient_error = max(gradient_error, float((cpu_gradient - parameter.grad.cpu()).abs().max()))
    result = dict(updates=0, forward_max_error=forward_error, gradient_max_error=gradient_error,
        loss_absolute_error=abs(float(cpu_loss) - float(mps_loss)), cpu_forward_backward_s=cpu_wall,
        mps_forward_backward_s=mps_wall, process_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        mps_allocated_bytes=torch.mps.current_allocated_memory(), mps_driver_bytes=torch.mps.driver_allocated_memory())
    result['passed'] = bool(forward_error <= plan['health_gate']['forward_max_error']
        and gradient_error <= plan['health_gate']['gradient_max_error'])
    optimizer.zero_grad(set_to_none=True)
    del cpu_model, cpu_inputs, cpu_target, cpu_output, mps_output
    return result


def main():
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise ValueError('No silent backend fallback')
    if PLAN.exists() or REPORT.exists():
        raise FileExistsError('Preserve the Mamba health experiment')
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.5)
    parent = json.loads(PARENT.read_text())
    model_config = {**parent['models']['capacity64'], 'ffn': 768, 'encoder_layers': 1,
        'ssm_state': 128, 'ssm_headdim': 64}
    plan = dict(id='local-portfolio-mamba-health-v1', seed=20261032, health_steps=100,
        quality_schedule_steps=2000, model=model_config, density_reference=parent['density_reference'],
        health_indices=parent['health_indices'], empty_health_index=parent['empty_health_index'],
        learning_rate=.001, minimum_learning_rate=.0001, warmup=50, weight_decay=0., clip=1.,
        health_wall_s=1800, cloud_compute_calls=0, confirmation_policy='No development or confirmation scoring.',
        mechanism='Attention-free non-autoregressive canvas; all context, query-memory fusion and temporal mixing use shared-weight bidirectional two-pass Mamba-2/SSD.',
        health_gate=dict(loss_max_ratio=.9, count_max_ratio=1.05, offset_max_ratio=1.05,
            forward_max_error=1e-4, gradient_max_error=3e-3, all_ancestral_outputs_complete=True,
            note='No F1, nonempty or type quota gate. Generation is one-shot canvas decoding, not AR ancestral sampling.'),
        source_sha256={name: sha(ROOT / name) for name in (
            'scripts/run_local_portfolio_mamba_health.py', 'beatfcn/model/local_mamba_canvas.py',
            'beatfcn/model/mamba2_reference.py', 'beatfcn/model/fullstack_mamba.py',
            'beatfcn/model/local_typed_canvas.py', 'beatfcn/train/relative_canvas.py',
            'tests/test_local_mamba_canvas.py', str(PARENT.relative_to(ROOT)))})
    write(PLAN, plan)
    comparison = json.loads((ROOT / 'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text())
    panel, store = setup(comparison)
    row = row_factory(panel, store)
    torch.manual_seed(plan['seed'])
    model = MambaCanvas(FullEventConfig(**plan['model']), density_reference=plan['density_reference']).to('mps')
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=plan['weight_decay'])
    LIVE.update(model=model, optimizer=optimizer, stage='health')
    specs = list(zip(plan['health_indices'], ('draft', 'single_condition', 'random_groups', 'middle')))
    specs.append((plan['empty_health_index'], 'draft'))
    batches = []
    for index, view in specs:
        example = row(index, view)
        batches.append((example, *relative_batch([example], 'mps')))
    report = dict(id=plan['id'], status='prepared', plan_sha256=sha(PLAN),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        initial_weight_sha256=weight_digest(model), health_updates=0, quality_updates=0,
        curves=[], failures=[], wall_s={}, cloud_compute_calls=0,
        development_scored=False, confirmation_scored=False,
        input_audit=[dict(index=example['index'], target_events=len(example['targets']),
            shapes={key: list(inputs[key].shape) for key in ('local', 'mel', 'song', 'reference', 'controls')},
            finite=all(bool(torch.isfinite(inputs[key].float()).all()) for key in ('local', 'mel', 'song', 'reference', 'controls')),
            canvas_valid=int(inputs['canvas_valid'].sum()), target_slots=int(target['slot_valid'].sum()))
            for example, inputs, target in batches])
    report['diagnostic_initial'] = diagnostic(model, batches)
    report['zero_update_parity'] = parity(model, optimizer, batches[0], plan)
    if not report['zero_update_parity']['passed']:
        report['status'] = 'stopped'
        write(REPORT, report)
        raise ValueError(f'Mamba CPU/MPS parity failed: {report["zero_update_parity"]}')
    write(REPORT, report)
    journal(f'冻结：完整双向Mamba-2/SSD画布，参数{report["parameters"]}，无Attention/END；真实5窗100步，固定clean loss下降>=10%、count/offset不恶化>5%、CPU/MPS零更新与合法解码门；无F1/非空配额门，质量未派发，云0。')

    def stop(*_):
        raise KeyboardInterrupt('External stop')

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    try:
        report['status'] = 'health_running'
        for step in range(1, plan['health_steps'] + 1):
            if time.monotonic() - started > plan['health_wall_s']:
                raise TimeoutError('Frozen Mamba health wall')
            if step <= plan['warmup']:
                lr = plan['learning_rate'] * step / plan['warmup']
            else:
                progress = (step - plan['warmup']) / (plan['quality_schedule_steps'] - plan['warmup'])
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
            report['health_updates'] = step
            if step == 1 or step % 5 == 0:
                point = dict(step=step, learning_rate=lr, loss=total, preclip_gradient_norm=gradient,
                    wall_s=time.monotonic() - started)
                report['curves'].append(point)
                report['wall_s']['health'] = point['wall_s']
                write(REPORT, report)
                print(json.dumps(dict(stage='mamba-health', **point)), flush=True)
        report['diagnostic_final'] = diagnostic(model, batches)
        predictions = []
        model.eval()
        for example, inputs, target in batches:
            prediction = generate_relative(model, inputs, example['request'], deadline=started + plan['health_wall_s'])
            events = [RhythmEvent(**event) for event in prediction['events']]
            predictions.append(dict(index=example['index'], prediction=prediction,
                target=[asdict(event) for event in example['targets']], scores=score(events, example['targets'])))
        write(ART / 'health-predictions.json', predictions)
        initial, final = report['diagnostic_initial']['mean'], report['diagnostic_final']['mean']
        gates = dict(zero_update_parity=report['zero_update_parity']['passed'],
            loss=final['loss'] <= plan['health_gate']['loss_max_ratio'] * initial['loss'],
            count=final['count'] <= plan['health_gate']['count_max_ratio'] * initial['count'],
            offset=final['offset'] <= plan['health_gate']['offset_max_ratio'] * initial['offset'],
            complete=all(item['prediction']['complete'] for item in predictions),
            finite=all(torch.isfinite(parameter).all() for parameter in model.parameters()))
        report['health_gates'] = {key: bool(value) for key, value in gates.items()}
        report['health_passed'] = all(gates.values())
        report['status'] = 'health_passed' if report['health_passed'] else 'health_failed'
        report['wall_s']['health'] = time.monotonic() - started
        report['final_weight_sha256'] = weight_digest(model)
        path = ART / 'health-0100.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=plan['model'],
            plan_sha256=sha(PLAN), updates=report['health_updates']), path)
        write(REPORT, report)
        journal(f'终态：health={report["status"]}，gates={report["health_gates"]}，clean loss {initial["loss"]:.6f}→{final["loss"]:.6f}；未派发质量训练，开发/确认未评分，云0。')
    except BaseException as exc:
        report.setdefault('failures', []).append(dict(stage=LIVE.get('stage'),
            error=f'{type(exc).__name__}: {exc}', health_updates=report['health_updates']))
        report['status'] = 'stopped'
        report['wall_s']['health'] = time.monotonic() - started
        write(REPORT, report)
        journal(f'停止：{report["failures"][-1]}；保留现场，不自动延长或续训，云0。')
        raise


if __name__ == '__main__':
    main()
