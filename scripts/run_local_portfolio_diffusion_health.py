"""Real fixed-batch health probe for slot-package absorbing canvas diffusion."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time

import torch

from beatfcn.model.canvas_diffusion import AbsorbingCanvasDiffusion, ancestral_sample, diffusion_loss
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.train.relative_canvas import relative_batch
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_scale import row_factory, weight_digest
from scripts.run_local_typed_canvas import score


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'bench/rhythm-v2/local-portfolio-scale-v1.json'
PLAN = ROOT / 'bench/rhythm-v2/local-portfolio-diffusion-health-v1.json'
REPORT = ROOT / 'reports/local_portfolio_diffusion_health_v1.json'
ART = ROOT / 'artifacts/local-portfolio-v1/diffusion-health'
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
        handle.write('\n\nlocal-portfolio-diffusion-health-v1 ' + message + '\n')


def prepare():
    if PLAN.exists() or REPORT.exists():
        raise FileExistsError('Preserve the diffusion health experiment')
    parent = json.loads(PARENT.read_text())
    plan = dict(id='local-portfolio-diffusion-health-v1', seed=20261031, steps=32, health_steps=100,
        model=parent['models']['capacity64'], density_reference=parent['density_reference'],
        health_indices=parent['health_indices'], empty_health_index=parent['empty_health_index'],
        learning_rate=.0005, weight_decay=0., clip=1., auxiliary_weight=.001,
        health_wall_s=1200, cloud_compute_calls=0, confirmation_policy='No development or confirmation scoring.',
        mechanism='Slot-package absorbing D3PM. Onset and complete hold endpoint share one MASK; carry-over; exact categorical reverse; legal jointly normalized tail.',
        health_gate=dict(fixed_mask_vb_max_ratio=.9, fixed_mask_auxiliary_max_ratio=1.05,
            zero_update_forward_max_error=1e-4, zero_update_gradient_max_error=3e-3,
            finite_loss_and_gradient=True, all_ancestral_outputs_complete=True,
            note='No F1, nonempty, count or type quota gate. Losses are not compared with canvas_loss.'),
        source_sha256={name: sha(ROOT / name) for name in (
            'scripts/run_local_portfolio_diffusion_health.py', 'beatfcn/model/canvas_diffusion.py',
            'beatfcn/model/local_typed_canvas.py', 'beatfcn/train/relative_canvas.py',
            'beatfcn/train/local_typed_canvas.py', 'tests/test_canvas_diffusion.py', str(PARENT.relative_to(ROOT)))})
    write(PLAN, plan)
    comparison = json.loads((ROOT / 'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text())
    panel, store = setup(comparison)
    report = dict(id=plan['id'], status='prepared', plan_sha256=sha(PLAN), health_updates=0,
        curves=[], failures=[], wall_s={}, cloud_compute_calls=0, quality_updates=0,
        confirmation_scored=False, development_scored=False)
    write(REPORT, report)
    return plan, report, panel, store


def diagnostic(model, batches, plan):
    rows = []
    with torch.no_grad():
        for example, inputs, target, noisy, t in batches:
            output = model(inputs, noisy, t)
            loss, detail = diffusion_loss(output, target, noisy, inputs, [example['request']], t,
                auxiliary_weight=plan['auxiliary_weight'], balance_positive=plan.get('balance_positive', False))
            rows.append(dict(index=example['index'], t=int(t[0]), loss=float(loss),
                **{key: float(value) for key, value in detail.items()}))
    keys = ('loss', 'vb', 'auxiliary', 'onset_vb', 'tail_vb')
    mean = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    return dict(rows=rows, mean=mean)


def input_audit(batches):
    result = []
    for example, inputs, target, _, _ in batches:
        tensors = {}
        for key in ('local', 'mel', 'song', 'reference', 'controls'):
            value = inputs[key].float()
            tensors[key] = dict(shape=list(value.shape), finite=bool(torch.isfinite(value).all()),
                min=float(value.min()), max=float(value.max()), mean=float(value.mean()), std=float(value.std()))
        result.append(dict(index=example['index'], target_events=len(example['targets']), tensors=tensors,
            canvas_valid=int(inputs['canvas_valid'].sum()), target_slots=int(target['slot_valid'].sum())))
    return result


def zero_update_parity(model, optimizer, batch, plan):
    example, inputs, target, noisy, t = batch
    config = FullEventConfig(**plan['model'])
    cpu_model = AbsorbingCanvasDiffusion(config, density_reference=plan['density_reference'],
        steps=plan['steps'])
    cpu_model.load_state_dict({key: value.detach().cpu() for key, value in model.state_dict().items()})
    cpu_inputs = {key: value.cpu() for key, value in inputs.items()}
    cpu_target = {key: value.cpu() for key, value in target.items()}
    cpu_noisy = {key: value.cpu() for key, value in noisy.items()}
    cpu_t = t.cpu()
    started = time.monotonic()
    cpu_output = cpu_model(cpu_inputs, cpu_noisy, cpu_t)
    cpu_loss, _ = diffusion_loss(cpu_output, cpu_target, cpu_noisy, cpu_inputs, [example['request']], cpu_t,
        auxiliary_weight=plan['auxiliary_weight'], balance_positive=plan.get('balance_positive', False))
    cpu_loss.backward()
    cpu_wall = time.monotonic() - started
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    mps_output = model(inputs, noisy, t)
    mps_loss, _ = diffusion_loss(mps_output, target, noisy, inputs, [example['request']], t,
        auxiliary_weight=plan['auxiliary_weight'], balance_positive=plan.get('balance_positive', False))
    mps_loss.backward()
    torch.mps.synchronize()
    mps_wall = time.monotonic() - started
    forward_error = max(float((cpu_output[key] - mps_output[key].cpu()).abs().max()) for key in cpu_output)
    gradient_error = 0.
    cpu_parameters = dict(cpu_model.named_parameters())
    for name, parameter in model.named_parameters():
        cpu_gradient = cpu_parameters[name].grad
        if cpu_gradient is None or parameter.grad is None:
            if cpu_gradient is not parameter.grad:
                raise ValueError(f'CPU/MPS gradient presence differs: {name}')
            continue
        gradient_error = max(gradient_error, float((cpu_gradient - parameter.grad.cpu()).abs().max()))
    result = dict(updates=0, forward_max_error=forward_error, gradient_max_error=gradient_error,
        loss_absolute_error=abs(float(cpu_loss) - float(mps_loss)), cpu_forward_backward_s=cpu_wall,
        mps_forward_backward_s=mps_wall, process_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        mps_allocated_bytes=torch.mps.current_allocated_memory(), mps_driver_bytes=torch.mps.driver_allocated_memory())
    result['passed'] = bool(forward_error <= plan['health_gate']['zero_update_forward_max_error']
        and gradient_error <= plan['health_gate']['zero_update_gradient_max_error'])
    optimizer.zero_grad(set_to_none=True)
    del cpu_model, cpu_inputs, cpu_target, cpu_noisy, cpu_output, mps_output
    return result


def main():
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise ValueError('No silent backend fallback')
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.5)
    plan, report, panel, store = prepare()
    row = row_factory(panel, store)
    torch.manual_seed(plan['seed'])
    model = AbsorbingCanvasDiffusion(FullEventConfig(**plan['model']),
        density_reference=plan['density_reference'], steps=plan['steps']).to('mps')
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=plan['weight_decay'])
    LIVE.update(model=model, optimizer=optimizer, stage='health')
    report['parameters'] = sum(parameter.numel() for parameter in model.parameters())
    report['initial_weight_sha256'] = weight_digest(model)
    specs = list(zip(plan['health_indices'], ('draft', 'single_condition', 'random_groups', 'middle')))
    specs.append((plan['empty_health_index'], 'draft'))
    batches = []
    for i, (index, view) in enumerate(specs):
        example = row(index, view)
        inputs, target = relative_batch([example], 'mps')
        t = torch.tensor([1 + (i * 7) % plan['steps']], device='mps')
        cpu_target = {key: value.cpu() for key, value in target.items()}
        noisy = model.corrupt(cpu_target, int(t[0]), generator=torch.Generator().manual_seed(plan['seed'] + i))
        noisy = {key: value.to('mps') for key, value in noisy.items()}
        batches.append((example, inputs, target, noisy, t))
    report['input_audit'] = input_audit(batches)
    report['diagnostic_initial'] = diagnostic(model, batches, plan)
    report['zero_update_parity'] = zero_update_parity(model, optimizer, batches[0], plan)
    if not report['zero_update_parity']['passed']:
        report['status'] = 'stopped'
        write(REPORT, report)
        raise ValueError(f'Diffusion CPU/MPS parity failed: {report["zero_update_parity"]}')
    report['status'] = 'health_running'
    write(REPORT, report)
    journal(f'冻结：slot-package absorbing D3PM，T={plan["steps"]}，参数{report["parameters"]}；固定5窗100步，固定掩码VB下降>=10%，辅助重建不恶化>5%，祖先输出合法完整；无F1/非空配额门，质量阶段未派发，云0。')

    def stop(*_):
        raise KeyboardInterrupt('External stop')

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    try:
        for step in range(1, plan['health_steps'] + 1):
            if time.monotonic() - started > plan['health_wall_s']:
                raise TimeoutError('Frozen diffusion health wall')
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for i, (example, inputs, target, _, _) in enumerate(batches):
                t = torch.tensor([1 + (step * 7 + i * 5) % plan['steps']], device='mps')
                noisy = model.corrupt(target, int(t[0]))
                output = model(inputs, noisy, t)
                loss, _ = diffusion_loss(output, target, noisy, inputs, [example['request']], t,
                    auxiliary_weight=plan['auxiliary_weight'])
                (loss / len(batches)).backward()
                total += float(loss.detach()) / len(batches)
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), plan['clip'], error_if_nonfinite=True))
            optimizer.step()
            report['health_updates'] = step
            if step == 1 or step % 10 == 0:
                point = dict(step=step, loss=total, preclip_gradient_norm=gradient,
                    wall_s=time.monotonic() - started)
                report['curves'].append(point)
                report['wall_s']['health'] = point['wall_s']
                write(REPORT, report)
                print(json.dumps(dict(stage='diffusion-health', **point)), flush=True)
        report['diagnostic_final'] = diagnostic(model, batches, plan)
        predictions = []
        model.eval()
        for i, (example, inputs, target, _, _) in enumerate(batches):
            prediction = ancestral_sample(model, inputs, example['request'], seed=plan['seed'] + 100 + i,
                deadline=started + plan['health_wall_s'])
            events = [RhythmEvent(**event) for event in prediction['events']]
            predictions.append(dict(index=example['index'], prediction=prediction,
                target=[asdict(event) for event in example['targets']], scores=score(events, example['targets'])))
        write(ART / 'health-predictions.json', predictions)
        initial, final = report['diagnostic_initial']['mean'], report['diagnostic_final']['mean']
        gates = dict(zero_update_parity=report['zero_update_parity']['passed'],
            fixed_mask_vb=final['vb'] <= plan['health_gate']['fixed_mask_vb_max_ratio'] * initial['vb'],
            fixed_mask_auxiliary=final['auxiliary'] <= plan['health_gate']['fixed_mask_auxiliary_max_ratio'] * initial['auxiliary'],
            ancestral_complete=all(item['prediction']['complete'] for item in predictions),
            finite=all(torch.isfinite(parameter).all() for parameter in model.parameters()))
        report['health_gates'] = {key: bool(value) for key, value in gates.items()}
        report['health_passed'] = all(gates.values())
        report['status'] = 'health_passed' if report['health_passed'] else 'health_failed'
        report['wall_s']['health'] = time.monotonic() - started
        report['final_weight_sha256'] = weight_digest(model)
        path = ART / 'health-0100.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=plan['model'],
            steps=plan['steps'], plan_sha256=sha(PLAN), updates=report['health_updates']), path)
        write(REPORT, report)
        journal(f'终态：health={report["status"]}，gates={report["health_gates"]}，fixed VB {initial["vb"]:.6f}→{final["vb"]:.6f}；未派发质量训练，开发/确认未评分，云0。')
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
