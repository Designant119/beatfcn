"""Class-balanced diffusion health after v1's empty-slot-dominated t=T failure."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import time

import torch

from beatfcn.model.canvas_diffusion import AbsorbingCanvasDiffusion, ancestral_sample, diffusion_loss
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.train.relative_canvas import relative_batch
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_diffusion_health import input_audit, sha, write, zero_update_parity
from scripts.run_local_portfolio_scale import row_factory, weight_digest
from scripts.run_local_typed_canvas import score


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'bench/rhythm-v2/local-portfolio-scale-v1.json'
PLAN = ROOT / 'bench/rhythm-v2/local-portfolio-diffusion-balanced-health-v1.json'
REPORT = ROOT / 'reports/local_portfolio_diffusion_balanced_health_v1.json'
ART = ROOT / 'artifacts/local-portfolio-v1/diffusion-balanced-health'
LIVE = {}


def journal(message):
    with (ROOT / 'EXPERIMENT_LOG.md').open('a') as handle:
        handle.write('\n\nlocal-portfolio-diffusion-balanced-health-v1 ' + message + '\n')


def diagnostic(model, batches, plan):
    rows = []
    with torch.no_grad():
        for example, inputs, target, noisy, t in batches:
            output = model(inputs, noisy, t)
            loss, detail = diffusion_loss(output, target, noisy, inputs, [example['request']], t,
                auxiliary_weight=plan['auxiliary_weight'], balance_positive=True)
            rows.append(dict(index=example['index'], t=int(t[0]), loss=float(loss),
                **{key: float(value) for key, value in detail.items()}))
    keys = ('loss', 'vb', 'auxiliary', 'raw_vb', 'raw_auxiliary', 'positive_vb', 'positive_auxiliary',
        'absent_vb', 'absent_auxiliary', 'tail_vb')
    return dict(rows=rows, mean={key: sum(row[key] for row in rows) / len(rows) for key in keys})


def expected_events(model, batches):
    rows = []
    with torch.no_grad():
        for example, inputs, target, noisy, t in batches:
            if int(t[0]) != 32:
                continue
            output = model(inputs, noisy, t)
            probability = output['offset0'].softmax(-1)
            expected = ((1 - probability[..., 0]).sum((2, 3)) * inputs['canvas_valid']).sum()
            rows.append(dict(index=example['index'], expected_events=float(expected)))
    return rows


def main():
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise ValueError('No silent backend fallback')
    if PLAN.exists() or REPORT.exists():
        raise FileExistsError('Preserve the balanced diffusion health experiment')
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.5)
    parent = json.loads(PARENT.read_text())
    plan = dict(id='local-portfolio-diffusion-balanced-health-v1', seed=20261034, steps=32, health_steps=100,
        model=parent['models']['capacity64'], density_reference=parent['density_reference'],
        health_indices=parent['health_indices'], empty_health_index=parent['empty_health_index'],
        learning_rate=.0005, weight_decay=0., clip=1., auxiliary_weight=.001, balance_positive=True,
        health_wall_s=1200, cloud_compute_calls=0, confirmation_policy='No development or confirmation scoring.',
        mechanism='Slot-package absorbing diffusion with explicitly reweighted positive/absent masked objective; raw KL is logged separately and not called the optimized objective.',
        health_gate=dict(fixed_mask_vb_max_ratio=.9, t32_positive_auxiliary_max_ratio=.5,
            empty_t32_expected_events_max=5., zero_update_forward_max_error=1e-4,
            zero_update_gradient_max_error=3e-3, all_ancestral_outputs_complete=True,
            note='Includes t=32 for every diagnostic row. No F1 or forced-nonempty gate.'),
        source_sha256={name: sha(ROOT / name) for name in (
            'scripts/run_local_portfolio_diffusion_balanced_health.py',
            'scripts/run_local_portfolio_diffusion_health.py', 'beatfcn/model/canvas_diffusion.py',
            'beatfcn/model/local_typed_canvas.py', 'beatfcn/train/relative_canvas.py',
            'tests/test_canvas_diffusion.py', str(PARENT.relative_to(ROOT)))})
    write(PLAN, plan)
    comparison = json.loads((ROOT / 'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text())
    panel, store = setup(comparison)
    row = row_factory(panel, store)
    torch.manual_seed(plan['seed'])
    model = AbsorbingCanvasDiffusion(FullEventConfig(**plan['model']),
        density_reference=plan['density_reference'], steps=plan['steps']).to('mps')
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=plan['weight_decay'])
    LIVE.update(model=model, optimizer=optimizer, stage='health')
    specs = list(zip(plan['health_indices'], ('draft', 'single_condition', 'random_groups', 'middle')))
    specs.append((plan['empty_health_index'], 'draft'))
    diagnostic_times = (32, 32, 32, 32, 32)
    batches = []
    for i, (index, view) in enumerate(specs):
        example = row(index, view)
        inputs, target = relative_batch([example], 'mps')
        t = torch.tensor([diagnostic_times[i]], device='mps')
        cpu_target = {key: value.cpu() for key, value in target.items()}
        noisy = model.corrupt(cpu_target, int(t[0]), generator=torch.Generator().manual_seed(plan['seed'] + i))
        noisy = {key: value.to('mps') for key, value in noisy.items()}
        batches.append((example, inputs, target, noisy, t))
    report = dict(id=plan['id'], status='prepared', plan_sha256=sha(PLAN),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        initial_weight_sha256=weight_digest(model), health_updates=0, quality_updates=0,
        curves=[], failures=[], wall_s={}, cloud_compute_calls=0,
        development_scored=False, confirmation_scored=False, input_audit=input_audit(batches))
    report['diagnostic_initial'] = diagnostic(model, batches, plan)
    report['expected_events_initial'] = expected_events(model, batches)
    report['zero_update_parity'] = zero_update_parity(model, optimizer, batches[0], plan)
    if not report['zero_update_parity']['passed']:
        report['status'] = 'stopped'
        write(REPORT, report)
        raise ValueError(f'Balanced diffusion CPU/MPS parity failed: {report["zero_update_parity"]}')
    write(REPORT, report)
    journal(f'冻结：class-balanced absorbing Diffusion v2，参数{report["parameters"]}；正/空槽分别归一，raw KL另记，t32纳入全部诊断；100步，t32正类辅助NLL降>=50%、空窗t32期望事件<=5、祖先合法；无F1门，质量未派发，云0。')

    def stop(*_):
        raise KeyboardInterrupt('External stop')

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    try:
        report['status'] = 'health_running'
        for step in range(1, plan['health_steps'] + 1):
            if time.monotonic() - started > plan['health_wall_s']:
                raise TimeoutError('Frozen balanced diffusion health wall')
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for i, (example, inputs, target, _, _) in enumerate(batches):
                t = torch.tensor([1 + (step * 7 + i * 5) % plan['steps']], device='mps')
                noisy = model.corrupt(target, int(t[0]))
                output = model(inputs, noisy, t)
                loss, _ = diffusion_loss(output, target, noisy, inputs, [example['request']], t,
                    auxiliary_weight=plan['auxiliary_weight'], balance_positive=True)
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
                print(json.dumps(dict(stage='balanced-diffusion-health', **point)), flush=True)
        report['diagnostic_final'] = diagnostic(model, batches, plan)
        report['expected_events_final'] = expected_events(model, batches)
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
        initial_positive = sum(row['positive_auxiliary'] for row in report['diagnostic_initial']['rows'] if row['t'] == 32) / 5
        final_positive = sum(row['positive_auxiliary'] for row in report['diagnostic_final']['rows'] if row['t'] == 32) / 5
        empty_expected = next(row['expected_events'] for row in report['expected_events_final']
            if row['index'] == plan['empty_health_index'])
        gates = dict(zero_update_parity=report['zero_update_parity']['passed'],
            fixed_mask_vb=final['vb'] <= plan['health_gate']['fixed_mask_vb_max_ratio'] * initial['vb'],
            t32_positive_auxiliary=final_positive <= plan['health_gate']['t32_positive_auxiliary_max_ratio'] * initial_positive,
            empty_t32_expected=empty_expected <= plan['health_gate']['empty_t32_expected_events_max'],
            ancestral_complete=all(item['prediction']['complete'] for item in predictions),
            finite=all(torch.isfinite(parameter).all() for parameter in model.parameters()))
        report['t32_positive_auxiliary'] = dict(initial=initial_positive, final=final_positive)
        report['empty_t32_expected_events'] = empty_expected
        report['health_gates'] = {key: bool(value) for key, value in gates.items()}
        report['health_passed'] = all(gates.values())
        report['status'] = 'health_passed' if report['health_passed'] else 'health_failed'
        report['wall_s']['health'] = time.monotonic() - started
        report['final_weight_sha256'] = weight_digest(model)
        path = ART / 'health-0100.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=plan['model'],
            steps=plan['steps'], plan_sha256=sha(PLAN), updates=report['health_updates'], balance_positive=True), path)
        write(REPORT, report)
        journal(f'终态：health={report["status"]}，gates={report["health_gates"]}，t32正类辅助{initial_positive:.6f}→{final_positive:.6f}，空窗期望{empty_expected:.3f}；未派发质量，开发/确认未评分，云0。')
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
