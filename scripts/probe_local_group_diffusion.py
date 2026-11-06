"""Zero-update real-input parity/resource probe, not a training or quality run."""
import json
import os
from pathlib import Path
import time

import torch

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_group_diffusion import (
    GroupDiffusionCanvas, clean_packages, corrupt_packages,
    group_diffusion_loss, valid_groups,
)
from beatfcn.train.local_typed_canvas import canvas_batch
from scripts.audit_local_access_inputs import statistics
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_mamba_health import sha, write
from scripts.run_local_portfolio_scale import row_factory, weight_digest


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'bench/rhythm-v2/local-group-diffusion-probe-v1.json'
REPORT = ROOT / 'reports/local_group_diffusion_probe_v1.json'


def main():
    if PLAN.exists() or REPORT.exists():
        raise FileExistsError('Preserve the zero-update probe')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise ValueError('No silent MPS fallback')
    queue = json.loads((ROOT / 'reports/local-portfolio-v1/followup-queue-v1.json').read_text())
    if queue['status'] != 'completed':
        raise ValueError('Prior MPS queue has not finished')
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.5)
    parent_path = ROOT / 'bench/rhythm-v2/local-portfolio-scale-v1.json'
    parent = json.loads(parent_path.read_text())
    plan = dict(id='local-group-diffusion-probe-v1', seed=20261051, diffusion_steps=32,
        model=parent['models']['capacity64'], density_reference=parent['density_reference'],
        indices=[parent['health_indices'][0], parent['empty_health_index']],
        optimizer_updates=0, new_generation_calls=0, parameter_cap=12_000_000,
        forward_error_cap=1e-4, gradient_error_cap=1e-4, wall_cap_s=180,
        scope='Prepare a complete-count-package candidate. Actual inputs, CPU/MPS parity and observed resource cost only; no optimizer, training, dev quality or confirmation score.',
        registration_boundary='Prototype unit tests preceded this resource plan. This is not retroactive preregistration of learning or a final adoption of C05 training gates.',
        source_sha256={name: sha(ROOT / name) for name in (
            'scripts/probe_local_group_diffusion.py', 'beatfcn/model/local_group_diffusion.py',
            'tests/test_local_group_diffusion.py', 'beatfcn/model/local_event_package.py',
            'beatfcn/model/canvas_diffusion.py', 'beatfcn/model/local_typed_canvas.py',
            'beatfcn/model/full_event_model.py', 'beatfcn/train/local_typed_canvas.py',
            'scripts/run_local_portfolio_scale.py', 'scripts/run_frontend_canvas_comparison.py',
            str(parent_path.relative_to(ROOT)))})
    write(PLAN, plan)
    report = dict(id=plan['id'], status='running', plan_sha256=sha(PLAN), records=[],
        optimizer_updates=0, new_generation_calls=0, development_scored=False,
        confirmation_scored=False, cloud_compute_calls=0, failures=[])
    write(REPORT, report)
    with (ROOT / 'EXPERIMENT_LOG.md').open('a') as handle:
        handle.write('\n\nlocal-group-diffusion-probe-v1: register zero-update actual-input CPU/MPS parity/resource probe before running it. Main dimension is whole-package probability/generation representation; prototype7 CPU semantic tests precede this plan, not claimed as learning evidence. No optimizer, generation, health fitting, dev/confirm quality or cloud call. Formal training contract waits for final C05 synthesis.\n')
    started = time.monotonic()
    try:
        panel, store = setup(json.loads((ROOT / 'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()))
        row = row_factory(panel, store)
        torch.manual_seed(plan['seed'])
        model = GroupDiffusionCanvas(FullEventConfig(**plan['model']),
            density_reference=plan['density_reference'], diffusion_steps=plan['diffusion_steps']).eval()
        report['parameters'] = sum(p.numel() for p in model.parameters())
        if report['parameters'] > plan['parameter_cap']:
            raise ValueError('Static parameter cap exceeded')
        report['initial_weight_sha256'] = weight_digest(model)
        for position, index in enumerate(plan['indices']):
            if time.monotonic() - started > plan['wall_cap_s']:
                raise TimeoutError('Zero-update probe wall')
            example = row(index, 'draft')
            inputs, target = canvas_batch([example])
            clean = clean_packages(target, example['request'])
            steps = torch.tensor([plan['diffusion_steps']])
            noisy = corrupt_packages(clean, valid_groups(inputs, example['request']), steps,
                                      plan['diffusion_steps'])
            record = dict(index=index, target_events=len(example['targets']),
                input_statistics={k: statistics(v) for k, v in inputs.items()},
                target_statistics={k: statistics(v) for k, v in target.items()},
                masked_packages=int((noisy['count'] == 17).sum()))
            if position == 0:
                moment = time.monotonic()
                with torch.no_grad():
                    cpu_output = {k: v.detach().clone() for k, v in model(inputs, noisy, steps).items()}
                record['cpu_forward_s'] = time.monotonic() - moment
                moment = time.monotonic()
                cpu_loss, _ = group_diffusion_loss(model, inputs, target, example['request'], steps, noisy=noisy)
                cpu_loss.backward()
                record['cpu_loss_forward_backward_s'] = time.monotonic() - moment
                cpu_loss_value = float(cpu_loss.detach())
                cpu_gradients = {k: p.grad.detach().clone() if p.grad is not None else None
                                 for k, p in model.named_parameters()}
                model.zero_grad(set_to_none=True)
                del cpu_loss
                model = model.to('mps')
            mi, mt, mn = ({k: v.to('mps') for k, v in values.items()} for values in (inputs, target, noisy))
            ms = steps.to('mps')
            model.zero_grad(set_to_none=True)
            torch.mps.synchronize()
            moment = time.monotonic()
            with torch.no_grad():
                mps_output = model(mi, mn, ms)
            torch.mps.synchronize()
            record['mps_forward_s'] = time.monotonic() - moment
            moment = time.monotonic()
            loss, parts = group_diffusion_loss(model, mi, mt, example['request'], ms, noisy=mn)
            loss.backward()
            torch.mps.synchronize()
            record.update(mps_loss_forward_backward_s=time.monotonic() - moment,
                loss=float(loss.detach()), parts={k: float(v.detach()) for k, v in parts.items()},
                all_gradients_finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None),
                mps_allocated_bytes=torch.mps.current_allocated_memory(),
                mps_driver_bytes=torch.mps.driver_allocated_memory(),
                mps_recommended_bytes=torch.mps.recommended_max_memory(),
                memory_measurement='Snapshot after backward, not a peak or optimizer-step estimate.')
            if position == 0:
                record['forward_max_error'] = max(float((cpu_output[k] - v.cpu()).abs().max()) for k, v in mps_output.items())
                errors = []
                for name, p in model.named_parameters():
                    reference = cpu_gradients[name]
                    if (reference is None) != (p.grad is None):
                        raise ValueError('CPU/MPS gradient presence mismatch: ' + name)
                    if reference is not None:
                        errors.append(float((reference - p.grad.cpu()).abs().max()))
                record['gradient_max_error'] = max(errors)
                record['loss_absolute_error'] = abs(cpu_loss_value - record['loss'])
                record['parity_passed'] = (record['forward_max_error'] <= plan['forward_error_cap']
                                           and record['gradient_max_error'] <= plan['gradient_error_cap'])
            report['records'].append(record)
            write(REPORT, report)
            if not record['all_gradients_finite'] or not record.get('parity_passed', True):
                raise ValueError('Actual-input numerical gate failed')
            print(json.dumps({k: record[k] for k in ('index', 'target_events', 'loss', 'mps_loss_forward_backward_s')}
                             | {'parity': record.get('parity_passed')}), flush=True)
        if time.monotonic() - started > plan['wall_cap_s']:
            raise TimeoutError('Zero-update total wall exceeded')
        report.update(status='completed', wall_s=time.monotonic() - started)
        write(REPORT, report)
    except BaseException as exc:
        report.update(status='stopped', wall_s=time.monotonic() - started)
        report['failures'].append(f'{type(exc).__name__}: {exc}')
        write(REPORT, report)
        raise


if __name__ == '__main__':
    main()
