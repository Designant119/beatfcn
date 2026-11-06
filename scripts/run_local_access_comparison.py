"""Serial, exact-parameter comparison of additional canvas access patterns."""
import argparse
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import signal
import time

import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_access_canvas import ACCESS_SCHEDULES, AccessCanvas
from beatfcn.model.local_typed_canvas import canvas_loss
from beatfcn.train.relative_canvas import generate_relative
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_candidate_quality import batch, evaluate
from scripts.run_local_portfolio_mamba_health import diagnostic, sha, write
from scripts.run_local_portfolio_scale import row_factory, weight_digest
from scripts.run_local_typed_canvas import score


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'bench/rhythm-v2/local-access-comparison-v1.json'
HEALTH = ROOT / 'reports/local_access_health_v1.json'
QUALITY = ROOT / 'reports/local_access_quality_v1.json'
ART = ROOT / 'artifacts/local-access-comparison-v1'
ARMS = ('global', 'mixed')


def model_for(plan, arm, device):
    torch.manual_seed(plan['seed'])
    return AccessCanvas(FullEventConfig(**plan['model']),
                        density_reference=plan['density_reference'], access=arm,
                        radius=plan['window_radius']).to(device)


def journal(message):
    with (ROOT / 'EXPERIMENT_LOG.md').open('a') as handle:
        handle.write('\n\nlocal-access-comparison-v1 ' + message + '\n')


def checked_plan():
    plan = json.loads(PLAN.read_text())
    for name, expected in plan['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen source drift: ' + name)
    return plan


def prepare():
    if PLAN.exists():
        raise FileExistsError('Preserve the frozen access plan')
    parent_path = ROOT / 'bench/rhythm-v2/local-portfolio-scale-v1.json'
    parent = json.loads(parent_path.read_text())
    plan = dict(id='local-access-comparison-v1', candidate='mel', seed=20261039,
        model=parent['models']['capacity64'], density_reference=parent['density_reference'],
        schedules=ACCESS_SCHEDULES, window_radius=16, cell_us=40_000,
        indices=parent['indices'], sample_stream=parent['sample_stream']['train64'][:1000],
        health_indices=parent['health_indices'], empty_health_index=parent['empty_health_index'],
        health_steps=100, quality_steps=1000, effective_batch=2,
        learning_rate=.0005, minimum_learning_rate=.00005, warmup=50, weight_decay=0., clip=1.,
        health_wall_per_arm_s=1500, quality_wall_per_arm_s=3600, evaluation_wall_per_arm_s=600,
        shifts_us=parent['shifts_us'], health_weights_used=False, confirmation_scored=False,
        cloud_compute_calls=0, main_dimension='Architecture: additional temporal access allocation',
        hypothesis='At identical parameters, depth, initial tensors, audio, supervision and schedule, restricting two of three temporal attention layers to +/-16 cells improves independent-song event generation.',
        core_screen=dict(full_f1_gain=.015, onset10_max_regression=.01,
            minimum_component_full_f1_wins=10,
            note='Exploratory local screen, not product readiness. Report type/tail/empty/reference separately; no post-hoc endpoint selection.'),
        boundary='Upstream fine/coarse encoders and address layers already have global access. This compares additional canvas mixing; it neither adds whole-song input nor establishes emotion understanding.',
        health_gate=dict(loss_max_ratio=.9, count_max_ratio=1.05, offset_max_ratio=1.05,
            forward_max_error=1e-4, gradient_max_error=.003,
            note='Finite fit, CPU/MPS numerical parity and complete actual decode; no production F1 or nonempty quota.'),
        stopping='Serial arms, no automatic retry, no threshold search, fresh quality after both healthy; fixed1000 endpoint. Wall limits include evaluation/save; timeout is incomplete.',
        source_sha256={name: sha(ROOT / name) for name in (
            'scripts/run_local_access_comparison.py', 'beatfcn/model/local_access_canvas.py',
            'tests/test_local_access_canvas.py', 'beatfcn/model/mel_only_canvas.py',
            'beatfcn/model/local_typed_canvas.py', 'beatfcn/model/full_event_model.py',
            'beatfcn/train/relative_canvas.py', 'beatfcn/train/local_typed_canvas.py',
            'scripts/run_local_portfolio_candidate_quality.py',
            'scripts/run_local_portfolio_mamba_health.py',
            'scripts/run_local_portfolio_scale.py', 'scripts/run_frontend_canvas_comparison.py',
            'scripts/review_frontend_canvas_comparison.py',
            'scripts/run_local_typed_canvas.py', str(parent_path.relative_to(ROOT)))})
    torch.set_num_threads(2)
    models = {arm: model_for(plan, arm, 'cpu') for arm in ARMS}
    plan['parameters'] = {arm: sum(p.numel() for p in model.parameters()) for arm, model in models.items()}
    plan['initial_weight_sha256'] = {arm: weight_digest(model) for arm, model in models.items()}
    if len(set(plan['parameters'].values())) != 1 or len(set(plan['initial_weight_sha256'].values())) != 1:
        raise ValueError('Access arms are not exactly matched')
    if max(plan['parameters'].values()) > 20_000_000:
        raise ValueError('Local model parameter cap exceeded')
    write(PLAN, plan)
    journal(f'预注册：同初始化/同参数{plan["parameters"]}；global {ACCESS_SCHEDULES["global"]} vs mixed {ACCESS_SCHEDULES["mixed"]}，只改变两层访问mask，mel-only输入和全部目标/解码固定。100健康后fresh1000质量，单臂健康1500s、质量3600s、云0。主门完整F1+.015、onset10退化<=.01、至少10/16组件完整F1改善；不预判产品质量。混合旧实验不伪造单维计数，既有失败不清零。')
    print(json.dumps(dict(status='prepared', parameters=plan['parameters'],
                          initial_weight_sha256=plan['initial_weight_sha256']), indent=2))


def data(plan):
    comparison = json.loads((ROOT / 'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text())
    panel, store = setup(comparison)
    return panel, row_factory(panel, store)


def parity(model, inputs, target, plan, arm):
    cpu = model_for(plan, arm, 'cpu')
    cpu.load_state_dict({key: value.cpu() for key, value in model.state_dict().items()})
    ci, ct = ({k: v.cpu() for k, v in values.items()} for values in (inputs, target))
    started = time.monotonic()
    co = cpu(ci)
    cl, _ = canvas_loss(co, ct, ci)
    cl.backward()
    cpu_wall = time.monotonic() - started
    started = time.monotonic()
    mo = model(inputs)
    ml, _ = canvas_loss(mo, target, inputs)
    ml.backward()
    torch.mps.synchronize()
    mps_wall = time.monotonic() - started
    cp = dict(cpu.named_parameters())
    forward = max(float((co[k] - mo[k].cpu()).abs().max()) for k in co)
    gradient = 0.
    for name, p in model.named_parameters():
        cg = cp[name].grad
        if (cg is None) != (p.grad is None):
            raise ValueError('Gradient presence mismatch: ' + name)
        if cg is not None:
            gradient = max(gradient, float((cg - p.grad.cpu()).abs().max()))
    result = dict(forward_max_error=forward, gradient_max_error=gradient,
        cpu_forward_backward_s=cpu_wall, mps_forward_backward_s=mps_wall,
        mps_allocated_bytes=torch.mps.current_allocated_memory(),
        mps_driver_bytes=torch.mps.driver_allocated_memory(),
        mps_recommended_bytes=torch.mps.recommended_max_memory())
    result['passed'] = bool(forward <= plan['health_gate']['forward_max_error']
                            and gradient <= plan['health_gate']['gradient_max_error'])
    model.zero_grad(set_to_none=True)
    return result


def learning_rate(plan, step):
    if step <= plan['warmup']:
        return plan['learning_rate'] * step / plan['warmup']
    progress = (step - plan['warmup']) / (plan['quality_steps'] - plan['warmup'])
    return plan['minimum_learning_rate'] + (plan['learning_rate'] - plan['minimum_learning_rate']) * (1 + math.cos(math.pi * progress)) / 2


def health(plan):
    if HEALTH.exists():
        raise FileExistsError('Preserve the health run')
    _, row = data(plan)
    report = dict(id=plan['id'], status='health_running', plan_sha256=sha(PLAN), arms={},
                  development_scored=False, confirmation_scored=False, cloud_compute_calls=0)
    write(HEALTH, report)
    for arm in ARMS:
        started = time.monotonic()
        current = dict(status='running', health_updates=0, curves=[], failures=[])
        report['arms'][arm] = current
        try:
            model = model_for(plan, arm, 'mps')
            if weight_digest(model) != plan['initial_weight_sha256'][arm]:
                raise ValueError('Initialization identity mismatch')
            optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=0.)
            specs = list(zip(plan['health_indices'], ('draft', 'single_condition', 'random_groups', 'middle')))
            specs.append((plan['empty_health_index'], 'draft'))
            batches = [(example, *batch(example, 'mel')) for index, view in specs
                       for example in [row(index, view)]]
            current['input_audit'] = [dict(index=e['index'], mel_shape=list(i['mel'].shape),
                mel_finite=bool(torch.isfinite(i['mel']).all()),
                mel_time_range=[float(i['mel_times'].min()), float(i['mel_times'].max())],
                valid_mel=int(i['mel_valid'].sum()), canvas_valid=int(i['canvas_valid'].sum()),
                target_events=len(e['targets']), pretrained_keys_passed=[k for k in i if k.startswith(('local', 'song'))])
                for e, i, t in batches]
            current['diagnostic_initial'] = diagnostic(model, batches)
            current['zero_update_parity'] = parity(model, batches[0][1], batches[0][2], plan, arm)
            write(HEALTH, report)
            if not current['zero_update_parity']['passed']:
                raise ValueError('CPU/MPS numerical parity failed')
            for step in range(1, plan['health_steps'] + 1):
                if time.monotonic() - started > plan['health_wall_per_arm_s']:
                    raise TimeoutError('Health wall exceeded')
                model.train()
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate(plan, step)
                total = 0.
                for _, inputs, target in batches:
                    loss, _ = canvas_loss(model(inputs), target, inputs)
                    (loss / len(batches)).backward()
                    total += float(loss.detach()) / len(batches)
                gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), plan['clip'], error_if_nonfinite=True))
                optimizer.step()
                current['health_updates'] = step
                if step == 1 or step % 10 == 0:
                    current['curves'].append(dict(step=step, loss=total, gradient=gradient,
                                                  wall_s=time.monotonic() - started))
                    write(HEALTH, report)
                    print(json.dumps(dict(arm=arm, stage='health', **current['curves'][-1])), flush=True)
            current['diagnostic_final'] = diagnostic(model, batches)
            model.eval()
            predictions = []
            for example, inputs, _ in batches:
                prediction = generate_relative(model, inputs, example['request'],
                                               deadline=started + plan['health_wall_per_arm_s'])
                predictions.append(dict(index=example['index'], prediction=prediction,
                    target=[asdict(e) for e in example['targets']],
                    scores=score([RhythmEvent(**e) for e in prediction['events']], example['targets'])))
            write(ART / arm / 'health-predictions.json', predictions)
            initial, final = (current[f'diagnostic_{stage}']['mean'] for stage in ('initial', 'final'))
            current['gates'] = dict(parity=current['zero_update_parity']['passed'],
                loss=final['loss'] <= initial['loss'] * plan['health_gate']['loss_max_ratio'],
                count=final['count'] <= initial['count'] * plan['health_gate']['count_max_ratio'],
                offset=final['offset'] <= initial['offset'] * plan['health_gate']['offset_max_ratio'],
                complete=all(p['prediction']['complete'] for p in predictions),
                finite=all(bool(torch.isfinite(p).all()) for p in model.parameters()))
            current['passed'] = all(current['gates'].values())
            current['status'] = 'health_passed' if current['passed'] else 'health_failed'
            current['wall_s'] = time.monotonic() - started
            path = ART / arm / 'health-0100.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(model=model.state_dict(), plan_sha256=sha(PLAN), health_updates=100), path)
            current['checkpoint_sha256'] = sha(path)
            write(HEALTH, report)
            journal(f'{arm} 健康终态{current["status"]}；loss {initial["loss"]:.6f}→{final["loss"]:.6f}，{current["wall_s"]:.1f}s。健康权重不进入质量，云0。')
        except BaseException as exc:
            current.update(status='stopped', passed=False, wall_s=time.monotonic() - started)
            current['failures'].append(f'{type(exc).__name__}: {exc}')
            report['status'] = 'stopped'
            write(HEALTH, report)
            raise
        finally:
            if 'model' in locals():
                del model
            if 'optimizer' in locals():
                del optimizer
            gc.collect()
            torch.mps.empty_cache()
    report['health_passed'] = all(r['passed'] for r in report['arms'].values())
    report['status'] = 'health_passed' if report['health_passed'] else 'health_failed'
    write(HEALTH, report)


def quality(plan):
    if QUALITY.exists():
        raise FileExistsError('Preserve the quality run')
    health_report = json.loads(HEALTH.read_text())
    if health_report['plan_sha256'] != sha(PLAN) or not health_report['health_passed']:
        raise ValueError('Require both independent healthy arms')
    panel, row = data(plan)
    report = dict(id=plan['id'], status='training', plan_sha256=sha(PLAN), arms={},
                  health_weights_used=False, confirmation_scored=False, cloud_compute_calls=0)
    write(QUALITY, report)
    for arm in ARMS:
        started = time.monotonic()
        current = dict(status='training', updates=0, curves=[], failures=[], evaluations={})
        report['arms'][arm] = current
        try:
            model = model_for(plan, arm, 'mps')
            current['initial_weight_sha256'] = weight_digest(model)
            if current['initial_weight_sha256'] != plan['initial_weight_sha256'][arm]:
                raise ValueError('Fresh quality initialization mismatch')
            optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=0.)
            for step, specs in enumerate(plan['sample_stream'], 1):
                if time.monotonic() - started > plan['quality_wall_per_arm_s']:
                    raise TimeoutError('Quality wall exceeded')
                optimizer.zero_grad(set_to_none=True)
                model.train()
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate(plan, step)
                total = 0.
                for spec in specs:
                    example = row(spec['index'], spec['view'], spec['field_index'])
                    inputs, target = batch(example, 'mel')
                    loss, _ = canvas_loss(model(inputs), target, inputs)
                    (loss / plan['effective_batch']).backward()
                    total += float(loss.detach()) / plan['effective_batch']
                gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), plan['clip'], error_if_nonfinite=True))
                optimizer.step()
                current['updates'] = step
                if step == 1 or step % 25 == 0:
                    current['curves'].append(dict(step=step, loss=total, gradient=gradient,
                                                  wall_s=time.monotonic() - started))
                    write(QUALITY, report)
                    print(json.dumps(dict(arm=arm, stage='quality', **current['curves'][-1])), flush=True)
                if step in (500, 1000):
                    path = ART / arm / f'quality-{step:04d}.pt'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        plan_sha256=sha(PLAN), updates=step,
                        cpu_rng=torch.get_rng_state(), mps_rng=torch.mps.get_rng_state()), path)
            moment = time.monotonic()
            deadline = min(started + plan['quality_wall_per_arm_s'], moment + plan['evaluation_wall_per_arm_s'])
            current['evaluations']['1000'] = evaluate(model, plan, row, panel, ART / arm,
                                                      'quality-1000', deadline, full=True)
            current.update(status='completed', wall_s=time.monotonic() - started,
                evaluation_wall_s=time.monotonic() - moment,
                final_checkpoint_sha256=sha(ART / arm / 'quality-1000.pt'))
            write(QUALITY, report)
            journal(f'{arm} fresh1000质量完成，{current["wall_s"]:.1f}s；确认未评分、云0。')
        except BaseException as exc:
            current.update(status='stopped', wall_s=time.monotonic() - started)
            current['failures'].append(f'{type(exc).__name__}: {exc}')
            report['status'] = 'stopped'
            write(QUALITY, report)
            raise
        finally:
            if 'model' in locals():
                del model
            if 'optimizer' in locals():
                del optimizer
            gc.collect()
            torch.mps.empty_cache()
    report['status'] = 'completed'
    write(QUALITY, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'health', 'quality'))
    stage = parser.parse_args().stage
    if stage == 'prepare':
        prepare()
        return
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise ValueError('No silent CPU fallback')
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.5)
    def stop(*_):
        raise KeyboardInterrupt('External stop')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    plan = checked_plan()
    (health if stage == 'health' else quality)(plan)


if __name__ == '__main__':
    main()
