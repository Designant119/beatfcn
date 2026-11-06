"""Execute the complete-count-package diffusion health contract once."""
from dataclasses import asdict
import argparse
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_group_diffusion import (
    GroupDiffusionCanvas, GroupSampleError, clean_packages, corrupt_packages,
    group_diffusion_loss, sample_group_diffusion, valid_groups,
)
from beatfcn.train.canonical_packages import canonical_batch, explicit_reference_row
from beatfcn.train.local_request_panel import event_request
from beatfcn.train.local_typed_canvas import encode_canvas, request_layout
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_mamba_health import sha, write
from scripts.run_local_portfolio_scale import row_factory, weight_digest
from scripts.run_local_typed_canvas import score


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'bench/rhythm-v2/local-group-diffusion-health-v1.json'
REPORT = ROOT / 'reports/local_group_diffusion_health_v1.json'
ART = ROOT / 'artifacts/local-group-diffusion-v1'


def prior_for(panel, indices):
    histogram = np.zeros((4, 17), dtype=np.int64)
    for index in indices:
        request = event_request(panel.records[index]['request'])
        count = encode_canvas(panel.targets(index), request)['count']
        valid = request_layout(request)['canvas_valid']
        for kind in range(4):
            histogram[kind] += np.bincount(count[valid, kind], minlength=17)
    probability = (histogram + .5) / (histogram.sum(-1, keepdims=True) + 17 * .5)
    return histogram, probability


def fresh_model(plan, device='mps'):
    torch.manual_seed(plan['seed'])
    model = GroupDiffusionCanvas(FullEventConfig(**plan['model']),
        density_reference=plan['density_reference'], diffusion_steps=32)
    prior = torch.tensor(plan['count_prior'], dtype=torch.float32)
    with torch.no_grad():
        model.count_gate.weight.zero_()
        model.count_positive.weight.zero_()
        model.count_gate.bias.copy_((1 - prior[:, 0]).log() - prior[:, 0].log())
        model.count_positive.bias.copy_((prior[:, 1:] / (1 - prior[:, :1])).log().flatten())
    return model.to(device)


def optimizer_for(model, plan):
    matrix, other = [], []
    for value in model.parameters():
        (matrix if value.ndim >= 2 else other).append(value)
    return torch.optim.AdamW([dict(params=matrix, weight_decay=plan['weight_decay']),
                              dict(params=other, weight_decay=0.)], lr=plan['learning_rate'])


def rate(plan, step):
    if step <= plan['warmup']:
        return plan['learning_rate'] * step / plan['warmup']
    ratio = (step - plan['warmup']) / (plan['quality_steps'] - plan['warmup'])
    return plan['minimum_learning_rate'] + (plan['learning_rate'] - plan['minimum_learning_rate']) * (1 + math.cos(math.pi * ratio)) / 2


def noisy_for(inputs, target, request, step, model):
    steps = torch.tensor([step], device=inputs['mode'].device)
    clean = clean_packages(target, request)
    noisy = corrupt_packages(clean, valid_groups(inputs, request), steps, model.diffusion_steps)
    return steps, noisy


def diagnostic(model, batches):
    model.eval()
    records = []
    with torch.no_grad():
        for example, inputs, target in batches:
            steps, noisy = noisy_for(inputs, target, example['request'], 32, model)
            loss, parts = group_diffusion_loss(model, inputs, target, example['request'], steps, noisy=noisy)
            output = model(inputs, noisy, steps)
            groups = valid_groups(inputs, example['request'])
            expected = (output['count'] * torch.arange(17, device=steps.device)).sum(-1)
            records.append(dict(index=example['index'], loss=float(loss),
                expected_events=float((expected * groups).sum()),
                parts={k: float(v) for k, v in parts.items()}))
    return records


def main():
    global PLAN, REPORT, ART
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repair',action='store_true')
    parser.add_argument('--restart-interrupted',action='store_true')
    args=parser.parse_args()
    repair=args.repair
    previous=None
    if args.restart_interrupted:
        if repair:raise ValueError('Choose one recovery identity')
        previous=json.loads((ROOT/'reports/local_group_diffusion_health_v2.json').read_text())
        if previous['status']!='interrupted_external' or previous['health_updates']!=70:
            raise ValueError('Only the documented externally interrupted 70-update attempt may restart')
        PLAN=ROOT/'bench/rhythm-v2/local-group-diffusion-health-v3.json'
        REPORT=ROOT/'reports/local_group_diffusion_health_v3.json'
        ART=ROOT/'artifacts/local-group-diffusion-v3'
    elif repair:
        previous=json.loads(REPORT.read_text())
        if previous['status']!='stopped' or previous['health_updates']!=0 or 'Target package outside the request support' not in str(previous['failures']):
            raise ValueError('Repair is restricted to the documented zero-update MPS support bug')
        PLAN=ROOT/'bench/rhythm-v2/local-group-diffusion-health-v2.json'
        REPORT=ROOT/'reports/local_group_diffusion_health_v2.json'
        ART=ROOT/'artifacts/local-group-diffusion-v2'
    if PLAN.exists() or REPORT.exists():
        raise FileExistsError('Preserve the health attempt; no automatic rerun')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise ValueError('No silent backend fallback')
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.5)
    parent_path = ROOT / 'bench/rhythm-v2/local-portfolio-scale-v1.json'
    parent = json.loads(parent_path.read_text())
    panel, store = setup(json.loads((ROOT / 'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()))
    histogram, prior = prior_for(panel, parent['indices']['train64'])
    plan = dict(id='local-group-diffusion-health-v1', seed=20261051,
        model=parent['models']['capacity64'], density_reference=parent['density_reference'],
        health_indices=parent['health_indices'], empty_health_index=parent['empty_health_index'],
        count_histogram=histogram.tolist(), count_prior=prior.tolist(),
        initialization='Fresh seed; native existing initialization except zero count-head matrices and natural train64-smoothed count-prior biases. Not a per-request quota.',
        health_steps=100, health_effective_batch=5, quality_steps=1000,
        learning_rate=.0003, minimum_learning_rate=.00003, warmup=50, weight_decay=.01, clip=1.,
        health_wall_s=1200, sampler_seed=20261051, steps=32, max_parameters=12_000_000,
        gates=dict(loss_ratio=.9, forward_error=1e-4, gradient_error=1e-4,
                   empty_expected=5., empty_generated=2, positive_count_ratio=[.5,1.75],
                   all_ancestral_complete=True, warmed_update_median_s=4.),
        main_hypothesis='Whole count+canonical members, natural joint masked likelihood and neighbor-package conditioning can fit/control the five real requests without slot balancing or quotas.',
        decision='Run this complete health contract to a terminal result now. Passed -> separate fresh1000 quality contract. Failed -> close this actual recipe; no automatic balance/step extension. No additional council.',
        source_sha256={name: sha(ROOT / name) for name in (
            'scripts/run_local_group_diffusion_health.py','beatfcn/model/local_group_diffusion.py',
            'beatfcn/train/canonical_packages.py','tests/test_local_group_diffusion.py',
            'beatfcn/model/local_event_package.py','beatfcn/model/canvas_diffusion.py',
            'beatfcn/model/full_event_model.py','beatfcn/train/local_typed_canvas.py',
            'scripts/run_frontend_canvas_comparison.py','scripts/run_local_portfolio_scale.py',str(parent_path.relative_to(ROOT)))},
        cloud_compute_calls=0, confirmation_scored=False)
    if args.restart_interrupted:
        plan.update(id='local-group-diffusion-health-v3',
            health_wall_s=1200-previous['curves'][-1]['wall_s']-90,
            recovery=dict(reason='External provider/session interruption at70; process absent and no checkpoint. Fresh restart, same scientific contract and initialization; prior70 updates remain charged.',
                previous_report='reports/local_group_diffusion_health_v2.json',
                previous_report_sha256=sha(ROOT/'reports/local_group_diffusion_health_v2.json'),
                previous_updates=70,cumulative_health_update_cap=170))
    elif repair:
        plan.update(id='local-group-diffusion-health-v2',
            health_wall_s=1200-float(previous['wall_s'])-90,
            repair_of='bench/rhythm-v2/local-group-diffusion-health-v1.json',
            repair='Non-inplace masks/field construction; CPU semantics and all science settings unchanged. Reserve90s for diagnostic/regression repair work; old zero-update failure preserved.')
    write(PLAN, plan)
    for name in plan['source_sha256']:
        dest = ART / 'health-source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / name).read_bytes())
    row = row_factory(panel, store)
    model = fresh_model(plan)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters > plan['max_parameters']:
        raise ValueError('Parameter cap')
    torch.mps.manual_seed(plan['seed'])
    optimizer = optimizer_for(model, plan)
    specs = list(zip(plan['health_indices'], ('draft', 'single_condition', 'random_groups', 'middle')))
    specs.append((plan['empty_health_index'], 'draft'))
    batches = []
    for index, view in specs:
        example = explicit_reference_row(row(index, view))
        inputs, target = canonical_batch([example], 'mps')
        batches.append((example, inputs, target))
    report = dict(id=plan['id'], status='health_running', plan_sha256=sha(PLAN),
        parameters=parameters, initial_weight_sha256=weight_digest(model), health_updates=0,
        quality_updates=0, curves=[], failures=[], cloud_compute_calls=0, confirmation_scored=False)
    if previous is not None and report['initial_weight_sha256']!=previous['initial_weight_sha256']:
        raise ValueError('Correctness repair changed initial parameters')
    write(REPORT, report)
    def stop(*_):
        raise KeyboardInterrupt('External stop')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    try:
        report['support_parity']=[]
        for example,inputs,target in batches:
            cpu_inputs={k:v.cpu() for k,v in inputs.items()}
            support_equal=torch.equal(valid_groups(cpu_inputs,example['request']),valid_groups(inputs,example['request']).cpu())
            cc=clean_packages({k:v.cpu() for k,v in target.items()},example['request'])
            gc=clean_packages(target,example['request'])
            fields_equal=all(torch.equal(cc[k],gc[k].cpu()) for k in cc)
            report['support_parity'].append(dict(index=example['index'],support_equal=support_equal,fields_equal=fields_equal))
            if not support_equal or not fields_equal:raise ValueError('CPU/MPS support/field parity')
        report['initial_diagnostic'] = diagnostic(model, batches)
        report['input_provenance'] = [dict(index=e['index'], view=e['view'],
            explicit_visible=len(e['visible']), known_carry=len(e['request'].known_carry),
            dropped_unprovided_carry=e['dropped_unprovided_carry'], target_events=len(e['targets']),
            tensor_shapes={k:list(v.shape) for k,v in inputs.items()},
            finite=all(bool(torch.isfinite(v).all()) for v in inputs.values())) for e,inputs,_ in batches]
        cpu = fresh_model(plan, 'cpu').eval()
        e, inputs, target = batches[0]
        ci, ct = ({k:v.cpu() for k,v in values.items()} for values in (inputs,target))
        cs, cn = noisy_for(ci, ct, e['request'], 32, cpu)
        ms, mn = noisy_for(inputs, target, e['request'], 32, model)
        with torch.no_grad():
            co, mo = cpu(ci, cn, cs), model(inputs, mn, ms)
        cl, _ = group_diffusion_loss(cpu, ci, ct, e['request'], cs, noisy=cn)
        ml, _ = group_diffusion_loss(model, inputs, target, e['request'], ms, noisy=mn)
        cl.backward();ml.backward()
        forward_error=max(float((co[k]-mo[k].cpu()).abs().max()) for k in co)
        cp=dict(cpu.named_parameters());gradient_error=0.
        for name,p in model.named_parameters():
            if (p.grad is None)!=(cp[name].grad is None):raise ValueError('Parity gradient presence')
            if p.grad is not None:gradient_error=max(gradient_error,float((p.grad.cpu()-cp[name].grad).abs().max()))
        report['parity']=dict(forward_max_error=forward_error,gradient_max_error=gradient_error)
        if max(forward_error,gradient_error)>1e-4:raise ValueError('CPU/MPS numerical gate')
        del cpu,co,mo,cl,ml,cp
        optimizer.zero_grad(set_to_none=True)
        write(REPORT,report)
        timings=[]
        for step in range(1,101):
            if time.monotonic()-started>plan['health_wall_s']:raise TimeoutError('Health wall')
            moment=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:group['lr']=rate(plan,step)
            total=0.
            for j,(example,inputs,target) in enumerate(batches):
                t=1+(step*5+j)%32;steps,noisy=noisy_for(inputs,target,example['request'],t,model)
                loss,_=group_diffusion_loss(model,inputs,target,example['request'],steps,noisy=noisy)
                (loss/5).backward();total+=float(loss.detach())/5
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
            optimizer.step();torch.mps.synchronize();timings.append(time.monotonic()-moment)
            report['health_updates']=step
            if step==1 or step%10==0:
                point=dict(step=step,loss=total,gradient=grad,wall_s=time.monotonic()-started,
                    mps_tensor_bytes=torch.mps.current_allocated_memory(),mps_driver_bytes=torch.mps.driver_allocated_memory())
                report['curves'].append(point);write(REPORT,report);print(json.dumps(point),flush=True)
            if step>=40 and float(np.median(timings[20:]))>4:raise TimeoutError('Warmed update median >4s')
        report['final_diagnostic']=diagnostic(model,batches)
        model.eval();predictions=[];ratios=[]
        for example,inputs,target in batches:
            try:
                events,info=sample_group_diffusion(model,inputs,example['request'],
                    seed=plan['sampler_seed']+example['index'],deadline=started+plan['health_wall_s'])
                prediction=dict(complete=True,events=[asdict(e) for e in events],trace=info['trace'])
            except GroupSampleError as exc:
                path=ART/f'failed-reverse-{example["index"]}.pt';path.parent.mkdir(parents=True,exist_ok=True)
                torch.save(dict(state=exc.state,trace=exc.trace,step=exc.step),path)
                prediction=dict(complete=False,events=[],failure=str(exc),partial_state=str(path.relative_to(ROOT)))
                events=[]
            predictions.append(dict(index=example['index'],prediction=prediction,target=[asdict(e) for e in example['targets']],scores=score(events,example['targets'])))
            if example['targets']:ratios.append(len(events)/len(example['targets']))
        write(ART/'health-predictions.json',predictions)
        initial=sum(v['loss'] for v in report['initial_diagnostic'])/5
        final=sum(v['loss'] for v in report['final_diagnostic'])/5
        empty=next(v for v in report['final_diagnostic'] if v['index']==plan['empty_health_index'])
        empty_pred=next(v for v in predictions if v['index']==plan['empty_health_index'])
        gates=dict(joint_loss=final<=initial*.9,empty_expected=empty['expected_events']<=5,
            empty_generated=len(empty_pred['prediction']['events'])<=2,
            positive_count_ratios=all(.5<=v<=1.75 for v in ratios),
            ancestral_complete=all(v['prediction']['complete'] for v in predictions),finite=all(bool(torch.isfinite(p).all()) for p in model.parameters()))
        report.update(status='health_passed' if all(gates.values()) else 'health_failed',health_passed=all(gates.values()),gates=gates,positive_count_ratios=ratios,wall_s=time.monotonic()-started)
        ART.mkdir(parents=True,exist_ok=True)
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),plan_sha256=sha(PLAN),updates=100),ART/'health-0100.pt')
        write(REPORT,report);print(json.dumps(dict(status=report['status'],gates=gates,ratios=ratios)),flush=True)
    except BaseException as exc:
        report.update(status='stopped',wall_s=time.monotonic()-started,health_passed=False)
        report['failures'].append(f'{type(exc).__name__}: {exc}');write(REPORT,report);raise
    finally:
        with (ROOT/'EXPERIMENT_LOG.md').open('a') as handle:
            handle.write(f'\n\n{plan["id"]} terminal/current: {report["status"]}, updates={report["health_updates"]}, no quality/confirm/cloud; full report {REPORT.relative_to(ROOT)}.\n')


if __name__=='__main__':
    main()
