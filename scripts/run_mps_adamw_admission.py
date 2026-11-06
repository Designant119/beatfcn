"""Eight real accumulation/AdamW updates in an isolated runtime, not quality."""
import argparse
from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np
import torch

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.native_cell_chain import NativeCellChain
from beatfcn.train.mps_request_lifecycle import clear_request_cache, memory_sample
from beatfcn.train.native_cell_chain import cell_row, cell_batch, cell_loss, decode_cells
from scripts.check_experiment_storage import admission, GIB
from scripts.run_joint_native_set import audit_tensors
from scripts.run_local_evolution import load_data
from scripts.run_local_portfolio_scale import sha, write, weight_digest
from scripts.run_native_cell_chain import closure


ROOT = Path(__file__).resolve().parents[1]
ID = 'c12-adamw-runtime-admission-v1'
ART = ROOT / f'artifacts/{ID}'
PLAN = ROOT / f'bench/rhythm-v2/{ID}.json'


def prepare():
    if PLAN.exists():
        raise FileExistsError('Preserve registered admission')
    parent_name = 'bench/rhythm-v2/c11-native-cell-chain-v1.json'
    runtime_name = 'bench/rhythm-v2/c12-mps-runtime-probe-v1.json'
    parent = json.loads((ROOT / parent_name).read_text())
    runtime = json.loads((ROOT / runtime_name).read_text())
    disk = admission(ROOT, GIB, 15 * GIB)
    if disk['status'] != 'admitted':
        raise ValueError('Disk admission')
    plan = dict(id=ID, parent_plan=parent_name, parent_sha256=sha(ROOT / parent_name),
        seed=parent['seed'], model=parent['model'], density_reference=parent['density_reference'],
        batches=parent['sample_stream'][:8], effective_batch=4,
        initial_state_path=runtime['initial_state_path'],
        initial_state_sha256=runtime['initial_state_sha256'],
        initial_weight_sha256=runtime['initial_weight_sha256'],
        runtime='2.14.0', runtime_target='artifacts/c12-mps-runtime-probe-v1/runtime',
        runtime_records=runtime['runtime_records'],
        optimizer=dict(lr=3e-4, weight_decay=.01, betas=[.9, .999], eps=1e-8, clip=1.),
        learning_rate_scope='Constant stress-probe LR, not formal warmup schedule',
        memory_fraction=.5, physical_cap_bytes=10 * GIB,
        last_clear_cap_bytes=3 * GIB, compressed_cap_bytes=GIB,
        last_four_growth_cap_bytes=256 * 2 ** 20,
        new_disk_peak_bytes=GIB, reserve_bytes=15 * GIB, wall_s=900,
        numeric_tolerance=1e-4, gradient_retention='First four microbatches: every live parameter gradient exactly unchanged across cleanup; first accumulated batch compares all CPU/MPS gradients',
        free_probe='After eight actual updates call legal request-only free decode on first request;record event count/type/hold coverage and actual completeness;no F1 gate',
        quality_authorized=False, health_authorized=False, cloud_compute_calls=0,
        failure_state='Save one model+optimizer checkpoint for diagnosis only;no automatic resume',
        source_sha256={f:sha(ROOT/f) for f in closure('scripts/run_mps_adamw_admission.py')+['tests/test_mps_request_lifecycle.py']},
        scope='Bounded pretraining runtime probe,not formal quality training',
        predecessors=['c11-native-cell-chain-v1','c12-context-shape-probe-v1','c12-mps-runtime-probe-v1'],
        shared_failure_modes=['total_process_memory','request_cleanup_gradient_loss'],
        mechanism_changes=dict(
            total_process_memory=dict(change='Verified isolated runtime+explicit cache lifetime+physical guard',prediction='Eight real updates remain under cap without compressed growth',discriminating_control='Same law/data,actual CPU/MPS accumulated gradient and resource samples'),
            request_cleanup_gradient_loss=dict(change='Never zero gradients within accumulation;drop current graph references only',prediction='All gradients survive cleanup and first accumulated batch agrees withCPU',discriminating_control='Actual MPS lifecycle regression and full-model first-four-microbatch exact retention')),
        pretraining_probe_receipts=[dict(status='completed',evidence='reports/local-integration-v1/mps-lifecycle-regression.json',scope='new_lifecycle_actual_mps_cpu_regression'),dict(status='completed',evidence='artifacts/c12-mps-runtime-probe-v1/summary.json',scope='same_model_newruntime_zero_update')],
        execution_ready=True,
        disk_admission=disk)
    write(PLAN, plan)
    for name in plan['source_sha256']:
        dest = ART / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    shutil.copy2(PLAN, ART / 'source/frozen-plan.json')


def fresh(plan, device):
    model = NativeCellChain(FullEventConfig(**plan['model']),
                            density_reference=plan['density_reference'])
    state = ROOT / plan['initial_state_path']
    if sha(state) != plan['initial_state_sha256']:
        raise ValueError('Common initial state identity')
    model.load_state_dict(torch.load(state, weights_only=True, map_location='cpu'))
    if weight_digest(model) != plan['initial_weight_sha256']:
        raise ValueError('Common parameter/buffer digest')
    return model.to(device)


def worker():
    plan = json.loads(PLAN.read_text())
    start = time.monotonic()
    report = dict(id=ID, status='running', updates=0, microbatches=[], steps=[],
                  cpu_mps={}, gradient_retention=[], cloud_compute_calls=0,
                  quality_authorized=False, health_authorized=False)
    stage = 'initialize'
    model = opt = None

    def stop(*_):
        raise TimeoutError('Registered admission wall/interrupt')

    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.alarm(plan['wall_s'])

    def save():
        report['wall_s'] = time.monotonic() - start
        write(ART / 'report.json', report)

    def resource():
        sample = memory_sample()
        sample['wall_s'] = time.monotonic() - start
        report['last_memory'] = sample
        if sample['physical_footprint_bytes'] > plan['physical_cap_bytes']:
            raise MemoryError('Own-process physical memory cap')
        if sample['driver_bytes'] > min(.8 * sample['recommended_bytes'], 10 * GIB):
            raise MemoryError('Driver cap')
        if admission(ROOT, GIB, 15 * GIB)['status'] != 'admitted':
            raise MemoryError('Disk reserve')
        if sum(p.stat().st_size for p in ART.rglob('*') if p.is_file()) > GIB:
            raise MemoryError('New artifact cap')
        return sample

    try:
        if torch.__version__ != plan['runtime']:
            raise ValueError('Isolated runtime required')
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
            raise ValueError('No fallback')
        torch.set_num_threads(4)
        torch.mps.set_per_process_memory_fraction(plan['memory_fraction'])
        report['runtime'] = dict(version=torch.__version__, path=torch.__file__, git=torch.version.git_version)
        panel, row = load_data()
        model = fresh(plan, 'mps')
        report['parameters'] = sum(p.numel() for p in model.parameters())
        report['initial_weight_sha256'] = weight_digest(model)
        cpu = fresh(plan, 'cpu')
        cpu.train()
        cpu_parts = []
        stage = 'cpu_accumulated_reference'
        for spec in plan['batches'][0]:
            rr = cell_row(panel, row(spec['index'], spec['view'], spec['field_index']))
            inputs, target = cell_batch(rr, 'cpu')
            output = cpu(inputs)
            loss, detail = cell_loss(cpu, output, inputs, rr['request'], target)
            (loss / 4).backward()
            cpu_parts.append({k:float(detail[k].detach()) for k in ('count','head','kind','tail')})
            del output, loss, detail, inputs, target, rr
            gc.collect()
        cpu_grad = {name:p.grad.detach().numpy().copy() for name,p in cpu.named_parameters() if p.grad is not None}
        del cpu
        gc.collect()
        resource()
        opt = torch.optim.AdamW(model.parameters(), lr=plan['optimizer']['lr'],
            weight_decay=.01, betas=(.9,.999), eps=1e-8)
        components = set()
        targets = Counter()
        for step, batch in enumerate(plan['batches'], 1):
            step_start = time.monotonic()
            model.train()
            opt.zero_grad(set_to_none=True)
            values = []
            detail_diffs = []
            for micro, spec in enumerate(batch):
                stage = f'update_{step}_micro_{micro}_forward'
                rr = cell_row(panel, row(spec['index'], spec['view'], spec['field_index']))
                components.add(str(panel.records[spec['index']]['component']))
                targets.update(e.type for e in rr['targets'])
                inputs, target = cell_batch(rr, 'mps')
                if step == 1 and micro == 0:
                    report['input_probe'] = audit_tensors(inputs)
                carried = sum(p.grad is not None for p in model.parameters())
                output = model(inputs)
                loss, detail = cell_loss(model, output, inputs, rr['request'], target)
                before_backward = resource()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite loss')
                values.append(float(loss.detach()))
                if step == 1:
                    detail_diffs.append({k:abs(float(detail[k].detach())-cpu_parts[micro][k]) for k in cpu_parts[micro]})
                (loss / 4).backward()
                after_backward = resource()
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise FloatingPointError('Nonfinite gradient')
                before = {name:p.grad.detach().cpu().numpy().copy() for name,p in model.named_parameters() if p.grad is not None} if step == 1 else None
                target_sha = hashlib.sha256(json.dumps([asdict(e) for e in sorted(rr['targets'])],sort_keys=True).encode()).hexdigest()
                del output, loss, detail, inputs, target, rr
                stage = f'update_{step}_micro_{micro}_cleanup'
                cleanup = clear_request_cache()
                after_clear = resource()
                if before is not None:
                    after_names = {name for name,p in model.named_parameters() if p.grad is not None}
                    if set(before) != after_names:
                        raise ValueError('Cleanup erased gradient support')
                    changed = [name for name,p in model.named_parameters() if p.grad is not None and not np.array_equal(before[name],p.grad.detach().cpu().numpy())]
                    report['gradient_retention'].append(dict(micro=micro,parameters=len(before),changed=changed))
                    if changed:
                        raise ValueError('Cleanup modified accumulated gradient')
                    del before
                report['microbatches'].append(dict(step=step,micro=micro,**spec,
                    carried_gradient_parameters=carried,target_sha256=target_sha,
                    before_backward=before_backward,after_backward=after_backward,
                    cleanup=cleanup,after_clear=after_clear))
            stage = f'update_{step}_optimizer'
            if step == 1:
                names = {name for name,p in model.named_parameters() if p.grad is not None}
                if set(cpu_grad) != names:
                    raise ValueError('CPU/MPS gradient support mismatch')
                differences = {name:float(np.abs(cpu_grad[name]-p.grad.detach().cpu().numpy()).max()) for name,p in model.named_parameters() if p.grad is not None}
                report['cpu_mps'] = dict(loss_part_differences=detail_diffs,
                    all_gradient_differences=differences,
                    gradient_max_diff=max(differences.values()),
                    loss_part_max_diff=max(v for d in detail_diffs for v in d.values()))
                if max(report['cpu_mps']['gradient_max_diff'],report['cpu_mps']['loss_part_max_diff']) > plan['numeric_tolerance']:
                    raise ValueError('CPU/MPS numeric admission failed')
                del cpu_grad
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
            if not math.isfinite(grad):
                raise FloatingPointError('Nonfinite clipped norm')
            before_step = resource()
            opt.step()
            report['updates'] = step
            after_step = resource()
            if any(not bool(torch.isfinite(v).all()) for state in opt.state.values() for v in state.values() if isinstance(v,torch.Tensor)):
                raise FloatingPointError('Nonfinite Adam state')
            if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise FloatingPointError('Nonfinite parameter')
            cleanup = clear_request_cache()
            sample = resource()
            report['steps'].append(dict(step=step,loss=float(np.mean(values)),gradient=grad,
                before_step=before_step,after_step=after_step,cleanup=cleanup,
                after_clear=sample,wall_s=time.monotonic()-step_start))
            save()
            print('update',step,round(sample['physical_footprint_bytes']/GIB,3),round(time.monotonic()-step_start,3),flush=True)
        report['exposure'] = dict(requests=32,unique_components=len(components),types=dict(targets),events=sum(targets.values()))
        stage = 'free_decode'
        first = plan['batches'][0][0]
        rr = cell_row(panel,row(first['index'],first['view'],first['field_index']))
        inputs,target = cell_batch(rr,'mps')
        with torch.no_grad():
            pred = decode_cells(model,inputs,rr['request'],{k:target[k] for k in ('heads','allowed')},deadline=start+plan['wall_s'])
        report['free_decode'] = pred
        report['free_decode_types'] = dict(Counter(e['type'] for e in pred['events']))
        report['free_decode_memory'] = resource()
        del inputs,target,rr
        clear_request_cache()
        last = resource()
        growth = report['steps'][7]['after_clear']['physical_footprint_bytes']-report['steps'][4]['after_clear']['physical_footprint_bytes']
        checks = dict(eight_updates=report['updates']==8,
            numeric=bool(report['cpu_mps']),gradients_preserved=all(not r['changed'] for r in report['gradient_retention']),
            free_complete=pred['complete'],last_footprint=last['physical_footprint_bytes']<=plan['last_clear_cap_bytes'],
            compressed=last['compressed_bytes']<=plan['compressed_cap_bytes'],
            last_four_growth=growth<=plan['last_four_growth_cap_bytes'])
        report['checks'] = checks
        report['last_four_growth_bytes'] = growth
        report['status'] = 'admission_passed' if all(checks.values()) else 'admission_checks_failed'
    except BaseException as exc:
        report['status'] = 'admission_memory_failed' if isinstance(exc,MemoryError) else 'admission_wall' if isinstance(exc,TimeoutError) else 'admission_failed'
        report['failure'] = dict(type=type(exc).__name__,error=str(exc),stage=stage)
    finally:
        signal.alarm(0)
        if report['status'] != 'admission_passed' and model is not None:
            path=ART/'failure-state.pt'
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict() if opt else None,updates=report['updates'],plan_sha256=sha(PLAN),resume_authorized=False),path)
            report['failure_checkpoint_sha256']=sha(path)
        save()


def run():
    if (ART/'report.json').exists():
        raise FileExistsError('No automatic rerun')
    plan=json.loads(PLAN.read_text())
    for source,expected in {**plan['source_sha256'],**plan['runtime_records']}.items():
        if sha(ROOT/source)!=expected:
            raise ValueError('Registered identity drift '+source)
    if admission(ROOT,GIB,15*GIB)['status']!='admitted':
        raise ValueError('Disk admission')
    env=os.environ.copy()
    env['PYTHONPATH']=str(ROOT/plan['runtime_target'])+os.pathsep+str(ROOT)
    with (ART/'run.log').open('w') as log:
        subprocess.run([str(ROOT/'.venv/bin/python'),'-u','-m','scripts.run_mps_adamw_admission','worker'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=plan['wall_s']+30,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run','worker'])
    args=parser.parse_args()
    globals()[args.action]()
