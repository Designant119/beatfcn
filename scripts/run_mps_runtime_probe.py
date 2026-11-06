"""Cold-process version comparison with identical request-boundary cache cleanup.

This probe performs forward/backward only. It never starts quality training.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.native_cell_chain import NativeCellChain
from beatfcn.train.native_cell_chain import cell_row, cell_batch, cell_loss
from beatfcn.train.process_memory import physical_footprint
from scripts.check_experiment_storage import admission, GIB
from scripts.run_local_evolution import load_data
from scripts.run_local_portfolio_scale import sha, write, weight_digest
from scripts.run_native_cell_chain import closure


ROOT = Path(__file__).resolve().parents[1]
ID = 'c12-mps-runtime-probe-v1'
PLAN = ROOT / f'bench/rhythm-v2/{ID}.json'
ART = ROOT / f'artifacts/{ID}'


def artifact_bytes():
    return sum(p.stat().st_size for p in ART.rglob('*') if p.is_file())


def prepare():
    if PLAN.exists():
        raise FileExistsError('Preserve the registered probe')
    if torch.__version__ != '2.6.0':
        raise ValueError('Save common initialization from the original runtime')
    parent_path = ROOT / 'bench/rhythm-v2/c11-native-cell-chain-v1.json'
    parent = json.loads(parent_path.read_text())
    disk = admission(ROOT, GIB, 15 * GIB)
    if disk['status'] != 'admitted' or artifact_bytes() > GIB:
        raise ValueError('Disk admission')
    state_path = ART / 'common-initial-state.pt'
    if state_path.exists():
        raise FileExistsError('Preserve common initialization')
    torch.manual_seed(parent['seed'])
    base = NativeCellChain(FullEventConfig(**parent['model']),
                           density_reference=parent['density_reference'])
    initial = weight_digest(base)
    torch.save(base.state_dict(), state_path)
    files = closure('scripts/run_mps_runtime_probe.py')
    records = sorted((ART / 'runtime').glob('*.dist-info/RECORD'))
    plan = dict(
        id=ID, parent_plan=str(parent_path.relative_to(ROOT)),
        parent_sha256=sha(parent_path), seed=parent['seed'],
        model=parent['model'], density_reference=parent['density_reference'],
        requests=[s for batch in parent['sample_stream'][:8] for s in batch],
        arms=['torch26_clear', 'torch214_clear'],
        versions={'torch26_clear': '2.6.0', 'torch214_clear': '2.14.0'},
        intervention='Runtime version only; both arms use identical explicit per-request gc/synchronize/empty_cache/synchronize; original dynamic model, loss and labels unchanged',
        inference_limit='Version comparison identifies a runtime-version effect, not one upstream commit in isolation; no quality claim',
        initial_state_path=str(state_path.relative_to(ROOT)),
        initial_state_sha256=sha(state_path), initial_weight_sha256=initial,
        runtime_records={str(p.relative_to(ROOT)): sha(p) for p in records},
        optimizer_updates=0, quality_authorized=False, cloud_compute_calls=0,
        arm_wall_s=300, total_wall_s=720,
        physical_footprint_cap_bytes=10 * GIB,
        new_disk_peak_bytes=GIB, reserve_bytes=15 * GIB,
        numeric_tolerance=1e-4,
        decision='Both arms retain actual loss/count outputs and first complete gradient. Runtime repair is supported only if new runtime completes all32 requests under cap and common numerics are within1e-4. No automatic quality launch. Missing/failed requests remain recorded.',
        source_sha256={name: sha(ROOT / name) for name in files},
        pretraining_probe_receipts=['reports/local-integration-v1/process-memory-validation.json'],
        disk_admission=disk,
    )
    write(PLAN, plan)
    for name in files:
        dest = ART / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    shutil.copy2(PLAN, ART / 'source/frozen-plan.json')


def arm(name):
    plan = json.loads(PLAN.read_text())
    started = time.monotonic()
    rows = []
    failure = None
    stage = 'initialization'
    if torch.__version__ != plan['versions'][name]:
        raise ValueError('Unexpected runtime')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise ValueError('No silent CPU fallback')
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.5)

    def stop(*_):
        raise TimeoutError('Registered arm wall/interrupt')

    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.alarm(plan['arm_wall_s'])
    base = NativeCellChain(FullEventConfig(**plan['model']),
                          density_reference=plan['density_reference'])
    state_path = ROOT / plan['initial_state_path']
    if sha(state_path) != plan['initial_state_sha256']:
        raise ValueError('Common state bytes changed')
    base.load_state_dict(torch.load(state_path, map_location='cpu', weights_only=True))
    if weight_digest(base) != plan['initial_weight_sha256']:
        raise ValueError('Common tensor state changed')
    model = base.to('mps')
    panel, row = load_data()
    identity = dict(version=torch.__version__, git_version=torch.version.git_version,
                    module_path=torch.__file__, initial_weight_sha256=plan['initial_weight_sha256'])

    def memory():
        torch.mps.synchronize()
        return dict(**physical_footprint(),
                    driver_bytes=torch.mps.driver_allocated_memory(),
                    allocated_bytes=torch.mps.current_allocated_memory(),
                    recommended_bytes=torch.mps.recommended_max_memory(),
                    wall_s=time.monotonic() - started)

    def guard(sample):
        if sample['physical_footprint_bytes'] > plan['physical_footprint_cap_bytes']:
            raise MemoryError('Own-process physical footprint hard cap')
        if sample['driver_bytes'] > min(.8 * sample['recommended_bytes'], 10 * GIB):
            raise MemoryError('MPS driver hard cap')
        if admission(ROOT, GIB, 15 * GIB)['status'] != 'admitted' or artifact_bytes() > GIB:
            raise MemoryError('Disk reserve/artifact cap')

    status = 'running'
    try:
        for i, spec in enumerate(plan['requests']):
            stage = f'request_{i}_forward'
            begin = time.monotonic()
            model.train()
            model.zero_grad(set_to_none=True)
            rr = cell_row(panel, row(spec['index'], spec['view'], spec['field_index']))
            inputs, target = cell_batch(rr, 'mps')
            output = model(inputs)
            loss, parts = cell_loss(model, output, inputs, rr['request'], target)
            after_forward = memory()
            guard(after_forward)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            stage = f'request_{i}_backward'
            loss.backward()
            after_backward = memory()
            guard(after_backward)
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise ValueError('Nonfinite gradient')
            value = float(loss.detach())
            np.save(ART / f'{name}-count-{i:02d}.npy', output['count'].detach().cpu().numpy())
            if i == 0:
                np.savez_compressed(ART / f'{name}-first-gradients.npz', **{
                    key: p.grad.detach().cpu().numpy()
                    for key, p in model.named_parameters() if p.grad is not None})
            # Identical lifetime and cache calls in both arms. No graph is
            # intentionally retained across requests, and no optimizer exists.
            model.zero_grad(set_to_none=True)
            del output, loss, parts, inputs, target, rr
            gc.collect()
            stage = f'request_{i}_clear'
            before_clear = memory()
            guard(before_clear)
            torch.mps.empty_cache()
            after_clear = memory()
            guard(after_clear)
            entry = dict(position=i, **spec, loss=value,
                         step_wall_s=time.monotonic() - begin,
                         after_forward=after_forward, after_backward=after_backward,
                         before_clear=before_clear, after_clear=after_clear)
            rows.append(entry)
            write(ART / f'{name}.json', dict(status='running', identity=identity,
                                            rows=rows, optimizer_updates=0))
            print(name, i + 1, round(after_backward['physical_footprint_bytes'] / GIB, 3),
                  round(after_clear['physical_footprint_bytes'] / GIB, 3),
                  round(entry['step_wall_s'], 3), flush=True)
        status = 'completed'
    except BaseException as exc:
        status = 'resource_stopped' if isinstance(exc, (MemoryError, TimeoutError)) else 'failed'
        failure = dict(type=type(exc).__name__, error=str(exc), stage=stage,
                       memory=physical_footprint())
    finally:
        signal.alarm(0)
        write(ART / f'{name}.json', dict(status=status, identity=identity, rows=rows,
              failure=failure, wall_s=time.monotonic() - started,
              optimizer_updates=0, cloud_compute_calls=0))


def run():
    if (ART / 'summary.json').exists():
        raise FileExistsError('No automatic rerun')
    plan = json.loads(PLAN.read_text())
    for source, expected in {**plan['source_sha256'], **plan['runtime_records']}.items():
        if sha(ROOT / source) != expected:
            raise ValueError('Registered identity drift: ' + source)
    if admission(ROOT, GIB, 15 * GIB)['status'] != 'admitted':
        raise ValueError('Disk admission')
    start = time.monotonic()
    for name in plan['arms']:
        if (ART / f'{name}.json').exists():
            raise FileExistsError('No automatic arm retry')
        if plan['total_wall_s'] - (time.monotonic() - start) < plan['arm_wall_s']:
            raise TimeoutError('Insufficient full arm budget')
        env = os.environ.copy()
        if name == 'torch214_clear':
            env['PYTHONPATH'] = str(ART / 'runtime') + os.pathsep + str(ROOT)
        with (ART / f'{name}.log').open('w') as log:
            subprocess.run([str(ROOT / '.venv/bin/python'), '-u', '-m',
                            'scripts.run_mps_runtime_probe', 'arm', name],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=plan['arm_wall_s'] + 30)
    a, b = [json.loads((ART / f'{name}.json').read_text()) for name in plan['arms']]
    common = min(len(a['rows']), len(b['rows']))
    count_diff = max((float(np.abs(np.load(ART / f'torch26_clear-count-{i:02d}.npy') -
                                  np.load(ART / f'torch214_clear-count-{i:02d}.npy')).max())
                      for i in range(common)), default=None)
    loss_diff = max((abs(a['rows'][i]['loss'] - b['rows'][i]['loss'])
                     for i in range(common)), default=None)
    gradient_diff = None
    if common:
        with np.load(ART / 'torch26_clear-first-gradients.npz') as x, np.load(ART / 'torch214_clear-first-gradients.npz') as y:
            if set(x.files) != set(y.files):
                raise ValueError('Gradient support mismatch')
            gradient_diff = max(float(np.abs(x[key] - y[key]).max()) for key in x.files)
    numeric = common > 0 and max(count_diff, loss_diff, gradient_diff) <= plan['numeric_tolerance']
    write(ART / 'summary.json', dict(id=ID, status='completed_bounded_probe',
        arms={n: dict(status=r['status'], completed_requests=len(r['rows']),
                     wall_s=r['wall_s'], failure=r['failure'])
              for n, r in zip(plan['arms'], (a, b))},
        common_requests=common, max_count_logit_diff=count_diff, max_loss_diff=loss_diff,
        first_all_gradient_max_diff=gradient_diff, numeric_parity_passed=numeric,
        runtime_repair_supported=numeric and b['status'] == 'completed' and len(b['rows']) == 32,
        wall_s=time.monotonic() - start, new_artifact_bytes=artifact_bytes(),
        optimizer_updates=0, quality_authorized=False, cloud_compute_calls=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'arm', 'run'])
    parser.add_argument('name', nargs='?')
    args = parser.parse_args()
    if args.action == 'arm':
        arm(args.name)
    else:
        globals()[args.action]()
