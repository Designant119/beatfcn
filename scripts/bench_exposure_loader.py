"""CPU-only throughput benchmark for the multiprocessing exposure loader.

Run inside the Modal training image (the MERT features live on the volume) or
locally when the plan points at a local feature manifest.  No CUDA is touched.
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
import numpy as np

from beatfcn.train.exposure_loader import build_exposure_loader, plan_loader_settings
from beatfcn.train.local_request_panel import LocalRequestPanel

ROOT = Path(__file__).resolve().parents[1]


def build_panel(plan):
    from beatfcn.train.training_panel import build_training_panel
    return build_training_panel(ROOT, plan)


def tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def benchmark_loader(loader, *, batches, warmup=8, consumer_ms=0.):
    """Measure exposure items, physical batches and valid tokens separately.

    consumer_ms models a supplied consumer interval; it is not GPU evidence.
    Pinning/H2D/compute must still be measured in the actual CUDA training run.
    """
    if batches < 1 or warmup < 0 or consumer_ms < 0 or batches+warmup > len(loader):
        raise ValueError('Benchmark needs enough exposure items for warmup + measurement')
    tick = time.monotonic()
    iterator = iter(loader)
    startup_s = time.monotonic()-tick
    waits, tokens, requests, physical, payload = [], 0, 0, 0, 0
    max_payload = 0
    first_item_s = None
    started = None
    for index in range(warmup+batches):
        if index == warmup:
            started = time.monotonic()
        tick = time.monotonic()
        item = next(iterator)
        waited = time.monotonic()-tick
        if first_item_s is None:
            first_item_s = waited
        if index >= warmup:
            waits.append(waited)
            size = tensor_bytes(item)
            payload += size
            max_payload = max(max_payload, size)
            requests += len(item['indices'])
            physical += len(item['batches'])
            tokens += sum(int(seq['valid'].sum()) for _, _, seq in item['batches'])
        if consumer_ms:
            time.sleep(consumer_ms/1000.)
    wall = time.monotonic()-started
    result = dict(measured_exposure_items=batches, warmup_items=warmup,
        iterator_startup_s=startup_s, first_item_wait_s=first_item_s,
        wall_s=wall, exposure_items_per_s=batches/wall,
        physical_microbatches_per_s=physical/wall, requests_per_s=requests/wall,
        valid_tokens_per_s=tokens/wall, payload_mib_per_item=payload/batches/2**20,
        max_payload_mib=max_payload/2**20, data_wait_s=sum(waits),
        data_wait_fraction=sum(waits)/wall,
        wait_ms={f'p{p}': float(np.percentile(waits, p))*1000 for p in (50, 95, 99)},
        simulated_consumer_ms=consumer_ms,
        scope='CPU delivery only; pinning/H2D/actual GPU compute not measured')
    del iterator
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--workers', default='8')
    parser.add_argument('--batches', type=int, default=200)
    parser.add_argument('--prefetch', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--context', choices=['fork', 'spawn', 'forkserver'])
    parser.add_argument('--consumer-ms', type=float, default=0.)
    parser.add_argument('--cpu-budget', type=int, default=16)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    exposure = json.loads((ROOT / plan['exposure']).read_text())
    if plan.get('exposure_slice'):
        exposure = dict(exposure)
        exposure['steps'] = exposure['steps'][:int(plan['exposure_slice'])]
    torch.set_num_threads(1)
    for worker_text in str(args.workers).split(','):
        workers = int(worker_text)
        if workers < 0 or workers+2 > args.cpu_budget:
            raise ValueError('Leave at least two CPU cores for the consumer and pinning/I/O')
        panel = build_panel(plan)
        loader = build_exposure_loader(panel, exposure,
            microbatch=plan.get('physical_microbatch_requests', 4),
            input_shapes=plan.get('input_shapes'),
            sequence_buckets=plan.get('sequence_buckets'),
            num_workers=workers, pin_memory=False,
            prefetch_factor=args.prefetch, persistent_workers=False,
            multiprocessing_context=args.context, seed=plan.get('seed', 0),
            # Same vocabulary / onset / curve / counterfactual settings as the
            # runner, so the benchmark measures the items training consumes.
            **plan_loader_settings(plan))
        measurement = benchmark_loader(loader, batches=args.batches,
            warmup=args.warmup, consumer_ms=args.consumer_ms)
        result = dict(plan=str(args.plan), workers=workers, batches=args.batches,
            prefetch=args.prefetch, requests=len(panel.records),
            steps=len(exposure['steps']), cpu_budget=args.cpu_budget,
            host_cpu_count=os.cpu_count(),
            nominal_prefetch_items=workers*args.prefetch, loader_settings=plan_loader_settings(plan), **measurement)
        print(json.dumps(result, ensure_ascii=False))
        del loader, panel
        gc.collect()


if __name__ == '__main__':
    main()
