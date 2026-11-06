"""Bounded, explicit HN2 probes; prints receipts, never writes model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import threading
import time

import torch

from beatfcn.model.hn2_native import HN2Config, HN2Native
from beatfcn.train.hn2_native import two_pass, validate_targets
from beatfcn.train.native_diagnostic_priors import fit_priors


ROOT = Path(__file__).resolve().parents[1]


def tensor_digest(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(args):
    preparation_start = time.monotonic()
    torch.set_num_threads(4)
    if args.action == 'meta':
        with torch.device('meta'):
            model = HN2Native()
        return dict(parameters=sum(p.numel() for p in model.parameters()),
                    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad), torch=torch.__version__)
    import os
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise ValueError('Silent MPS fallback prohibited')
    if torch.__version__ != '2.14.0':
        raise ValueError('Real probes require the already installed isolated torch2.14.0 runtime')
    if args.device == 'mps':
        torch.mps.set_per_process_memory_fraction(.5)
    from beatfcn.train.modal_event_dataset import ModalEventDataset, collate_event_batch
    dataset = ModalEventDataset(ROOT, args.manifest, split='train', views=['draft'])
    row = dataset[args.index]
    batch = collate_event_batch([row])
    validate_targets(batch['inputs'], batch['targets'], batch['target_valid'])
    input_sha256 = tensor_digest(batch['inputs'])
    target_sha256 = tensor_digest(dict(targets=batch['targets'], target_valid=batch['target_valid']))
    stats = {}
    for k,v in batch['inputs'].items():
        if isinstance(v, torch.Tensor):
            stats[k] = dict(shape=list(v.shape), dtype=str(v.dtype), nonfinite=int((~torch.isfinite(v)).sum()),
                            min=float(v.min()) if v.numel() else None, max=float(v.max()) if v.numel() else None,
                            mean=float(v.float().mean()) if v.numel() else None,
                            std=float(v.float().std(unbiased=False)) if v.numel() else None)
    c = HN2Config() if args.full else HN2Config(d=args.d, heads=4, ffn=args.d*4, fine=max(16, args.d//2))
    torch.manual_seed(20261030)
    from beatfcn.train.process_memory import physical_footprint
    from scripts.check_experiment_storage import admission, GIB
    if admission(ROOT, 0, 15*GIB)['status'] != 'admitted':
        raise MemoryError('Disk15GiB reserve')
    done = threading.Event()
    samples = []
    start_guard = time.monotonic()
    old_handler = signal.getsignal(signal.SIGUSR1)
    def interrupt(*unused):
        raise MemoryError('HN2 probe physical10GiB or wall300s watchdog')
    signal.signal(signal.SIGUSR1, interrupt)
    def watch():
        while not done.wait(.25):
            value = physical_footprint()
            samples.append(value['physical_footprint_bytes'])
            if value['physical_footprint_bytes'] > 10*GIB or time.monotonic()-start_guard > 300:
                os.kill(os.getpid(), signal.SIGUSR1)
                return
    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    model = HN2Native(c).to(args.device)
    initial_tensor_sha256 = tensor_digest(model.state_dict())
    batch = {k: ({name:value.to(args.device) for name,value in v.items()} if k == 'inputs' else
                  v.to(args.device) if isinstance(v, torch.Tensor) else v) for k,v in batch.items()}
    prior = fit_priors([], [], [1, 1, 1, 1])
    if args.prior:
        p = json.loads((ROOT/args.prior).read_text())
        prior = p.get('duration', p)
    start = time.monotonic()
    preparation_wall_s = start-preparation_start
    if args.device == 'mps':
        torch.mps.synchronize()
    updates = getattr(args, 'updates', 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01) if updates else None
    initial = {n:p.detach().clone() for n,p in model.named_parameters()} if updates else None
    rng = torch.Generator(device=args.device).manual_seed(20261030)
    curve = []
    for step in range(max(updates, 1)):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        loss, metrics, z, _ = two_pass(model, batch['inputs'], batch['targets'], batch['target_valid'], prior,
                                      generator=rng)
        loss.backward()
        curve.append(float(loss.detach().cpu()))
        if optimizer is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
    if args.device == 'mps':
        torch.mps.synchronize()
    grads = {name:dict(finite=bool(torch.isfinite(p.grad).all()), norm=float(p.grad.float().norm()),
                       nonzero=int(torch.count_nonzero(p.grad))) for name,p in model.named_parameters() if p.grad is not None}
    adam = {name:dict(step=int(optimizer.state[p]['step']), delta_norm=float((p.detach()-initial[name]).float().norm()))
            for name,p in model.named_parameters() if p in optimizer.state} if optimizer is not None else {}
    final_tensor_sha256 = tensor_digest(model.state_dict())
    done.set()
    thread.join(timeout=1)
    signal.signal(signal.SIGUSR1, old_handler)
    def numbers(value):
        if isinstance(value, dict):
            return {k:numbers(v) for k,v in value.items()}
        return value.detach().cpu().item() if isinstance(value, torch.Tensor) else value
    return dict(parameters=sum(p.numel() for p in model.parameters() if p.requires_grad), torch=torch.__version__,
                device=args.device, dataset_index=args.index, target_count=int(batch['target_valid'].sum()),
                loss=float(loss.detach()), wall_s=time.monotonic()-start, metrics=numbers(metrics),
                preparation_wall_s=preparation_wall_s, total_wall_s=time.monotonic()-preparation_start,
                input_sha256=input_sha256, target_sha256=target_sha256,
                initial_tensor_sha256=initial_tensor_sha256, final_tensor_sha256=final_tensor_sha256,
                manifest_sha256=hashlib.sha256((ROOT/args.manifest).read_bytes()).hexdigest(),
                sampled_provisional_count=int(z['active'].sum()), tensors=stats, gradients=grads,
                optimizer_updates=updates, adam=adam, loss_curve=curve, cloud_calls=0, physical_sample_interval_s=.25,
                physical_peak_bytes=max(samples, default=physical_footprint()['physical_footprint_bytes']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['meta', 'real'])
    parser.add_argument('--manifest', default='artifacts/modal120m-data-v1/prepared-v3/manifest.json')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--prior')
    parser.add_argument('--updates', type=int, default=0)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False))
