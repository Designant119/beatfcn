"""Bounded training rounds with an explicit review boundary every 1000 updates."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import modal
from beatfcn.cloud import train_app as training

app = modal.App('beatfcn-evolution').include(training.app)
image = training.image.add_local_dir(str(training.REPO / 'bench' / 'capability-v1'), '/bench/capability-v1')
data = training.data
CAMPAIGN = 'capability-evolution-20261024'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def load_model(path, device='cuda'):
    import torch
    from beatfcn.model.p1 import P1Config, P1Model
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if not all(bool(torch.isfinite(t).all()) for t in ck['model'].values()
               if torch.is_tensor(t) and t.is_floating_point()):
        raise ValueError('nonfinite checkpoint')
    model = P1Model(P1Config(**ck['config'])).to(device).eval()
    model.load_state_dict(ck['model'])
    return model, ck


@app.function(image=image, gpu='L40S', volumes={'/data': data}, cpu=6.0, memory=49152, timeout=240)
def replay_gradients(source_run: str, checkpoint_path: str, variant: dict, policy: dict,
                     output: str, last_batches: int = 4):
    """Rebuild consumed requests from provenance and isolate backward without the async loader."""
    import torch
    from dataclasses import replace
    from beatfcn.model.p1 import P1Model
    from beatfcn.model.batch import collate
    from beatfcn.train.data import Corpus
    from beatfcn.train.loop import CoreSampler, to_device
    started = time.perf_counter()
    base, ck = load_model(checkpoint_path, 'cpu')
    model = P1Model(replace(base.c, **variant)).cuda().train()
    model.load_state_dict(base.state_dict(), strict=False)
    del base
    ca = ck['data']
    rows, _, _ = training._select_rows(ca['taps_key'], tuple(ca['layers']), ca['pca'], ca['whiten'], '', 0, '', ck['manifest_name'])
    corpus = Corpus(**ca)
    chart_rows = corpus.train_rows(rows)
    lookup = {r['chart_id']: r for r in chart_rows}
    sampler = CoreSampler(chart_rows, corpus.chart, corpus.song_info, 7, salient=corpus.salient_us,
                          perturb=True, request_mix=policy['request_mix'],
                          control_drop_all_p=policy['control_drop_all_p'])
    records = [json.loads(line) for line in Path(f'/data/runs/{source_run}/samples.jsonl').read_text().splitlines()]
    report = {'source_run': source_run, 'microbatches_consumed': len(records), 'cases': []}
    Path(output).mkdir(parents=True, exist_ok=True)
    for index in range(max(0, len(records) - last_batches), len(records)):
        examples, songs = [], []
        cache = {}
        for row in records[index]:
            paired = row.get('pair_id')
            ex = sampler.core(lookup[row['chart_id']], row['seed'],
                              request_kind='reference' if paired else None,
                              core_start=row['c0_us'], pair_id=paired)
            if ex.provenance['notes'] != row['notes'] or ex.provenance['holds'] != row['holds']:
                raise ValueError('provenance reconstruction differs from consumed request')
            if row['song'] not in cache:
                cache[row['song']] = corpus.song(row['song'], in_memory=True)
            examples.append(ex)
            songs.append(cache[row['song']])
        bt = to_device(collate(examples, songs, model.c.taps, model.c.top_tap), torch.device('cuda'))
        model.zero_grad(set_to_none=True)
        torch.manual_seed(7)
        try:
            with torch.autograd.detect_anomaly():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, metrics = model(bt)
                loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            case = {'batch_index': index, 'finite': True, 'loss': float(loss.detach()), 'grad_norm': float(norm)}
        except RuntimeError as exc:
            case = {'batch_index': index, 'finite': False, 'error': str(exc)}
            torch.save({'batch': {k: v.cpu() if torch.is_tensor(v) else v for k, v in bt.items()},
                        'config': model.c, 'checkpoint_path': checkpoint_path}, f'{output}/failed_batch.pt')
        report['cases'].append(case)
        print(json.dumps(case), flush=True)
        write_json(f'{output}/report.json', report)
        data.commit()
        if not case['finite']:
            break
        del bt, examples, songs, cache
    report['function_wall_s'] = time.perf_counter() - started
    write_json(f'{output}/report.json', report)
    data.commit()
    return report


@app.function(image=image, gpu='L40S', volumes={'/data': data}, cpu=4.0, memory=16384, timeout=120)
def probe_staged_allocation(output: str, checkpoint_path: str):
    """Exercise the actual song-loader allocation against a pending default-stream writer."""
    import gc
    import tempfile
    from unittest.mock import patch
    import numpy as np
    import torch
    from beatfcn.train.pool import SongPool
    report = {'cases': []}
    with tempfile.TemporaryDirectory() as directory:
        paths = []
        shape = (2048, 8, 32)
        for index in range(2):
            path = Path(directory) / f'{index}.npy'
            np.save(path, np.full(shape, index + 1, dtype=np.float16))
            paths.append(str(path))
        def loader(sha):
            raise AssertionError('staged path expected')
        loader.paths = lambda sha, mel: paths
        pool = SongPool(loader, (9,), loaders=1, mel=False)
        original_empty = torch.empty
        def side_alloc(*args, **kwargs):
            if torch.device(kwargs.get('device', 'cpu')).type == 'cuda':
                with torch.cuda.stream(pool.stream):
                    return original_empty(*args, **kwargs)
            return original_empty(*args, **kwargs)
        for safe in (False, True):
            for repeat in range(3):
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                old = original_empty(shape, dtype=torch.float16, device='cuda')
                pointer = old.data_ptr()
                # Delayed old writes make reuse on another stream deterministic.
                torch.cuda._sleep(200_000_000)
                old.fill_(float('nan'))
                del old
                if safe:
                    with patch('torch.empty', side_effect=side_alloc):
                        outputs, _, _ = pool._load_staged('probe')
                else:
                    outputs, _, _ = pool._load_staged('probe')
                torch.cuda.synchronize()
                case = {'allocation': 'copy_stream' if safe else 'current_code', 'repeat': repeat,
                        'reused_pending_pointer': any(t.data_ptr() == pointer for t in outputs),
                        'nonfinite_values': sum(int((~torch.isfinite(t)).sum()) for t in outputs),
                        'matches_files': all(bool((t == i + 1).all()) for i, t in enumerate(outputs))}
                report['cases'].append(case)
                print(json.dumps(case), flush=True)
                del outputs
        pool.pool.shutdown()
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    report['checkpoint'] = {'path': checkpoint_path, 'step': ck['step'],
                            'finite_model': all(bool(torch.isfinite(t).all()) for t in ck['model'].values())}
    report['reproduced'] = any(not r['matches_files'] for r in report['cases'] if r['allocation'] == 'current_code')
    report['candidate_passed'] = all(r['matches_files'] for r in report['cases'] if r['allocation'] == 'copy_stream')
    write_json(output, report)
    data.commit()
    return report


@app.function(image=image, volumes={'/data': data}, cpu=8.0, memory=32768, timeout=300)
def prepare_metadata_tape(output: str, checkpoint_path: str, policy: dict, rollout_cache: str,
                          n_batches: int, seed: int, micro_batch: int = 8):
    """Finish request construction on CPU; preserve every accepted batch and its provenance."""
    from collections import Counter
    import torch
    from beatfcn.train.tape import policy_digest
    torch.set_num_threads(1)
    started = time.perf_counter()
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    ca, config, rows_sha = ck['data'], ck['config'], ck['rows_sha256']
    rows, _, info = training._select_rows(ca['taps_key'], tuple(ca['layers']), ca['pca'], ca['whiten'], '', 0, '', ck['manifest_name'])
    if info['rows_sha256'] != rows_sha:
        raise ValueError('metadata source checkpoint uses different data')
    del ck
    root = Path(output)
    if (root / 'manifest.json').exists():
        raise ValueError('metadata tape already exists; inspect it rather than overwrite')
    root.mkdir(parents=True, exist_ok=True)
    stream = torch.utils.data.DataLoader(training._stream(rows, ca, seed, micro_batch, tuple(config['taps']), config['top_tap'],
                                                          pool=1, training_policy=policy, rollout_cache=rollout_cache),
                                         batch_size=None, num_workers=8, prefetch_factor=4, pin_memory=False, in_order=False)
    raw = iter(stream)
    manifest = {'schema': 'beatfcn.metadata_tape.v1', 'complete': False, 'rows_sha256': rows_sha,
                'policy_sha256': policy_digest(policy), 'policy': policy, 'micro_batch': micro_batch,
                'seed': seed, 'checkpoint': checkpoint_path, 'rollout_cache': rollout_cache, 'parts': [], 'batches': 0}
    exposure = Counter()
    with (root / 'provenance.jsonl').open('w') as provenance:
        while manifest['batches'] < n_batches:
            chunk = [next(raw) for _ in range(min(32, n_batches - manifest['batches']))]
            for batch, _ in chunk:
                records = batch.get('_provenance', [])
                provenance.write(json.dumps(records) + '\n')
                for record in records:
                    exposure[f"{record.get('mode')}:{record.get('kind')}:cores"] += 1
                    exposure[f"{record.get('mode')}:{record.get('kind')}:notes"] += record.get('notes', 0)
            path = root / f'part-{len(manifest["parts"]):04d}.pt'
            torch.save(chunk, path)
            payload = path.read_bytes()
            manifest['parts'].append({'file': path.name, 'bytes': len(payload),
                                      'sha256': hashlib.sha256(payload).hexdigest(), 'batches': len(chunk)})
            manifest['batches'] += len(chunk)
            if manifest['batches'] % 128 == 0:
                print(json.dumps({'event': 'metadata_prepared', 'batches': manifest['batches'],
                                  'requested': n_batches, 'wall_s': time.perf_counter() - started}), flush=True)
    del raw, stream
    manifest.update(complete=True, exposure=dict(exposure), function_wall_s=time.perf_counter() - started)
    write_json(root / 'manifest.json', manifest)
    data.commit()
    return manifest


@app.function(image=image, volumes={'/data': data}, cpu=4.0, memory=32768, timeout=300)
def compress_metadata_tape(source: str, output: str):
    """Losslessly compress the frozen tape on CPU; report actual volume-read cost."""
    import gzip
    from beatfcn.train.data import read_bytes
    started = time.perf_counter()
    source, root = Path(source), Path(output)
    if (root / 'manifest.json').exists():
        raise ValueError('compressed metadata tape already exists')
    manifest = json.loads((source / 'manifest.json').read_text())
    if not manifest.get('complete'):
        raise ValueError('cannot compress an incomplete tape')
    root.mkdir(parents=True, exist_ok=True)
    original_parts = manifest['parts']
    manifest = {**manifest, 'parts': [], 'complete': False, 'compressed_from': str(source)}
    timing = dict(read_s=0., compress_s=0., write_s=0.)
    print(json.dumps({'event': 'tape_compress_start', 'parts': len(original_parts)}), flush=True)
    for index, part in enumerate(original_parts):
        if Path(part['file']).name != part['file'] or part.get('codec') not in (None, 'raw'):
            raise ValueError('expected an uncompressed tape part filename')
        t = time.perf_counter()
        payload = read_bytes(source / part['file'])
        timing['read_s'] += time.perf_counter() - t
        if len(payload) != part['bytes'] or hashlib.sha256(payload).hexdigest() != part['sha256']:
            raise ValueError('source tape integrity check failed')
        t = time.perf_counter()
        compressed = gzip.compress(payload, compresslevel=1, mtime=0)
        if gzip.decompress(compressed) != payload:
            raise ValueError('compression changed the frozen tape')
        timing['compress_s'] += time.perf_counter() - t
        name = part['file'] + '.gz'
        t = time.perf_counter()
        (root / name).write_bytes(compressed)
        timing['write_s'] += time.perf_counter() - t
        manifest['parts'].append({**part, 'file': name, 'codec': 'gzip', 'raw_bytes': part['bytes'],
                                  'raw_sha256': part['sha256'], 'bytes': len(compressed),
                                  'sha256': hashlib.sha256(compressed).hexdigest()})
        if index == 0 or (index + 1) % 8 == 0:
            print(json.dumps({'event': 'tape_compress', 'parts': index + 1, 'timing': timing,
                              'bytes': sum(p['bytes'] for p in manifest['parts'])}), flush=True)
    (root / 'provenance.jsonl').write_bytes(read_bytes(source / 'provenance.jsonl'))
    manifest.update(complete=True, compression_timing=timing, compression_wall_s=time.perf_counter() - started)
    write_json(root / 'manifest.json', manifest)
    data.commit()
    return manifest


@app.function(image=image, gpu='L40S', volumes={'/data': data}, cpu=12.0, memory=49152, timeout=240)
def probe_transport(output: str, checkpoint_path: str, prep: str):
    """Fixed real inputs: isolate host copies, then compare resident and staged CUDA work.

    No optimiser updates or checkpoint writes. Repeat default/one-thread/default
    so a warmed allocator cannot masquerade as a CPU-thread improvement.
    """
    import os
    import statistics
    import numpy as np
    import torch
    from beatfcn.cloud.gpu_util import GpuUtil
    from beatfcn.model.batch import collate_meta, pack_batch, unpack_batch, attach_audio
    from beatfcn.train.data import Corpus
    from beatfcn.train.loop import to_device
    from beatfcn.train.pinned import PinnedRing
    from beatfcn.train.pool import SongPool
    started = time.perf_counter()
    default_threads = torch.get_num_threads()
    def cpu_stat():
        path = Path('/sys/fs/cgroup/cpu.stat')
        return {k: int(v) for k, v in (line.split() for line in path.read_text().splitlines())} if path.exists() else {}
    def cpu_delta(before):
        return {k: v - before.get(k, v) for k, v in cpu_stat().items()}
    report = {'default_intraop_threads': default_threads, 'interop_threads': torch.get_num_interop_threads(),
              'affinity_cpus': len(os.sched_getaffinity(0)),
              'cpu_max': (Path('/sys/fs/cgroup/cpu.max').read_text().strip()
                          if Path('/sys/fs/cgroup/cpu.max').exists() else None),
              'host_copy': [], 'compute': [], 'checkpoint': checkpoint_path,
              'note': 'Fixed 32 validation cores; short throughput diagnostic, not quality evaluation.'}
    def save_report():
        report['function_wall_s'] = time.perf_counter() - started
        write_json(output, report)
        data.commit()
    torch.set_num_threads(1)
    model, ck = load_model(checkpoint_path)
    model.train()
    corpus = Corpus(**ck['data'])
    prepared = torch.load(f'/data/prep/{prep}.pt', weights_only=False)
    if prepared['info']['rows_sha256'] != ck['rows_sha256']:
        raise ValueError('diagnostic validation fixture differs from checkpoint data')
    cores = prepared['val'][:32]
    metas = []
    for i in range(0, len(cores), 8):
        chunk = cores[i:i + 8]
        metas.append(pack_batch(collate_meta([e for e, _ in chunk], [corpus.song_info(s) for _, s in chunk],
                                            model.c.taps, model.c.top_tap, model.c.tap_dim, 1024, corpus.quant)))
    ring = PinnedRing(depth=8, pin=True)
    # Allocate all slots at the largest actual buffer sizes before timing reuse.
    maxima = {k: max((m['_packed'][k] for m in metas), key=lambda t: t.numel()) for k in metas[0]['_packed']}
    for _ in range(ring.depth):
        ring.take(maxima)
    report['packed_bytes'] = [sum(t.numel() * t.element_size() for t in m['_packed'].values()) for m in metas]
    for threads in (default_threads, 1, default_threads):
        torch.set_num_threads(threads)
        durations = []
        before = cpu_stat()
        for i in range(16):
            packed = metas[i % len(metas)]['_packed']
            t = time.perf_counter()
            views, _ = ring.take(packed)
            durations.append(time.perf_counter() - t)
        # Check values outside the timing window without the multi-thread pool.
        torch.set_num_threads(1)
        exact = all(torch.equal(views[k], packed[k]) for k in packed)
        case = {'threads': threads, 'median_ms': 1000 * statistics.median(durations),
                'total_s': sum(durations), 'p95_ms': 1000 * sorted(durations)[-1], 'exact': exact,
                'cgroup_cpu_delta': cpu_delta(before)}
        report['host_copy'].append(case)
        print(json.dumps({'event': 'transport_host_copy', **case}), flush=True)
        save_report()
    torch.set_num_threads(1)
    pool = SongPool(training._song_loader(corpus), model.c.taps, 'cuda', capacity=16, loaders=4, mel=False)
    keys = sorted({s for _, s in cores})
    pool.request(keys)
    for key in keys:
        pool.get(key)
    scales = torch.stack([torch.broadcast_to(torch.as_tensor(np.asarray(corpus.scale[l], np.float32)), (model.c.tap_dim,))
                          for l in model.c.taps]).clone().cuda() if corpus.quant else None
    def staged(index):
        packed = dict(metas[index])
        packed['_packed'], slot = ring.take(packed['_packed'])
        batch = unpack_batch(packed)
        attach_audio(batch, pool.get, 'cuda', scales)
        batch = to_device(batch, torch.device('cuda'))
        ring.done(slot, torch.cuda.current_stream())
        return batch
    full = [staged(i) for i in range(len(metas))]
    torch.cuda.synchronize()
    report['input_exact'] = all(torch.equal(full[i][k], candidate[k])
                                for i in range(len(metas)) for candidate in [staged(i)]
                                for k in full[i] if torch.is_tensor(full[i][k]))
    if not report['input_exact']:
        raise RuntimeError('staged and resident diagnostic inputs differ')
    def backward(batch):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = model(dict(batch))
        (loss / 4).backward()
    for batch in full:
        backward(batch)
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    mon = GpuUtil()
    reference_grads = None
    try:
        for mode, threads in [('resident', 1), ('resident', default_threads),
                              ('staged', default_threads), ('staged', 1), ('staged', default_threads)]:
            torch.set_num_threads(threads)
            torch.manual_seed(20261024)
            torch.cuda.synchronize()
            mon.take()
            before = cpu_stat()
            t = time.perf_counter()
            for i in range(16):
                if i % 4 == 0:
                    model.zero_grad(set_to_none=True)
                backward(full[i % len(full)] if mode == 'resident' else staged(i % len(full)))
                if i % 4 == 3:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t
            case = {'mode': mode, 'threads': threads, 'microbatches': 16, 'wall_s': elapsed,
                    'cores_per_s': 128 / elapsed, 'cgroup_cpu_delta': cpu_delta(before),
                    'gpu': mon.take(), 'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
            torch.set_num_threads(1)
            grads = {k: p.grad.detach().cpu() for k, p in model.named_parameters() if p.grad is not None}
            if reference_grads is None:
                reference_grads = grads
            case['gradient_max_abs_diff'] = max(float((grads[k] - v).abs().max()) for k, v in reference_grads.items())
            case['gradient_names_match'] = grads.keys() == reference_grads.keys()
            case['grad_norm'] = float(norm)
            report['compute'].append(case)
            print(json.dumps({'event': 'transport_compute', **case}), flush=True)
            save_report()
    finally:
        mon.close()
        pool.pool.shutdown()
    report['complete'] = True
    save_report()
    return report


@app.function(image=image, gpu='L40S', volumes={'/data': data}, cpu=12.0, memory=49152, timeout=240)
def probe_live_input(output: str, checkpoint_path: str, policy: dict, rollout_cache: str,
                     compare_copy_methods: bool = False, phase_batches: int = 32, metadata_tape: str = ''):
    """Reintroduce the actual producers and song swaps without advancing model weights."""
    from collections import deque
    import numpy as np
    import torch
    from beatfcn.cloud.gpu_util import GpuUtil
    from beatfcn.model.batch import unpack_batch, attach_audio
    from beatfcn.train.data import Corpus
    from beatfcn.train.loop import to_device
    from beatfcn.train.pinned import PinnedRing
    from beatfcn.train.pool import SongPool
    started = time.perf_counter()
    default_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    print(json.dumps({'event': 'live_probe_start', 'checkpoint': checkpoint_path}), flush=True)
    model, ck = load_model(checkpoint_path, 'cpu')
    startup = {'checkpoint_s': time.perf_counter() - started}
    print(json.dumps({'event': 'live_probe_checkpoint', **startup}), flush=True)
    ca = ck['data']
    rows, _, _ = training._select_rows(ca['taps_key'], tuple(ca['layers']), ca['pca'], ca['whiten'], '', 0, '', ck['manifest_name'])
    if metadata_tape:
        from beatfcn.train.tape import MetadataTape
        tape_started = time.perf_counter()
        def tape_progress(progress):
            if progress['batches'] == 32 or progress['batches'] % 256 == 0:
                print(json.dumps({'event': 'tape_load', **progress}), flush=True)
        stream = MetadataTape(metadata_tape, rows_sha256=ck['rows_sha256'], policy=policy, micro_batch=8,
                              progress=tape_progress)
        startup['tape_s'] = time.perf_counter() - tape_started
        startup['tape_timing'] = stream.load_timing
        print(json.dumps({'event': 'live_probe_tape_loaded', **startup}), flush=True)
    else:
        stream = torch.utils.data.DataLoader(training._stream(rows, ca, 20261024, 8, model.c.taps, model.c.top_tap,
                                                          pool=1, training_policy=policy, rollout_cache=rollout_cache),
                                         batch_size=None, num_workers=8, prefetch_factor=4,
                                         persistent_workers=True, pin_memory=False, in_order=False)
    raw = iter(stream)  # fork workers before initialising the CUDA context
    model.cuda().train()
    corpus = Corpus(**ca)
    pool = SongPool(training._song_loader(corpus), model.c.taps, 'cuda', capacity=40, loaders=4, mel=False)
    ring, queue = PinnedRing(depth=40, pin=True), deque()
    scales = torch.stack([torch.broadcast_to(torch.as_tensor(np.asarray(corpus.scale[l], np.float32)), (model.c.tap_dim,))
                          for l in model.c.taps]).clone().cuda() if corpus.quant else None
    report = {'checkpoint': checkpoint_path, 'default_intraop_threads': default_threads, 'phases': [], 'metadata_tape': metadata_tape,
              'startup': startup,
              'note': 'Actual random worker batches; same model weights, no optimizer updates. Phase inputs differ.'}
    mon = GpuUtil()
    try:
        phases = ([(default_threads, 'torch')] * 3 if metadata_tape else
                  [(default_threads, method) for method in ('torch', 'bytes', 'torch', 'bytes')]
                  if compare_copy_methods else [(n, 'torch') for n in (default_threads, 1, default_threads)])
        for threads, method in phases:
            torch.set_num_threads(threads)
            ring.copy_method = method
            torch.cuda.synchronize()
            mon.take()
            before_ring = dict(ring.stats)
            timing = dict(loader_s=0., pin_s=0., attach_s=0., model_dispatch_s=0.)
            phase_start = time.perf_counter()
            for index in range(phase_batches):
                while len(queue) < 16:
                    t = time.perf_counter()
                    packed, _ = next(raw)
                    timing['loader_s'] += time.perf_counter() - t
                    t = time.perf_counter()
                    packed['_packed'], slot = ring.take(packed['_packed'])
                    timing['pin_s'] += time.perf_counter() - t
                    batch = unpack_batch(packed)
                    keys = list(batch['_audio']['keys'])
                    pool.request(keys)
                    queue.append((batch, slot, keys))
                batch, slot, keys = queue.popleft()
                t = time.perf_counter()
                attach_audio(batch, pool.get, 'cuda', scales)
                batch = to_device(batch, torch.device('cuda'))
                pool.release(keys)
                ring.done(slot, torch.cuda.current_stream())
                timing['attach_s'] += time.perf_counter() - t
                t = time.perf_counter()
                if index % 4 == 0:
                    model.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, _ = model(batch)
                (loss / 4).backward()
                if index % 4 == 3:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                timing['model_dispatch_s'] += time.perf_counter() - t
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - phase_start
            case = {'threads': threads, 'copy_method': method, 'microbatches': phase_batches, 'wall_s': elapsed,
                    'cores_per_s': 8 * phase_batches / elapsed,
                    'timing': timing, 'ring_delta': {k: v - before_ring[k] for k, v in ring.stats.items()},
                    'gpu': mon.take(), 'grad_norm': float(norm), 'pool': dict(pool.stats), 'pool_memory': pool.telemetry()}
            report['phases'].append(case)
            report['function_wall_s'] = time.perf_counter() - started
            print(json.dumps({'event': 'live_transport', **case}), flush=True)
            write_json(output, report)
            data.commit()
        if compare_copy_methods:
            report['paired_copy'] = []
            rings = {method: PinnedRing(depth=1, pin=True, copy_method=method) for method in ('torch', 'bytes')}
            for index in range(8):
                packed, _ = next(raw)
                # Keep real CUDA model work in flight, as at the production copy site.
                model.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, _ = model(batch)
                loss.backward()
                order = ('torch', 'bytes') if index % 2 == 0 else ('bytes', 'torch')
                values, times = {}, {}
                for method in order:
                    before = rings[method].stats['copy_s']
                    values[method], _ = rings[method].take(packed['_packed'])
                    times[method] = rings[method].stats['copy_s'] - before
                exact = all(np.array_equal(values['torch'][k].view(torch.uint8).numpy(),
                                           values['bytes'][k].view(torch.uint8).numpy()) for k in packed['_packed'])
                report['paired_copy'].append({'order': order, 'copy_s': times, 'exact': exact,
                                               'shared': all(t.is_shared() for t in packed['_packed'].values())})
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            print(json.dumps({'event': 'paired_copy', 'cases': report['paired_copy']}), flush=True)
    finally:
        mon.close()
        del raw, stream
        pool.pool.shutdown()
    report['complete'] = True
    write_json(output, report)
    data.commit()
    return report


@app.function(image=image, gpu='L40S', volumes={'/data': data}, cpu=4.0, memory=32768, timeout=360)
def prepare_own_history(checkpoint_path: str, output: str, count: int = 24, wall_s: float = 280):
    import collections
    import numpy as np
    import torch
    from beatfcn.events.requests import build_request
    from beatfcn.model import features as F
    from beatfcn.train.data import Corpus
    from beatfcn.train.evolution import own_history_core
    started = time.perf_counter()
    model, ck = load_model(checkpoint_path)
    ca = ck['data']
    rows, _, info = training._select_rows(ca['taps_key'], tuple(ca['layers']), ca['pca'], ca['whiten'], '', 0, '',
                                          ck['manifest_name'])
    if info['rows_sha256'] != ck['rows_sha256']:
        raise ValueError('rollout source uses another data selection')
    corpus = Corpus(**ca)
    corpus.train_rows(rows)
    by_mode = {mode: sorted([r for r in rows if r['mode'] == mode and r['audio_sec'] >= 24],
                            key=lambda r: hashlib.sha256(r['chart_id'].encode()).hexdigest())
               for mode in F.MODES}
    rng = np.random.default_rng(20261024)
    cores, failures, seen = [], collections.Counter(), set()
    attempt = 0
    while len(cores) < count and time.perf_counter() - started < wall_s:
        mode = F.MODES[attempt % 2]
        row = by_mode[mode][int(rng.integers(len(by_mode[mode])))]
        attempt += 1
        if row['audio_sha256'] in seen:
            continue
        seen.add(row['audio_sha256'])
        seed = 20261024 + attempt
        try:
            chart = corpus.chart(row['chart_id'])
            duration = round(row['audio_sec'] * 1e6)
            sample = build_request(chart, kind='from_scratch', seed=seed, audio_id=row['audio_sha256'],
                                   audio_duration_us=duration, control_drop_all_p=.5)
            c0 = int(rng.integers(0, max(1, duration // F.CORE_US - 1))) * F.CORE_US
            song = corpus.song(row['audio_sha256'], in_memory=True)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                ex = own_history_core(model, sample, chart.events, song, c0, seed=seed, device='cuda',
                                      deadline=min(started + wall_s, time.perf_counter() + 15))
            ex.provenance.update(chart_id=row['chart_id'], song=row['audio_sha256'], mode=mode,
                                 source_checkpoint=checkpoint_path, source_step=ck['step'])
            cores.append((ex, row['audio_sha256']))
        except (F.SkipExample, ValueError, AssertionError) as exc:
            failures[f'{type(exc).__name__}:{exc}'] += 1
    report = {'source': checkpoint_path, 'step': ck['step'], 'rows_sha256': ck['rows_sha256'], 'attempts': attempt,
              'accepted': len(cores), 'by_mode': dict(collections.Counter(F.MODES[ex.mode] for ex, _ in cores)),
              'failures': dict(failures), 'function_wall_s': time.perf_counter() - started,
              'protocol': 'one generated 8s core, unchanged human next 8s core; conflicts rejected and counted'}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'cores': cores, 'report': report}, output)
    write_json(output + '.json', report)
    data.commit()
    return report


@app.function(image=image, gpu='L40S', volumes={'/data': data}, cpu=4.0, memory=32768, timeout=1200)
def benchmark_pair(checkpoints: list[str], output: str, wall_s: float = 1080, per_request_s: float = 60):
    """Interleaved matched requests; partial coverage is explicit and cannot promote a candidate."""
    import torch
    from beatfcn.model.evaluation import evaluate_songs
    from beatfcn.model.benchmark import scorecard
    from beatfcn.train.data import Corpus
    from beatfcn.train.evolution import digest
    started = time.perf_counter()
    manifest = json.loads(Path('/bench/capability-v1/smoke8.json').read_text())
    if digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'}) != manifest['manifest_sha256']:
        raise ValueError('frozen benchmark changed')
    models = []
    for path in checkpoints:
        model, ck = load_model(path)
        models.append((path, model, ck, Corpus(**ck['data'])))
    records = {path: [] for path in checkpoints}
    result = {'schema': 'beatfcn.evolution.benchmark.v1', 'manifest_sha256': manifest['manifest_sha256'],
              'checkpoints': checkpoints, 'expected_per_checkpoint': len(manifest['requests']), 'records': records}
    for row in manifest['rows']:
        if time.perf_counter() - started >= wall_s:
            break
        # One raw song load reused by both checkpoints and all request types.
        resident = models[0][3].song(row['audio_sha256'], in_memory=True)
        for kind in manifest['request_kinds']:
            for path, model, ck, corpus in models:
                remaining = started + wall_s - time.perf_counter()
                if remaining <= 2:
                    break
                original = corpus.song
                corpus.song = lambda sha, in_memory=False, resident=resident: resident
                try:
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        evaluated = evaluate_songs(model, corpus, [row], n_songs=1, seed=manifest['seed'], device='cuda',
                                                   greedy=True, per_song_timeout_s=min(per_request_s, remaining),
                                                   request_kinds=(kind,))
                    records[path].extend(evaluated['songs'])
                finally:
                    corpus.song = original
                result['function_wall_s'] = time.perf_counter() - started
                write_json(output, result)
                data.commit()
    result['coverage'] = {}
    for path in checkpoints:
        actual = {(r['chart_id'], r['request_kind']) for r in records[path]}
        for request in manifest['requests']:
            if (request['chart_id'], request['kind']) not in actual:
                records[path].append({'chart_id': request['chart_id'], 'song': request['song'], 'mode': request['mode'],
                                      'request_kind': request['kind'], 'complete': False, 'legality': [],
                                      'failure': {'reason': 'campaign_evaluation_time_limit'}})
        result['coverage'][path] = {'requested': len(manifest['requests']),
                                   'completed': sum(bool(r['complete']) for r in records[path]),
                                   'legal_completed': sum(bool(r['complete']) and not r['legality'] for r in records[path])}
    result['scorecards'] = {path: scorecard(rs) for path, rs in records.items()}
    result['complete_for_review'] = all(v['completed'] == v['requested'] for v in result['coverage'].values())
    result['function_wall_s'] = time.perf_counter() - started
    write_json(output, result)
    data.commit()
    return {k: v for k, v in result.items() if k not in ('records', 'scorecards')}


@app.function(image=image, volumes={'/data': data}, cpu=.25, memory=1024, timeout=4800, max_containers=1)
def execute_round(plan: dict):
    """Run one bounded round, persist every stage, then release all compute for agent review."""
    from beatfcn.train.evolution import reserve_stage
    root = Path('/data/evolution') / CAMPAIGN
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / 'campaign.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        'approved_usd': 10.0, 'stages': {}, 'rounds': {}, 'pricing_source': 'https://modal.com/pricing',
        'initial_inspection_reserve_usd': .05}
    if plan['round'] in ledger['rounds']:
        raise ValueError('round already dispatched; inspect it instead of repeating it')
    ledger['rounds'][plan['round']] = {'plan': plan, 'status': 'preparing', 'started_at': time.time()}
    current = ledger['rounds'][plan['round']]
    write_json(ledger_path, ledger)
    data.commit()

    def dispatch(label, function, resources, kwargs):
        name = plan['round'] + '/' + label
        # Keep orchestrator/read-only inspection allowance outside the compute stages.
        budget = {**ledger, 'approved_usd': ledger['approved_usd'] - .25}
        stage = reserve_stage(budget, name, resources)
        current['status'] = label
        stage.update(status='dispatched', started_at=time.time())
        write_json(ledger_path, ledger)
        data.commit()
        options = dict(cpu=(resources['cpu'], resources['cpu']),
                       memory=(resources['memory_gib'] * 1024, resources['memory_gib'] * 1024),
                       timeout=resources['timeout'], retries=0, max_containers=1, scaledown_window=2)
        if resources.get('gpu'):
            options['gpu'] = resources['gpu']
        call = function.with_options(**options).spawn(**kwargs)
        stage['call_id'] = call.object_id
        write_json(ledger_path, ledger)
        data.commit()
        try:
            stage['result'] = call.get(timeout=resources['timeout'] + 100)
            stage['status'] = 'complete'
        except BaseException as exc:
            call.cancel(terminate_containers=True)
            stage.update(status='failed', error=f'{type(exc).__name__}:{exc}')
            raise
        finally:
            stage['dispatch_wall_s'] = time.time() - stage['started_at']
            stage['estimated_usd'] = stage['rate_usd_s'] * (stage['dispatch_wall_s'] + 2)
            data.reload()
            write_json(ledger_path, ledger)
            data.commit()
        return stage['result']

    try:
        args = dict(plan['training'])
        if plan.get('metadata_compress'):
            current['diagnostic_result'] = dispatch('metadata_compress', compress_metadata_tape, plan['resources'], plan['metadata_compress'])
            current['status'] = 'prepared'
            return current
        if plan.get('metadata_prepare'):
            current['diagnostic_result'] = dispatch('metadata_prepare', prepare_metadata_tape, plan['resources'], plan['metadata_prepare'])
            current['status'] = 'prepared'
            return current
        if plan.get('live_input_probe'):
            current['diagnostic_result'] = dispatch('live_input_probe', probe_live_input, plan['resources'], plan['live_input_probe'])
            current['status'] = 'diagnosed'
            return current
        if plan.get('transport_probe'):
            current['diagnostic_result'] = dispatch('transport_probe', probe_transport, plan['resources'], plan['transport_probe'])
            current['status'] = 'diagnosed'
            return current
        if plan.get('allocation_probe'):
            current['diagnostic_result'] = dispatch('allocation_probe', probe_staged_allocation, plan['resources'], plan['allocation_probe'])
            current['status'] = 'diagnosed'
            return current
        if plan.get('gradient_replay'):
            current['diagnostic_result'] = dispatch('gradient_replay', replay_gradients, plan['resources'], plan['gradient_replay'])
            current['status'] = 'diagnosed'
            return current
        if plan.get('diagnostic_only'):
            current['diagnostic_result'] = dispatch('diagnostic', training.train, plan['resources'], args)
            current['status'] = 'diagnosed'
            return current
        if args.get('prep') and args.get('rollout_cache'):
            prep, replay_path = {'key': args['prep']}, args['rollout_cache']
            replay = json.loads(Path(replay_path + '.json').read_text())
            if replay['source'] != plan.get('rollout_source', args['initialize_from']):
                raise ValueError('cached own-history source differs from warm-start checkpoint')
        else:
            prep = dispatch('prepare', training.prep_train, dict(cpu=4, memory_gib=16, timeout=240),
                            dict(taps_key=args['taps_key'], taps=tuple(args['taps']), pca=args['pca'],
                                 val_cores=args['val_cores'], val_songs=args['val_songs'], audit=False,
                                 manifest_name=args['manifest_name']))
            replay_path = str(root / plan['round'] / 'own_history.pt')
            replay = dispatch('own_history', prepare_own_history, dict(gpu='L40S', cpu=4, memory_gib=32, timeout=360),
                              dict(checkpoint_path=args['initialize_from'], output=replay_path))
        if replay['accepted'] < 8 or len(replay['by_mode']) < 2:
            raise RuntimeError('insufficient valid own-history samples; inspect rejected cases before training')
        if plan.get('validate_staged_allocation'):
            checked = dispatch('allocation_validation', probe_staged_allocation,
                               dict(gpu='L40S', cpu=4, memory_gib=16, timeout=120),
                               dict(output=str(root / plan['round'] / 'allocation_validation.json'),
                                    checkpoint_path=args['initialize_from']))
            if not all(case['matches_files'] for case in checked['cases']):
                raise RuntimeError('staged allocation regression failed; training not started')
        args.update(prep=prep['key'], rollout_cache=replay_path)
        result = dispatch('train', training.train, plan['resources'], args)
        current['training_result'] = result
        if result['aborted'] or result.get('round_effective_updates', result['effective_updates']) != 1000:
            current['status'] = 'needs_diagnosis'
        else:
            comparison = dispatch('benchmark', benchmark_pair, dict(gpu='L40S', cpu=4, memory_gib=32, timeout=1200),
                                  dict(checkpoints=[plan.get('benchmark_baseline', args['initialize_from']), f"/data/runs/{args['run_id']}/last.pt"],
                                       output=str(root / plan['round'] / 'benchmark.json')))
            current['benchmark_result'] = comparison
            current['status'] = 'awaiting_review'
    except BaseException as exc:
        current.update(status='needs_diagnosis', error=f'{type(exc).__name__}:{exc}')
        raise
    finally:
        current['finished_at'] = time.time()
        write_json(ledger_path, ledger)
        data.commit()
    return current


@app.local_entrypoint()
def run(plan_file: str):
    plan = json.loads(Path(plan_file).read_text())
    if not any(plan.get(key) for key in ('diagnostic_only', 'gradient_replay', 'allocation_probe', 'transport_probe', 'live_input_probe', 'metadata_prepare', 'metadata_compress')):
        write_json(training.REPO / 'reports' / 'training_monitor_target.json',
                   {'run_id': plan['training']['run_id'], 'round': plan['round'], 'plan_file': plan_file})
    print(json.dumps(execute_round.remote(plan), indent=2))


@app.function(image=image, volumes={'/data': data}, cpu=2.0, memory=16384, timeout=300)
def inspect_checkpoint(run_id: str = 's1-v21'):
    import torch
    result = {'run_id': run_id, 'checkpoints': [], 'checked_at': time.time()}
    for name in ('last.pt', 'best.pt'):
        path = Path('/data/runs') / run_id / name
        if not path.exists():
            continue
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(8 << 20), b''):
                digest.update(block)
        ck = torch.load(path, map_location='cpu', weights_only=False)
        bad = [k for k, t in ck['model'].items() if torch.is_tensor(t) and t.is_floating_point()
               and not bool(torch.isfinite(t).all())]
        bad_opt = []
        for key, state in ck.get('opt', {}).get('state', {}).items():
            for field, value in state.items():
                if torch.is_tensor(value) and value.is_floating_point() and not bool(torch.isfinite(value).all()):
                    bad_opt.append(f'{key}:{field}')
        result['checkpoints'].append({'path': str(path), 'step': ck['step'], 'sha256': digest.hexdigest(),
                                     'bytes': path.stat().st_size, 'nonfinite_model': bad, 'nonfinite_optimizer': bad_opt,
                                     'config': ck['config'], 'data': ck['data'], 'rows_sha256': ck.get('rows_sha256'),
                                     'manifest_name': ck.get('manifest_name'), 'val': ck.get('val'), 'best': ck.get('best')})
    return result


@app.local_entrypoint()
def inspect(run_id: str = 's1-v21'):
    result = inspect_checkpoint.remote(run_id)
    output = training.REPO / 'reports' / 'evolution_checkpoint_inspection.json'
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
