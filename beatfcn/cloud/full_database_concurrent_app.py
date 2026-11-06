"""Bounded concurrent MERT pipeline for the v2 throughput pilot.

The model, BF16 recipe, output schema and immutable feature receipts are unchanged.
Only decode/prefetch, CUDA-stream and writer scheduling changes. No local entrypoint.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from pathlib import Path
import json
import threading
import time

from .full_database_app import (
    CHECKPOINT, CPU_WORKERS, DATA, REMOTE_ROOT, StableGpuMonitor, _prefetch,
    _write_feature, app, data, encoder, image, verify_beatfcn_sources,
    verify_feature_folder)


GPU_STREAMS = 2
WRITE_WORKERS = 4
IN_FLIGHT = 8
COMMIT_EVERY = 16
_gpu_local = threading.local()


def _encode_on_stream(prefetched, model):
    """Run the unchanged single-song MERT recipe on an independent CUDA stream."""
    import torch
    from beatfcn.train.full_database import feature_payloads
    if not hasattr(_gpu_local, 'stream'):
        _gpu_local.stream = torch.cuda.Stream()
    with torch.cuda.stream(_gpu_local.stream):
        features, _, expected = feature_payloads(prefetched['wave'], model,
            mel=prefetched['mel'])
        _gpu_local.stream.synchronize()
    return prefetched, features, expected


@app.function(image=image, volumes={DATA: data}, gpu='H100', cpu=(8, 8),
              memory=(32768, 32768), timeout=3600, retries=0,
              max_containers=1, scaledown_window=120)
def encode_concurrent(run_id, contract_json, dispatch_deadline_unix):
    """Keep bounded work in flight across decode, two GPU streams and writers."""
    from beatfcn.audio.mert2 import validate_local_source
    contract = json.loads(contract_json)
    verify_beatfcn_sources(contract)
    validate_local_source(CHECKPOINT)
    output_root = contract['output_root']
    if not output_root.startswith(REMOTE_ROOT + '/'):
        raise ValueError('Output root must stay inside full-database-v1')
    jobs = contract['jobs']
    if not jobs or len({job['audio_sha256'] for job in jobs}) != len(jobs):
        raise ValueError('Jobs must be nonempty and unique by audio hash')
    pipeline = contract.get('pipeline', {})
    expected_pipeline = dict(prefetch_workers=8, gpu_streams=2,
        write_workers=4, in_flight=8, commit_every=16, independent_models=2)
    if pipeline != expected_pipeline:
        raise ValueError('Frozen concurrent pipeline envelope changed')
    model = encoder()
    if model.spec.layers != (12, 23):
        raise ValueError('Full database requires frozen MERT layers 12/23')
    # The pinned rotary cache is mutable: each stream gets independent model/cache state.
    from beatfcn.audio.mert2 import FrozenMert2
    models = [model, FrozenMert2(device='cuda', local_source=CHECKPOINT)]
    if len(models) != GPU_STREAMS or any(value.spec.layers != (12, 23) for value in models):
        raise ValueError('Independent MERT model instances do not match the frozen layers')
    output_root_path = Path(DATA) / output_root
    temporary_root = Path('/tmp') / 'full-database-audio-v2'
    temporary_root.mkdir(parents=True, exist_ok=True)
    records = []
    failures = []
    stopped = False
    monitor = StableGpuMonitor()
    started = time.monotonic()
    with ExitStack() as stack:
        prefetch_pool = stack.enter_context(ThreadPoolExecutor(
            max_workers=CPU_WORKERS, thread_name_prefix='mert-prefetch'))
        # Each single-thread executor owns exactly one independent model instance.
        gpu_pools = [stack.enter_context(ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f'mert-gpu-{index}'))
            for index in range(GPU_STREAMS)]
        writer_pool = stack.enter_context(ThreadPoolExecutor(
            max_workers=WRITE_WORKERS, thread_name_prefix='mert-write'))
        def existing(job):
            folder = output_root_path / 'features' / job['audio_sha256']
            if not (folder / 'receipt.json').exists():
                return job, None
            return job, verify_feature_folder(folder, job, model.revision)

        ready = []
        # Any resume verification is cold setup and is excluded from stable GPU sampling.
        for job, prior in prefetch_pool.map(existing, jobs):
            if prior is None:
                ready.append(job)
            else:
                records.append(dict(audio_sha256=job['audio_sha256'], complete=True,
                    wall_s=0., files=prior['files'], existing=True))
        if not ready:
            return dict(run_id=run_id, output_root=output_root,
                attempted=len(records), planned=len(jobs), successes=records, failures=[],
                stopped=False, wall_s=time.monotonic() - started,
                recovery_verified=[record['audio_sha256'] for record in records],
                gpu_utilization=monitor.summary(applicable=False), all_existing=True)
        monitor.start()
        next_index = 0
        new_audio_seconds = 0.
        new_commits = 0
        stop_submitting = False
        prefetch_futures = {}
        gpu_futures = {}
        writer_futures = {}
        gpu_round_robin = 0
        try:
            while next_index < len(ready) or prefetch_futures or gpu_futures or writer_futures:
                if time.time() >= dispatch_deadline_unix or monitor.stop_reason:
                    stopped = True
                    stop_submitting = True
                outstanding = len(prefetch_futures) + len(gpu_futures) + len(writer_futures)
                while (not stop_submitting and next_index < len(ready)
                        and outstanding < IN_FLIGHT):
                    job = ready[next_index]
                    future = prefetch_pool.submit(_prefetch, job, output_root_path, temporary_root)
                    prefetch_futures[future] = job
                    next_index += 1
                    outstanding += 1
                pending = set(prefetch_futures) | set(gpu_futures) | set(writer_futures)
                if not pending:
                    continue
                done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    if future in prefetch_futures:
                        job = prefetch_futures.pop(future)
                        try:
                            prefetched = future.result()
                            slot = gpu_round_robin % GPU_STREAMS
                            gpu_round_robin += 1
                            gpu_future = gpu_pools[slot].submit(_encode_on_stream,
                                prefetched, models[slot])
                            gpu_futures[gpu_future] = job
                        except Exception as exc:
                            failures.append(dict(audio_sha256=job['audio_sha256'],
                                stage='prefetch', error=f'{type(exc).__name__}: {exc}'))
                            stopped = stop_submitting = True
                    elif future in gpu_futures:
                        job = gpu_futures.pop(future)
                        try:
                            prefetched, features, expected = future.result()
                            writer_future = writer_pool.submit(_write_feature, prefetched,
                                output_root_path, temporary_root, features, expected)
                            writer_futures[writer_future] = job
                        except Exception as exc:
                            failures.append(dict(audio_sha256=job['audio_sha256'],
                                stage='gpu', error=f'{type(exc).__name__}: {exc}'))
                            stopped = stop_submitting = True
                    else:
                        job = writer_futures.pop(future)
                        try:
                            result = future.result()
                            records.append(dict(audio_sha256=job['audio_sha256'],
                                complete=result['complete'], wall_s=result['wall_s'],
                                files=result['files'], existing=False))
                            new_audio_seconds += float(job['audio_sec'])
                            new_commits += 1
                        except Exception as exc:
                            failures.append(dict(audio_sha256=job['audio_sha256'],
                                stage='write', error=f'{type(exc).__name__}: {exc}'))
                            stopped = stop_submitting = True
                        if new_commits and (new_commits % COMMIT_EVERY == 0
                                or (next_index == len(ready) and not prefetch_futures
                                    and not gpu_futures and not writer_futures)):
                            data.commit()
                        if new_commits and new_commits % COMMIT_EVERY == 0:
                            print(dict(stage='full-database-mert-v2', completed=new_commits,
                                planned=len(ready), failures=len(failures),
                                in_flight=len(prefetch_futures) + len(gpu_futures) + len(writer_futures),
                                wall_s=time.monotonic() - started,
                                gpu_monitor=monitor.summary()), flush=True)
            data.commit()
        finally:
            monitor.close()
        utilization = monitor.summary()
        # Recovery is remote-only and excluded from the stable GPU pipeline window.
        data.reload()
        by_id = {job['audio_sha256']: job for job in jobs}
        recovery = list(prefetch_pool.map(lambda record: verify_feature_folder(
            output_root_path / 'features' / record['audio_sha256'],
            by_id[record['audio_sha256']], model.revision), records))
    stable_s = utilization.get('stable_observed_s') or 0.
    throughput = dict(new_audio_seconds=new_audio_seconds,
        stable_pipeline_seconds=stable_s,
        audio_seconds_per_second=new_audio_seconds / stable_s if stable_s > 0 else None,
        new_jobs_per_second=new_commits / stable_s if stable_s > 0 else None,
        prefetch_workers=CPU_WORKERS, gpu_streams=GPU_STREAMS,
        independent_model_instances=len(models),
        write_workers=WRITE_WORKERS, in_flight=IN_FLIGHT)
    return dict(run_id=run_id, output_root=output_root,
        attempted=len(records) + len(failures), planned=len(jobs),
        successes=records, failures=failures,
        stopped=stopped or bool(utilization['rolling_stop_reason']),
        recovery_verified=[record['audio_sha256'] for record in recovery],
        throughput=throughput, wall_s=time.monotonic() - started,
        gpu_utilization=utilization, all_existing=False)
