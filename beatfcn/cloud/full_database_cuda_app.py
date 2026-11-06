"""Deep CUDA diagnostic: direct verified raw IO, CUDA mel and single-pass writes."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from pathlib import Path
import json
import os
import threading
import time

import numpy as np

from .full_database_app import (
    CHECKPOINT, CPU_WORKERS, DATA, REMOTE_ROOT, StableGpuMonitor, app, data,
    encoder, image, verify_beatfcn_sources, verify_feature_folder)
from .full_database_concurrent_app import GPU_STREAMS, WRITE_WORKERS, IN_FLIGHT, COMMIT_EVERY


EXPECTED_PIPELINE = dict(prefetch_workers=8, gpu_streams=2, independent_models=2,
    write_workers=4, in_flight=8, commit_every=16,
    direct_verified_raw=True, cuda_mel=True, single_pass_write_hash=True)
_gpu_local = threading.local()


def _decode(job, raw_index, temporary_root):
    from beatfcn.train.cuda_pipeline import decoded_audio_from_receipt
    started = time.monotonic()
    audio, original, wave = decoded_audio_from_receipt(job, raw_index, DATA, temporary_root)
    return dict(job=job, audio=audio, original=original, wave=wave,
        decode_s=time.monotonic() - started)


def _encode_cuda(prefetched, model):
    import torch
    from beatfcn.audio.cuda_mel import complete_song_mel_cuda
    from beatfcn.train.full_database import feature_payloads
    if not hasattr(_gpu_local, 'stream'):
        _gpu_local.stream = torch.cuda.Stream()
    with torch.cuda.stream(_gpu_local.stream):
        started = time.monotonic()
        mel = complete_song_mel_cuda(prefetched['wave'])
        _gpu_local.stream.synchronize()
        mel_s = time.monotonic() - started
        started = time.monotonic()
        features, _, expected = feature_payloads(prefetched['wave'], model, mel=mel)
        _gpu_local.stream.synchronize()
        mert_s = time.monotonic() - started
        memory = int(torch.cuda.max_memory_allocated())
    return prefetched, features, expected, mel, mel_s, mert_s, memory


def _write_fast(prefetched, output_root_path, temporary_root, features, expected):
    from beatfcn.train.cuda_pipeline import commit_feature_hashing
    started = time.monotonic()
    result = commit_feature_hashing(prefetched['audio'], Path(prefetched['original']),
        output_root_path, temporary_root, features, prefetched['mel'], expected)
    result['wall_s'] = time.monotonic() - started
    return result


def _parity_preflight(jobs, raw_index, temporary_root):
    from beatfcn.audio.cuda_mel import complete_song_mel_cuda, mel_parity
    from beatfcn.audio.local_mel_cache import complete_song_mel
    reports = []
    for index in sorted({0, len(jobs) // 2, len(jobs) - 1}):
        prefetched = _decode(jobs[index], raw_index, temporary_root)
        cuda = complete_song_mel_cuda(prefetched['wave'])
        cpu = complete_song_mel(prefetched['wave'])
        report = mel_parity(cpu['mel'], cuda['mel'])
        report['audio_sha256'] = jobs[index]['audio_sha256']
        report['passed'] = bool(report['finite'] and report['max_abs'] <= .01
            and report['float16_exact_fraction'] >= .99)
        reports.append(report)
        prefetched['mel'] = cuda
        original = Path(prefetched['original'])
        if original.is_relative_to(Path(temporary_root)):
            original.unlink()
    if not all(report['passed'] for report in reports):
        raise ValueError('CUDA/CPU mel parity gate failed: ' + json.dumps(reports))
    return reports


@app.function(image=image, volumes={DATA: data}, gpu='H100', cpu=(8, 8),
              memory=(32768, 32768), timeout=900, retries=0,
              max_containers=1, scaledown_window=30)
def encode_cuda_profile(run_id, contract_json, dispatch_deadline_unix):
    from beatfcn.audio.mert2 import FrozenMert2, validate_local_source
    from beatfcn.train.cuda_pipeline import load_raw_index
    contract = json.loads(contract_json)
    verify_beatfcn_sources(contract)
    if contract.get('pipeline') != EXPECTED_PIPELINE or not contract.get('diagnostic_only'):
        raise ValueError('CUDA diagnostic pipeline identity changed')
    validate_local_source(CHECKPOINT)
    output_root = contract['output_root']
    if not output_root.startswith(REMOTE_ROOT + '/'):
        raise ValueError('Output root must stay inside full-database-v1')
    jobs = contract['jobs']
    if not jobs or len({job['audio_sha256'] for job in jobs}) != len(jobs):
        raise ValueError('Jobs must be nonempty and unique')
    raw_index = load_raw_index(DATA, contract['raw_content_sha256'],
        contract['raw_manifest_sha256'])
    output_root_path = Path(DATA) / output_root
    temporary_root = Path('/tmp') / 'full-database-audio-cuda-profile'
    temporary_root.mkdir(parents=True, exist_ok=True)
    parity = _parity_preflight(jobs, raw_index, temporary_root)
    model = encoder()
    models = [model, FrozenMert2(device='cuda', local_source=CHECKPOINT)]
    if any(value.spec.layers != (12, 23) for value in models):
        raise ValueError('MERT layer identity changed')
    records = []
    failures = []
    stopped = False
    monitor = StableGpuMonitor()
    started = time.monotonic()
    cpu_start = os.times()
    stage = dict(decode_s=0., mel_s=0., mert_s=0., write_s=0.,
        cuda_memory_allocated_max=0)
    with ExitStack() as stack:
        prefetch_pool = stack.enter_context(ThreadPoolExecutor(
            max_workers=CPU_WORKERS, thread_name_prefix='cuda-decode'))
        gpu_pools = [stack.enter_context(ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f'cuda-mert-{index}'))
            for index in range(GPU_STREAMS)]
        writer_pool = stack.enter_context(ThreadPoolExecutor(
            max_workers=WRITE_WORKERS, thread_name_prefix='cuda-write'))
        ready = list(jobs)
        import torch
        torch.cuda.reset_peak_memory_stats()
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
                while not stop_submitting and next_index < len(ready) and outstanding < IN_FLIGHT:
                    job = ready[next_index]
                    future = prefetch_pool.submit(_decode, job, raw_index, temporary_root)
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
                            stage['decode_s'] += prefetched['decode_s']
                            slot = gpu_round_robin % GPU_STREAMS
                            gpu_round_robin += 1
                            gpu_future = gpu_pools[slot].submit(_encode_cuda,
                                prefetched, models[slot])
                            gpu_futures[gpu_future] = job
                        except Exception as exc:
                            failures.append(dict(audio_sha256=job['audio_sha256'],
                                stage='decode', error=f'{type(exc).__name__}: {exc}'))
                            stopped = stop_submitting = True
                    elif future in gpu_futures:
                        job = gpu_futures.pop(future)
                        try:
                            prefetched, features, expected, mel, mel_s, mert_s, memory = future.result()
                            stage['mel_s'] += mel_s
                            stage['mert_s'] += mert_s
                            stage['cuda_memory_allocated_max'] = max(
                                stage['cuda_memory_allocated_max'], memory)
                            prefetched['mel'] = mel
                            writer_future = writer_pool.submit(_write_fast, prefetched,
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
                            stage['write_s'] += result['wall_s']
                            records.append(dict(audio_sha256=job['audio_sha256'],
                                complete=result['complete'], wall_s=result['wall_s'],
                                files=result['files'], existing=False))
                            new_audio_seconds += float(job['audio_sec'])
                            new_commits += 1
                        except Exception as exc:
                            failures.append(dict(audio_sha256=job['audio_sha256'],
                                stage='write', error=f'{type(exc).__name__}: {exc}'))
                            stopped = stop_submitting = True
                        if new_commits and new_commits % COMMIT_EVERY == 0:
                            data.commit()
                            print(dict(stage='cuda-profile', completed=new_commits,
                                planned=len(ready), failures=len(failures),
                                wall_s=time.monotonic() - started,
                                gpu_monitor=monitor.summary(), stage_totals=stage), flush=True)
            data.commit()
        finally:
            monitor.close()
        utilization = monitor.summary()
        cpu_end = os.times()
        stable_s = utilization.get('stable_observed_s') or 0.
        cpu_seconds = ((cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
            + (cpu_end.children_user - cpu_start.children_user)
            + (cpu_end.children_system - cpu_start.children_system))
        profile = dict(stage_totals=stage,
            process_cpu_seconds=cpu_seconds,
            estimated_cpu_percent_of_eight_cores=100 * cpu_seconds / (8 * stable_s)
                if stable_s > 0 else None,
            cuda_mel_parity=parity, raw_repeated_sha_skipped=True,
            write_reread_for_sha_skipped=True)
        data.reload()
        by_id = {job['audio_sha256']: job for job in jobs}
        recovery = list(prefetch_pool.map(lambda record: verify_feature_folder(
            output_root_path / 'features' / record['audio_sha256'],
            by_id[record['audio_sha256']], model.revision), records))
    throughput = dict(new_audio_seconds=new_audio_seconds,
        stable_pipeline_seconds=stable_s,
        audio_seconds_per_second=new_audio_seconds / stable_s if stable_s > 0 else None,
        new_jobs_per_second=new_commits / stable_s if stable_s > 0 else None)
    return dict(run_id=run_id, output_root=output_root, diagnostic_only=True,
        attempted=len(records) + len(failures), planned=len(jobs),
        successes=records, failures=failures,
        stopped=stopped or bool(utilization['rolling_stop_reason']),
        recovery_verified=[record['audio_sha256'] for record in recovery],
        throughput=throughput, profile=profile, wall_s=time.monotonic() - started,
        gpu_utilization=utilization, all_existing=False)
