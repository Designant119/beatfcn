"""Formal full-database multi-stream eager MERT pipeline.

The model, retained layers, clocks and immutable receipts are unchanged from the
pilot path. Batched MERT is permanently closed; per-shape CUDA graphs are closed
for the remaining corpus shape distribution. This pipeline runs the reference
per-song eager forward on concurrent independent CUDA streams, proven bitwise
identical to serial per-song forward (mert-eager-streams-probe-v1). Decode,
GPU, write, commit and every inter-job gap below are inside the official
stable-period monitor. Existing receipts are fully re-verified before reuse;
invalid ones are quarantined, never overwritten.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
import json
import os
import queue
import random
import shutil
import threading
import time

import numpy as np
import torch

from .full_database_app import (
    CHECKPOINT, DATA, REMOTE_ROOT, StableGpuMonitor, app, data,
    encoder, image, verify_beatfcn_sources, verify_feature_folder)

GPU_STREAMS = 2
DECODE_WORKERS = 28
DECODE_QUEUE = 12
WRITE_WORKERS = 8
WRITE_QUEUE_LIMIT = 24
COMMIT_EVERY = 16
RECOVERY_SMOKE = 8
EXPECTED_PIPELINE = dict(path='mert_per_song_eager_streams_v1', gpu_streams=2,
    decode_workers=28, decode_queue=12, write_workers=8, write_queue_limit=24,
    commit_every=16, verified_raw_receipt_decode=True, cpu_mel=True,
    resume='local_audit_contract_v1', quarantine_invalid=True, recovery_smoke=8,
    instrumented='unconditional_time_commit_v2')


def _decode(job, raw_index, temporary_root):
    from beatfcn.audio.local_mel_cache import complete_song_mel
    from beatfcn.audio.mert2 import CELL_SAMPLES
    from beatfcn.train.cuda_pipeline import decoded_audio_from_receipt
    started = time.monotonic()
    audio, original, wave = decoded_audio_from_receipt(job, raw_index, DATA, temporary_root)
    # The legacy mel10 receipt is the authoritative decoded-sample clock.
    if 4 * (len(wave) // CELL_SAMPLES) != job['expected_mel_frames']:
        raise ValueError('Decoded sample clock differs from authoritative mel receipt: ' + audio)
    mel = complete_song_mel(wave)
    return dict(job=job, audio=audio, original=original, wave=wave, mel=mel,
        decode_s=time.monotonic() - started)


def _write(prefetched, features, mel, expected, output_root_path, temporary_root):
    from beatfcn.train.cuda_pipeline import commit_feature_hashing
    started = time.monotonic()
    result = commit_feature_hashing(prefetched['audio'], Path(prefetched['original']),
        output_root_path, temporary_root, features, mel, expected)
    result['wall_s'] = time.monotonic() - started
    return result


@app.function(image=image, volumes={DATA: data}, gpu='H100', cpu=(16, 16),
              memory=(49152, 49152), timeout=2050, retries=0,
              max_containers=1, scaledown_window=120)
def encode_mert_eager(run_id, contract_json, dispatch_deadline_unix):
    from beatfcn.audio.mert2 import validate_local_source
    from beatfcn.train.cuda_pipeline import load_raw_index
    from beatfcn.train.full_database import feature_payloads
    contract = json.loads(contract_json)
    verify_beatfcn_sources(contract)
    if contract.get('pipeline') != EXPECTED_PIPELINE:
        raise ValueError('Formal eager pipeline identity changed')
    if contract.get('full_corpus_encode') is not True:
        raise ValueError('Formal run must be the authorized full-corpus encode')
    validate_local_source(CHECKPOINT)
    if not contract['output_root'].startswith(REMOTE_ROOT + '/'):
        raise ValueError('Output root must stay inside full-database-v1')
    jobs = contract['jobs']
    if not jobs or len({job['audio_sha256'] for job in jobs}) != len(jobs):
        raise ValueError('Jobs must be nonempty and unique')
    raw_index = load_raw_index(DATA, contract['raw_content_sha256'],
        contract['raw_manifest_sha256'])
    output_root_path = Path(DATA) / contract['output_root']
    quarantine_root = output_root_path / 'quarantine' / run_id
    temporary_root = Path('/tmp') / 'mert-eager-formal'
    temporary_root.mkdir(parents=True, exist_ok=True)
    model = encoder()
    if model.spec.layers != (12, 23):
        raise ValueError('MERT retained layers changed')

    streams = [torch.cuda.Stream() for _ in range(GPU_STREAMS)]
    # Warm rotary cache and per-stream state to the longest window before any
    # concurrent forward; this is cold setup, excluded from the monitor.
    for stream in streams:
        with torch.cuda.stream(stream):
            zeros = np.zeros(300 * 24000, dtype=np.float32)
            feature_payloads(zeros, model)
    torch.cuda.synchronize()

    monitor = StableGpuMonitor()
    started = time.monotonic()
    cpu_start = os.times()
    stage = dict(decode_s=0., decode_osu_s=0., decode_phira_s=0., encode_s=0.,
        write_s=0., decode_busy=0, writes_high_water=0,
        cuda_memory_allocated_max=0)
    records = []
    failures = []
    quarantined = []
    ineligible = []
    stopped = False
    lock = threading.Lock()
    decoded_count = [0]
    encoded_count = [0]
    completed_count = [0]
    pending_writes = [0]
    write_slots = threading.Semaphore(WRITE_QUEUE_LIMIT)
    with ExitStack() as stack:
        pool = stack.enter_context(ThreadPoolExecutor(
            max_workers=DECODE_WORKERS + WRITE_WORKERS + GPU_STREAMS + 16,
            thread_name_prefix='eager-formal'))
        by_id = {job['audio_sha256']: job for job in jobs}

        # Resume comes from the locally frozen audit contract: no in-container
        # FUSE metadata sweeps (they serialize and stalled formal-v2/v3).
        resume = contract.get('resume') or {}
        skip_ids = set(resume.get('skip_ids') or [])
        invalid_ids = set(resume.get('invalid_ids') or [])
        job_ids = set(by_id)
        if (not skip_ids or skip_ids & invalid_ids
                or not skip_ids <= job_ids or not invalid_ids <= job_ids):
            raise ValueError('Resume audit contract identity changed')
        print(dict(stage='resume-contract', skip=len(skip_ids),
            quarantine=len(invalid_ids)), flush=True)
        quarantine_root.mkdir(parents=True, exist_ok=True)
        for audio in sorted(invalid_ids):
            folder = output_root_path / 'features' / audio
            if folder.exists():
                shutil.move(str(folder), str(quarantine_root / audio))
                quarantined.append(dict(audio_sha256=audio,
                    reason='local audit identity failure'))
        ready = []
        for job in jobs:
            audio = job['audio_sha256']
            if audio in skip_ids:
                records.append(dict(audio_sha256=audio, existing=True))
                continue
            record = raw_index.get(audio)
            if not record or not record.get('verified'):
                ineligible.append(dict(audio_sha256=audio,
                    reason='raw content lacks a verified receipt'))
            else:
                ready.append(job)
        print(dict(stage='resume-done', reused=len(records), quarantined=len(quarantined),
            ineligible=len(ineligible), to_compute=len(ready)), flush=True)

        monitor.start()
        torch.cuda.reset_peak_memory_stats()
        decoded_queue = queue.Queue(maxsize=DECODE_QUEUE)
        stop_work = threading.Event()
        sentinels_sent = threading.Event()
        job_lock = threading.Lock()
        ready_index = [0]

        def decode_worker():
            while not stop_work.is_set():
                with job_lock:
                    if ready_index[0] >= len(ready):
                        return
                    index = ready_index[0]
                    ready_index[0] += 1
                job = ready[index]
                try:
                    decoded = _decode(job, raw_index, temporary_root)
                except Exception as exc:
                    with lock:
                        failures.append(dict(audio_sha256=job['audio_sha256'],
                            stage='decode', error=f'{type(exc).__name__}: {exc}'))
                    stop_work.set()
                    return
                with lock:
                    stage['decode_s'] += decoded['decode_s']
                    if decoded['job']['source'] == 'osu':
                        stage['decode_osu_s'] += decoded['decode_s']
                    else:
                        stage['decode_phira_s'] += decoded['decode_s']
                    decoded_count[0] += 1
                    decoded_n = decoded_count[0]
                if decoded_n == 1:
                    print(dict(stage='probe', first_decode_done=True,
                        wall_s=round(time.monotonic() - started, 1)), flush=True)
                decoded_queue.put(decoded)

        def gpu_worker(slot):
            while True:
                item = decoded_queue.get()
                if item is None:
                    decoded_queue.task_done()
                    return
                try:
                    if not stop_work.is_set():
                        with torch.cuda.stream(streams[slot % GPU_STREAMS]):
                            encode_started = time.monotonic()
                            features, mel, expected = feature_payloads(
                                item['wave'], model, mel=item['mel'])
                            with lock:
                                stage['encode_s'] += time.monotonic() - encode_started
                                encoded_count[0] += 1
                                encoded = encoded_count[0]
                                stage['cuda_memory_allocated_max'] = max(
                                    stage['cuda_memory_allocated_max'],
                                    int(torch.cuda.max_memory_allocated()))
                        if encoded == 1:
                            print(dict(stage='probe', first_encode_done=True,
                                wall_s=round(time.monotonic() - started, 1)), flush=True)
                        write_slots.acquire()
                        with lock:
                            pending_writes[0] += 1
                            stage['writes_high_water'] = max(stage['writes_high_water'],
                                pending_writes[0])
                        pool.submit(_handle_write, item, features, mel, expected,
                            output_root_path, temporary_root)
                except Exception as exc:
                    with lock:
                        failures.append(dict(audio_sha256=item['job']['audio_sha256'],
                            stage='gpu', error=f'{type(exc).__name__}: {exc}'))
                    stop_work.set()
                finally:
                    decoded_queue.task_done()

        def _handle_write(value, features, mel, expected, output_root, temporary):
            try:
                result = _write(value, features, mel, expected, output_root, temporary)
                with lock:
                    stage['write_s'] += result['wall_s']
                    records.append(dict(audio_sha256=value['audio_sha256'],
                        existing=False, files=result['files']))
                    completed_count[0] += 1
                    pending_writes[0] -= 1
            except Exception as exc:
                with lock:
                    failures.append(dict(audio_sha256=value['audio_sha256'],
                        stage='write', error=f'{type(exc).__name__}: {exc}'))
                    pending_writes[0] -= 1
                stop_work.set()
            finally:
                write_slots.release()

        decode_futures = [pool.submit(decode_worker) for _ in range(DECODE_WORKERS)]
        gpu_futures = [pool.submit(gpu_worker, slot) for slot in range(GPU_STREAMS)]
        last_commit = [time.monotonic()]
        last_print = [time.monotonic()]
        while True:
            now = time.monotonic()
            if time.time() >= dispatch_deadline_unix or monitor.stop_reason:
                stopped = True
                stop_work.set()
                with job_lock:
                    ready_index[0] = len(ready)
            if all(future.done() for future in decode_futures) \
                    and not sentinels_sent.is_set():
                sentinels_sent.set()
                for _ in range(GPU_STREAMS):
                    decoded_queue.put(None)
            # Unconditional time-based Volume commit: Modal buffers unmounted
            # writes until commit() flushes; gating commit on progress deadlocks
            # because zero completed writes then means zero commits and the buffer
            # fills, blocking every write forever.
            if now - last_commit[0] >= 15:
                data.commit()
                last_commit[0] = now
            if now - last_print[0] >= 12:
                last_print[0] = now
                print(dict(stage='mert-eager', completed=completed_count[0],
                    planned=len(ready), decoded=decoded_count[0],
                    encoded=encoded_count[0], pending_writes=pending_writes[0],
                    failures=len(failures), wall_s=round(now - started, 1)), flush=True)
            if sentinels_sent.is_set() and all(future.done() for future in gpu_futures) \
                    and pending_writes[0] <= 0:
                break
            time.sleep(0.05)
        for future in (*decode_futures, *gpu_futures):
            future.result()
        data.commit()
        monitor.close()
        # Recovery smoke must run while the executor is still alive.
        data.reload()
        fresh_ids = [record['audio_sha256'] for record in records
            if not record.get('existing')]
        existing_ids = [record['audio_sha256'] for record in records
            if record.get('existing')]
        rng_smoke = random.Random(run_id + '/recovery')
        smoke_fresh = rng_smoke.sample(fresh_ids,
            min(RECOVERY_SMOKE, len(fresh_ids))) if fresh_ids else []
        smoke_existing = rng_smoke.sample(existing_ids,
            min(RECOVERY_SMOKE, len(existing_ids))) if existing_ids else []

        def receipt_identity(audio):
            folder = output_root_path / 'features' / audio
            try:
                verify_feature_folder(folder, by_id[audio], model.revision)
                return audio
            except Exception as exc:
                with lock:
                    failures.append(dict(audio_sha256=audio, stage='recovery_smoke',
                        error=f'{type(exc).__name__}: {exc}'))
                return None

        smoke_verified = [audio for audio in pool.map(receipt_identity,
            smoke_fresh + smoke_existing) if audio]
    utilization = monitor.summary()
    cpu_end = os.times()
    stable_s = utilization.get('stable_observed_s') or 0.
    cpu_seconds = ((cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
        + (cpu_end.children_user - cpu_start.children_user)
        + (cpu_end.children_system - cpu_start.children_system))
    new_audio_seconds = sum(float(by_id[record['audio_sha256']]['audio_sec'])
        for record in records if not record.get('existing'))
    profile = dict(stage_totals=stage, process_cpu_seconds=cpu_seconds,
        gpu_streams=GPU_STREAMS, decode_workers=DECODE_WORKERS,
        decode_jobs=decoded_count[0])
    throughput = dict(new_audio_seconds=new_audio_seconds,
        stable_pipeline_seconds=stable_s,
        audio_seconds_per_second=new_audio_seconds / stable_s if stable_s > 0 else None,
        new_jobs_per_second=len(fresh_ids) / stable_s if stable_s > 0 else None)
    return dict(run_id=run_id, output_root=contract['output_root'],
        full_corpus_encode=True,
        attempted=len(records) + len(failures), planned=len(jobs),
        existing_reused=sum(1 for record in records if record.get('existing')),
        quarantined=quarantined, ineligible=ineligible,
        success_ids=[record['audio_sha256'] for record in records], failures=failures,
        stopped=stopped or bool(utilization['rolling_stop_reason']),
        recovery_smoke=dict(sampled=len(smoke_fresh) + len(smoke_existing),
            verified=len(smoke_verified)),
        recovery_verified=smoke_verified,
        throughput=throughput, profile=profile, wall_s=time.monotonic() - started,
        gpu_utilization=utilization, all_existing=not fresh_ids)
