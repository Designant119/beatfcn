"""Formal exact-shape MERT CUDA-graph pipeline.

The model, retained layers, clocks and immutable receipts are unchanged. CUDA
graphs only remove host launch gaps; every capture, decode, mel, replay, copy
and write below is inside the official stable-period monitor.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
import json
import os
import threading
import time

import numpy as np
import torch

from .full_database_app import (
    CHECKPOINT, CPU_WORKERS, DATA, REMOTE_ROOT, StableGpuMonitor, app, data,
    encoder, image, verify_beatfcn_sources, verify_feature_folder)
from beatfcn.audio.mert2 import CELL_SAMPLES, PREPROCESS_RECIPE, REPO, REVISION, SR, Mert2Spec, pad_tail


WRITE_WORKERS = 8
PREFETCH_WORKERS = 24
IN_FLIGHT_GROUPS = 24
COMMIT_EVERY = 16
EXPECTED_PIPELINE = dict(graph='mert_exact_shape_cuda_graph_v1', prefetch_workers=24,
    write_workers=8, in_flight_groups=24, commit_every=16,
    direct_verified_raw=True, cuda_mel=True, single_pass_write_hash=True)


class ShapeGraph:
    """One captured fixed-shape MERT forward returning retained layers only."""
    def __init__(self, model, samples):
        self.samples = samples
        self.input = torch.zeros((1, samples), dtype=torch.float32, device='cuda')
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._forward(model)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = self._forward(model)

    def _forward(self, model):
        # Public forward has a host-synchronizing finite check; input is validated here.
        _, states = model.model._encode(self.input, True)
        return torch.stack((states[model.spec.layers[0] - 1],
                            states[model.spec.layers[1] - 1]))

    def replay(self, padded):
        if len(padded) != self.samples:
            raise ValueError('MERT graph shape mismatch')
        if not np.isfinite(padded).all():
            raise ValueError('Nonfinite MERT graph input')
        self.input[0].copy_(torch.from_numpy(padded))
        self.graph.replay()

    def close(self):
        self.graph.reset()
        del self.output
        del self.input
        del self.graph


def _decode(job, raw_index, temporary_root):
    from beatfcn.train.cuda_pipeline import decoded_audio_from_receipt
    started = time.monotonic()
    audio, original, wave = decoded_audio_from_receipt(job, raw_index, DATA, temporary_root)
    return dict(job=job, audio=audio, original=original, wave=wave,
        decode_s=time.monotonic() - started)


def _write(prefetched, mel, layers, expected, output_root_path, temporary_root):
    from beatfcn.train.cuda_pipeline import commit_feature_hashing
    started = time.monotonic()
    features = dict(layers=layers,
        times_s=(np.arange(expected, dtype=np.int64) * CELL_SAMPLES + 360) / SR,
        intervals_s=np.stack((np.arange(expected, dtype=np.int64) * CELL_SAMPLES,
            np.minimum((np.arange(expected, dtype=np.int64) + 1) * CELL_SAMPLES,
                       len(prefetched['wave']))), axis=-1) / SR,
        metadata=dict(repo=REPO, revision=REVISION, spec=asdict(Mert2Spec()),
            sample_rate=SR, real_samples=len(prefetched['wave']),
            nominal_first_center_ms=15, measured_acoustic_offset_ms=None,
            full_song_context=True, windows=[dict(start_sample=0,
                end_sample=len(prefetched['wave']), padded_samples=(len(prefetched['wave'])
                    + CELL_SAMPLES - 1) // CELL_SAMPLES * CELL_SAMPLES,
                kept_frames=expected, retain_interval_samples=[0, len(prefetched['wave'])])],
            encoded_samples=(len(prefetched['wave']) + CELL_SAMPLES - 1)
                // CELL_SAMPLES * CELL_SAMPLES,
            fine_mel_recipe=PREPROCESS_RECIPE))
    result = commit_feature_hashing(prefetched['audio'], Path(prefetched['original']),
        output_root_path, temporary_root, features, mel, expected)
    result['wall_s'] = time.monotonic() - started
    return result


def _graph_parity(model, jobs, raw_index, temporary_root):
    from beatfcn.audio.mert2_batch import feature_parity
    reference_job = min(jobs, key=lambda job: abs(job['audio_sec'] - 30))
    prefetched = _decode(reference_job, raw_index, temporary_root)
    reference = model.encode(prefetched['wave'])
    padded = pad_tail(prefetched['wave'])
    runner = ShapeGraph(model, len(padded))
    runner.replay(padded)
    torch.cuda.synchronize()
    candidate = runner.output[:, 0].float().cpu().numpy().astype(np.float16)
    report = feature_parity(reference['layers'], candidate)
    runner.close()
    original = Path(prefetched['original'])
    if original.is_relative_to(Path(temporary_root)):
        original.unlink()
    report['audio_sha256'] = reference_job['audio_sha256']
    report['clock_exact'] = bool((reference['times_s']
        == (np.arange(candidate.shape[1], dtype=np.int64) * CELL_SAMPLES + 360) / SR).all())
    report['passed'] = bool(report['finite'] and report['clock_exact']
        and report['max_abs'] == 0 and report['float16_exact_fraction'] == 1)
    if not report['passed']:
        raise ValueError('Formal graph/single parity failed: ' + json.dumps(report))
    return report


@app.function(image=image, volumes={DATA: data}, gpu='H100', cpu=(8, 8),
              memory=(32768, 32768), timeout=2400, retries=0,
              max_containers=1, scaledown_window=120)
def encode_mert_graph(run_id, contract_json, dispatch_deadline_unix):
    from beatfcn.audio.mert2 import validate_local_source
    from beatfcn.audio.cuda_mel import complete_song_mel_cuda
    from beatfcn.train.cuda_pipeline import load_raw_index
    contract = json.loads(contract_json)
    verify_beatfcn_sources(contract)
    if contract.get('pipeline') != EXPECTED_PIPELINE or not isinstance(contract.get('diagnostic_only'), bool):
        raise ValueError('Formal MERT graph pipeline identity changed')
    validate_local_source(CHECKPOINT)
    jobs = contract['jobs']
    if not jobs or len({job['audio_sha256'] for job in jobs}) != len(jobs):
        raise ValueError('Jobs must be nonempty and unique')
    raw_index = load_raw_index(DATA, contract['raw_content_sha256'],
        contract['raw_manifest_sha256'])
    output_root_path = Path(DATA) / contract['output_root']
    temporary_root = Path('/tmp') / 'mert-graph-formal'
    temporary_root.mkdir(parents=True, exist_ok=True)
    model = encoder()
    if model.spec.layers != (12, 23):
        raise ValueError('MERT retained layers changed')
    parity = _graph_parity(model, jobs, raw_index, temporary_root)

    # Selection is shape-grouped. Exact decoded shape is re-established after decode.
    monitor = StableGpuMonitor()
    started = time.monotonic()
    cpu_start = os.times()
    stage = dict(decode_s=0., decode_osu_s=0., decode_phira_s=0.,
        prefetch_high_water=0, mel_s=0., capture_s=0., replay_s=0., copy_s=0., write_s=0.,
        graph_shapes=0, cuda_memory_allocated_max=0)
    records = []
    failures = []
    stopped = False
    lock = threading.Lock()
    completed_count = 0
    new_audio_seconds = 0.
    with ExitStack() as stack:
        prefetch_pool = stack.enter_context(ThreadPoolExecutor(
            max_workers=PREFETCH_WORKERS, thread_name_prefix='graph-decode'))
        writer_pool = stack.enter_context(ThreadPoolExecutor(
            max_workers=WRITE_WORKERS, thread_name_prefix='graph-write'))
        recovery_pool = prefetch_pool

        def existing(job):
            folder = output_root_path / 'features' / job['audio_sha256']
            if not (folder / 'receipt.json').exists():
                return job, None
            return job, verify_feature_folder(folder, job, model.revision)

        ready = []
        for job, prior in prefetch_pool.map(existing, jobs):
            if prior is None:
                ready.append(job)
            else:
                records.append(dict(audio_sha256=job['audio_sha256'], existing=True))
        planned_groups = []
        previous_key = None
        for job in ready:
            # Keep each expected exact padded shape together so one graph serves the
            # entire repeated-shape group instead of being recaptured per 8-song chunk.
            key = job['expected_mel_frames'] // 4 + 1
            if planned_groups and key == previous_key:
                planned_groups[-1].append(job)
            else:
                planned_groups.append([job])
            previous_key = key
        monitor.start()
        torch.cuda.reset_peak_memory_stats()
        group_index = 0
        group_futures = {}
        stop_submitting = False
        writer_futures = {}
        try:
            while (group_index < len(planned_groups) or group_futures or writer_futures):
                if time.time() >= dispatch_deadline_unix or monitor.stop_reason:
                    stopped = True
                    stop_submitting = True
                while (not stop_submitting and group_index < len(planned_groups)
                        and len(group_futures) < IN_FLIGHT_GROUPS):
                    group = planned_groups[group_index]
                    future = prefetch_pool.submit(lambda values: [
                        _decode(value, raw_index, temporary_root) for value in values], group)
                    group_futures[future] = group
                    with lock:
                        stage['prefetch_high_water'] = max(stage['prefetch_high_water'],
                            len(group_futures))
                    group_index += 1
                pending = set(group_futures) | set(writer_futures)
                if not pending:
                    continue
                done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    if future in group_futures:
                        group = group_futures.pop(future)
                        try:
                            decoded = future.result()
                            with lock:
                                stage['decode_s'] += sum(value['decode_s'] for value in decoded)
                                stage['decode_osu_s'] += sum(value['decode_s']
                                    for value in decoded if value['job']['source'] == 'osu')
                                stage['decode_phira_s'] += sum(value['decode_s']
                                    for value in decoded if value['job']['source'] == 'phira')
                            by_shape = {}
                            for value in decoded:
                                padded = pad_tail(value['wave'])
                                if not np.isfinite(padded).all():
                                    raise ValueError('Nonfinite decoded audio')
                                by_shape.setdefault(len(padded), []).append((value, padded))
                            for samples, values in sorted(by_shape.items()):
                                expected = samples // CELL_SAMPLES
                                capture_started = time.monotonic()
                                runner = ShapeGraph(model, samples)
                                capture_s = time.monotonic() - capture_started
                                # Bound device result memory while preserving back-to-back replays.
                                for offset in range(0, len(values), 8):
                                    chunk = values[offset:offset + 8]
                                    device_output = torch.empty(
                                        (len(chunk), 2, expected, 1024),
                                        dtype=torch.float16, device='cuda')
                                    replay_started = time.monotonic()
                                    for index, (value, padded) in enumerate(chunk):
                                        runner.replay(padded)
                                        device_output[index].copy_(runner.output[:, 0])
                                    torch.cuda.synchronize()
                                    replay_s = time.monotonic() - replay_started
                                    copy_started = time.monotonic()
                                    hosts = device_output.cpu().numpy()
                                    del device_output
                                    with lock:
                                        stage['replay_s'] += replay_s
                                        stage['copy_s'] += time.monotonic() - copy_started
                                        stage['cuda_memory_allocated_max'] = max(
                                            stage['cuda_memory_allocated_max'],
                                            int(torch.cuda.max_memory_allocated()))
                                    for index, (value, padded) in enumerate(chunk):
                                        mel_started = time.monotonic()
                                        mel = complete_song_mel_cuda(value['wave'])
                                        with lock:
                                            stage['mel_s'] += time.monotonic() - mel_started
                                        writer = writer_pool.submit(_write, value, mel,
                                            hosts[index], expected, output_root_path,
                                            temporary_root)
                                        writer_futures[writer] = value['job']
                                with lock:
                                    stage['capture_s'] += capture_s
                                    stage['graph_shapes'] += 1
                                runner.close()
                        except Exception as exc:
                            for job in group:
                                failures.append(dict(audio_sha256=job['audio_sha256'],
                                    stage='graph', error=f'{type(exc).__name__}: {exc}'))
                            stopped = stop_submitting = True
                    else:
                        job = writer_futures.pop(future)
                        try:
                            result = future.result()
                            with lock:
                                stage['write_s'] += result['wall_s']
                                completed_count += 1
                                new_audio_seconds += float(job['audio_sec'])
                                records.append(dict(audio_sha256=job['audio_sha256'],
                                    existing=False, files=result['files']))
                                commit_now = completed_count % COMMIT_EVERY == 0
                            if commit_now:
                                data.commit()
                                print(dict(stage='mert-graph', completed=completed_count,
                                    planned=len(ready), failures=len(failures),
                                    wall_s=time.monotonic() - started,
                                    gpu_monitor=monitor.summary(), stage_totals=stage),
                                    flush=True)
                        except Exception as exc:
                            failures.append(dict(audio_sha256=job['audio_sha256'],
                                stage='write', error=f'{type(exc).__name__}: {exc}'))
                            stopped = stop_submitting = True
            data.commit()
        finally:
            monitor.close()
        utilization = monitor.summary()
        cpu_end = os.times()
        stable_s = utilization.get('stable_observed_s') or 0.
        cpu_seconds = ((cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
            + (cpu_end.children_user - cpu_start.children_user)
            + (cpu_end.children_system - cpu_start.children_system))
        profile = dict(stage_totals=stage, process_cpu_seconds=cpu_seconds,
            estimated_cpu_percent_of_eight_cores=100 * cpu_seconds / (8 * stable_s)
                if stable_s > 0 else None,
            graph_single_parity=parity)
        data.reload()
        by_id = {job['audio_sha256']: job for job in jobs}
        recovery = list(recovery_pool.map(lambda record: verify_feature_folder(
            output_root_path / 'features' / record['audio_sha256'],
            by_id[record['audio_sha256']], model.revision), records))
    throughput = dict(new_audio_seconds=new_audio_seconds,
        stable_pipeline_seconds=stable_s,
        audio_seconds_per_second=new_audio_seconds / stable_s if stable_s > 0 else None,
        new_jobs_per_second=completed_count / stable_s if stable_s > 0 else None)
    return dict(run_id=run_id, output_root=contract['output_root'],
        diagnostic_only=contract['diagnostic_only'],
        attempted=len(records) + len(failures), planned=len(jobs),
        success_ids=[record['audio_sha256'] for record in records], failures=failures,
        stopped=stopped or bool(utilization['rolling_stop_reason']),
        recovery_verified=[record['audio_sha256'] for record in recovery],
        throughput=throughput, profile=profile, wall_s=time.monotonic() - started,
        gpu_utilization=utilization, all_existing=False)
