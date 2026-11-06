"""Concurrent per-song eager MERT forward: bitwise identity + throughput probe."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import time

import numpy as np
import torch

from .full_database_app import CHECKPOINT, DATA, app, data, encoder, image, verify_beatfcn_sources


PHASE_SECONDS = 20.0
PHASE_STREAMS = (1, 2, 3, 4)
CONCURRENCY_ROUNDS = 2
MAX_STREAMS = 8


def _parity(reference, candidate):
    difference = np.abs(reference.astype(np.float32) - candidate.astype(np.float32))
    return dict(max_abs=float(difference.max()), mean_abs=float(difference.mean()),
        float16_exact_fraction=float((reference == candidate).mean()),
        finite=bool(np.isfinite(candidate).all()))


def _phase(model, waves, streams, stream_count):
    from beatfcn.cloud.gpu_util import GpuUtil
    monitor = GpuUtil(period_ms=500)
    monitor.take()
    started = time.monotonic()
    deadline = started + PHASE_SECONDS
    counts = {}

    def worker(slot):
        local = 0
        while time.monotonic() < deadline:
            with torch.cuda.stream(streams[slot % stream_count]):
                model.encode(waves[slot % len(waves)])
            local += 1
        counts[slot] = local

    with ThreadPoolExecutor(max_workers=stream_count) as pool:
        list(pool.map(worker, range(stream_count)))
    wall = time.monotonic() - started
    sample = monitor.take()
    monitor.close()
    replays = sum(counts.values())
    audio_seconds = sum(count * len(waves[slot % len(waves)]) / 24000
        for slot, count in counts.items())
    return dict(streams=stream_count, replays=replays, wall_s=wall,
        replays_per_second=replays / wall,
        audio_seconds_per_second=audio_seconds / wall,
        gpu_util_mean_percent=sample.get('util_mean'), gpu_samples=sample.get('samples'))


@app.function(image=image, volumes={DATA: data}, gpu='H100', cpu=(8, 8),
              memory=(16384, 16384), timeout=300, retries=0,
              max_containers=1, scaledown_window=2)
def probe_eager_streams(run_id, contract_json):
    from concurrent.futures import ThreadPoolExecutor
    from beatfcn.audio.mert2 import validate_local_source
    from beatfcn.train.cuda_pipeline import decoded_audio_from_receipt, load_raw_index
    contract = json.loads(contract_json)
    verify_beatfcn_sources(contract)
    if contract.get('probe') != 'mert_eager_streams_v1' or len(contract['jobs']) != 4:
        raise ValueError('Eager streams probe identity changed')
    validate_local_source(CHECKPOINT)
    raw_index = load_raw_index(DATA, contract['raw_content_sha256'],
        contract['raw_manifest_sha256'])
    temporary = Path('/tmp') / 'mert-eager-streams-probe'
    temporary.mkdir(parents=True, exist_ok=True)
    jobs = contract['jobs']
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        decoded = list(pool.map(lambda job: decoded_audio_from_receipt(
            job, raw_index, DATA, temporary)[2], jobs))

    model = encoder()
    references = [model.encode(wave)['layers'] for wave in decoded]

    streams = [torch.cuda.Stream() for _ in range(MAX_STREAMS)]

    def encode_on_stream(slot, wave):
        with torch.cuda.stream(streams[slot % MAX_STREAMS]):
            return model.encode(wave)['layers']

    reports = []
    for round_index in range(CONCURRENCY_ROUNDS):
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            outputs = list(pool.map(lambda pair: encode_on_stream(*pair),
                [(index, wave) for index, wave in enumerate(decoded)]))
        for index, (reference, candidate) in enumerate(zip(references, outputs)):
            report = _parity(reference, candidate)
            report['audio_sha256'] = jobs[index]['audio_sha256']
            report['round'] = round_index
            report['passed'] = bool(report['finite'] and report['max_abs'] == 0
                and report['float16_exact_fraction'] == 1)
            reports.append(report)
    if not all(report['passed'] for report in reports):
        raise ValueError('Concurrent eager/serial parity failed: ' + json.dumps(reports))

    waves_short = decoded[:2]
    phases = [_phase(model, waves_short, streams, count) for count in PHASE_STREAMS]
    for phase in phases:
        print(dict(stage='eager-streams', **phase), flush=True)
    return dict(run_id=run_id, probe='mert_eager_streams_v1', songs=len(jobs),
        parity=reports, phases=phases, phase_seconds=PHASE_SECONDS,
        optimizer_updates=0, diagnostic_only=True, feature_writes=0,
        full_corpus_encoding_authorized=False, training_authorized=False)
