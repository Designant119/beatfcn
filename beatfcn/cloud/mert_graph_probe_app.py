"""Bounded MERT CUDA-graph probe on one real song; no training or corpus write."""
from pathlib import Path
import json
import os
import time

import numpy as np
import torch

from .full_database_app import (
    CHECKPOINT, DATA, app, data, encoder, image, verify_beatfcn_sources)


PHASE_SECONDS = 30.0
REPLAY_BLOCK = 20


def _layer_payload(output, expected_frames, layers):
    if hasattr(output, 'hidden_states'):
        states = output.hidden_states
    else:
        _, states = output
    values = torch.stack([states[layer - 1][0]
                          for layer in layers]).float().cpu().numpy()
    return values[:, :expected_frames].astype(np.float16)


def _parity(reference, candidate):
    difference = np.abs(reference.astype(np.float32) - candidate.astype(np.float32))
    exact = reference == candidate
    return dict(shape=list(reference.shape), max_abs=float(difference.max()),
        mean_abs=float(difference.mean()), float16_exact_fraction=float(exact.mean()),
        finite=bool(np.isfinite(candidate).all()))


def _phase(model, graph, static_input, seconds):
    from beatfcn.cloud.gpu_util import GpuUtil
    monitor = GpuUtil(period_ms=500)
    monitor.take()  # drain setup; the repeated real forward phase starts now
    cpu_start = os.times()
    started = time.monotonic()
    replays = 0
    torch.cuda.synchronize()
    while time.monotonic() - started < seconds:
        for _ in range(REPLAY_BLOCK):
            if graph is None:
                output = model.model(input_values=static_input, output_hidden_states=True)
                del output
            else:
                graph.replay()
            replays += 1
        # Keep the launch queue bounded without synchronizing every replay.
        torch.cuda.synchronize()
    wall = time.monotonic() - started
    sample = monitor.take()
    cpu_end = os.times()
    monitor.close()
    cpu_seconds = ((cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
        + (cpu_end.children_user - cpu_start.children_user)
        + (cpu_end.children_system - cpu_start.children_system))
    return dict(replays=replays, wall_s=wall, replays_per_second=replays / wall,
        gpu_util_mean_percent=sample.get('util_mean'), gpu_samples=sample.get('samples'),
        process_cpu_seconds=cpu_seconds,
        cpu_percent_of_four_cores=100 * cpu_seconds / (4 * wall))


@app.function(image=image, volumes={DATA: data}, gpu='H100', cpu=(4, 4),
              memory=(4096, 4096), timeout=150, retries=0,
              max_containers=1, scaledown_window=2)
def probe_graph(run_id, contract_json):
    from beatfcn.audio.mert2 import pad_tail, validate_local_source
    from beatfcn.train.cuda_pipeline import decoded_audio_from_receipt, load_raw_index
    contract = json.loads(contract_json)
    verify_beatfcn_sources(contract)
    if contract.get('probe') != 'mert_cuda_graph_v1' or len(contract['jobs']) != 1:
        raise ValueError('MERT graph probe identity changed')
    validate_local_source(CHECKPOINT)
    job = contract['jobs'][0]
    raw_index = load_raw_index(DATA, contract['raw_content_sha256'],
        contract['raw_manifest_sha256'])
    temporary_root = Path('/tmp') / 'mert-graph-probe'
    temporary_root.mkdir(parents=True, exist_ok=True)
    _, original, wave = decoded_audio_from_receipt(job, raw_index, DATA, temporary_root)
    model = encoder()
    layers = model.spec.layers
    reference = model.encode(wave)
    expected_frames = (len(wave) + 959) // 960
    padded = pad_tail(wave)
    static_input = torch.zeros((1, len(padded)), dtype=torch.float32, device='cuda')
    static_input[0].copy_(torch.from_numpy(padded))
    if not bool(torch.isfinite(static_input).all()):
        raise ValueError('Graph-probe input is nonfinite')

    # Warm up outside capture, including the pinned model's mutable rotary cache.
    # Capture _encode directly: public forward contains a host-synchronizing
    # `if not torch.isfinite(input).all()` validation on every call.
    def graph_forward(value):
        return model.model._encode(value, True)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            graph_forward(static_input)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = graph_forward(static_input)
    graph.replay()
    torch.cuda.synchronize()
    candidate = _layer_payload(graph_output, expected_frames, layers)
    parity = _parity(reference['layers'], candidate)
    parity['clock_exact'] = bool(
        (reference['times_s']
         == (np.arange(expected_frames, dtype=np.int64) * 960 + 360) / 24000).all())
    parity['passed'] = bool(parity['finite'] and parity['clock_exact']
        and parity['max_abs'] == 0 and parity['float16_exact_fraction'] == 1)
    if not parity['passed']:
        raise ValueError('MERT CUDA graph is not bitwise equivalent to single forward: '
            + json.dumps(parity))
    original = Path(original)
    if original.is_relative_to(temporary_root):
        original.unlink()

    eager = _phase(model, None, static_input, PHASE_SECONDS)
    graph_phase = _phase(model, graph, static_input, PHASE_SECONDS)
    return dict(run_id=run_id, probe='mert_cuda_graph_v1', parity_only=False,
        optimizer_updates=0, audio_sha256=job['audio_sha256'],
        audio_seconds=job['audio_sec'], padded_samples=len(padded),
        expected_frames=expected_frames, parity=parity,
        eager=eager, graph=graph_phase,
        gpu_utilization_improvement_percentage=(
            graph_phase['gpu_util_mean_percent'] - eager['gpu_util_mean_percent']
            if graph_phase.get('gpu_util_mean_percent') is not None
            and eager.get('gpu_util_mean_percent') is not None else None),
        stable_period_started=False, diagnostic_only=True,
        full_corpus_encoding_authorized=False, training_authorized=False)
