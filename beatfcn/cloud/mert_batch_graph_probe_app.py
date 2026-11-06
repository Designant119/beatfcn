"""Small real-song batched MERT CUDA-graph parity/perf probe."""
from pathlib import Path
import json
import os
import time

import numpy as np
import torch

from .full_database_app import CHECKPOINT, DATA, app, data, encoder, image, verify_beatfcn_sources
from .full_database_mert_graph_app import ShapeGraph
from beatfcn.audio.mert2 import pad_tail


PHASE_SECONDS = 20.0


class BatchShapeGraph:
    """Captured fixed [batch, samples] MERT forward for one exact shape."""
    def __init__(self, model, batch, samples):
        self.batch, self.samples = batch, samples
        self.input = torch.zeros((batch, samples), dtype=torch.float32, device='cuda')

        def forward(value):
            _, states = model.model._encode(value, True)
            return torch.stack((states[model.spec.layers[0] - 1],
                                states[model.spec.layers[1] - 1]))
        self._forward = forward
        stream = torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward(self.input)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = forward(self.input)

    def replay(self, waves):
        if len(waves) != self.batch or any(len(wave) != self.samples for wave in waves):
            raise ValueError('Batch graph shape mismatch')
        for index, wave in enumerate(waves):
            self.input[index].copy_(torch.from_numpy(wave))
        self.graph.replay()

    def close(self):
        self.graph.reset();del self.output;del self.input;del self.graph


def _phase(run):
    from beatfcn.cloud.gpu_util import GpuUtil
    monitor = GpuUtil(period_ms=500);monitor.take()
    cpu_start = os.times();started = time.monotonic();replays = 0
    while time.monotonic() - started < PHASE_SECONDS:
        for _ in range(10):
            run();replays += 1
        torch.cuda.synchronize()
    wall = time.monotonic() - started;sample = monitor.take();cpu_end = os.times();monitor.close()
    cpu = ((cpu_end.user-cpu_start.user)+(cpu_end.system-cpu_start.system)
        +(cpu_end.children_user-cpu_start.children_user)+(cpu_end.children_system-cpu_start.children_system))
    return dict(replays=replays,wall_s=wall,replays_per_second=replays/wall,
        gpu_util_mean_percent=sample.get('util_mean'),gpu_samples=sample.get('samples'),
        process_cpu_seconds=cpu)


def _parity(reference, candidate):
    difference = np.abs(reference.astype(np.float32)-candidate.astype(np.float32))
    return dict(max_abs=float(difference.max()),mean_abs=float(difference.mean()),
        float16_exact_fraction=float((reference==candidate).mean()),
        finite=bool(np.isfinite(candidate).all()))


@app.function(image=image, volumes={DATA: data}, gpu='H100', cpu=(8, 8),
              memory=(16384, 16384), timeout=240, retries=0,
              max_containers=1, scaledown_window=2)
def probe_batch_graph(run_id, contract_json):
    from concurrent.futures import ThreadPoolExecutor
    from beatfcn.audio.mert2 import validate_local_source
    from beatfcn.train.cuda_pipeline import decoded_audio_from_receipt, load_raw_index
    contract = json.loads(contract_json)
    verify_beatfcn_sources(contract)
    if contract.get('probe') != 'mert_batch_cuda_graph_v1' or len(contract['jobs']) != contract.get('batch'):
        raise ValueError('Batch graph probe identity changed')
    validate_local_source(CHECKPOINT)
    raw_index = load_raw_index(DATA, contract['raw_content_sha256'],
        contract['raw_manifest_sha256'])
    temporary = Path('/tmp')/'mert-batch-graph-probe';temporary.mkdir(parents=True,exist_ok=True)
    jobs = contract['jobs']
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        decoded = list(pool.map(lambda job: decoded_audio_from_receipt(
            job, raw_index, DATA, temporary)[2], jobs))
    padded = [pad_tail(wave) for wave in decoded]
    samples = len(padded[0])
    if len({len(value) for value in padded}) != 1:
        raise ValueError('Probe selection did not decode to one exact padded shape')
    model = encoder();references = [model.encode(wave)['layers'] for wave in decoded]

    batch = BatchShapeGraph(model,len(padded),samples)
    batch.replay(padded);torch.cuda.synchronize()
    candidates = [batch.output[:, index].float().cpu().numpy().astype(np.float16)
                  for index in range(len(padded))]
    reports=[]
    for index,(reference,candidate) in enumerate(zip(references,candidates)):
        report=_parity(reference,candidate);report['audio_sha256']=jobs[index]['audio_sha256']
        report['passed']=bool(report['finite'] and report['max_abs']==0
            and report['float16_exact_fraction']==1);reports.append(report)
    if not all(report['passed'] for report in reports):
        raise ValueError('Batch graph/single parity failed: '+json.dumps(reports))

    single=ShapeGraph(model,samples)
    single_phase=_phase(lambda:(single.replay(padded[0])))
    batch_phase=_phase(lambda:batch.replay(padded))
    eager_phase=_phase(lambda:model.model._encode(batch.input,True))
    single.close();batch.close()
    audio_seconds=sum(job['audio_sec'] for job in jobs)
    return dict(run_id=run_id,probe='mert_batch_cuda_graph_v1',batch=len(jobs),
        samples=samples,parity=reports,
        phases=dict(single_graph=single_phase,batch_graph=batch_phase,batch_eager=eager_phase),
        audio_seconds_per_graph_replay=audio_seconds,
        optimizer_updates=0,diagnostic_only=True,feature_writes=0,
        full_corpus_encoding_authorized=False,training_authorized=False)
