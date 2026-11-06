"""CPU-only decode concurrency probe; no GPU, MERT output writes or training."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import random
import time

from .full_database_app import DATA, app, data, image, verify_beatfcn_sources


def _cleanup(original):
    original = Path(original)
    temporary = Path('/tmp') / 'mert-decode-concurrency-probe'
    if original.is_relative_to(temporary):
        original.unlink(missing_ok=True)


@app.function(image=image, volumes={DATA: data}, cpu=(8, 8), memory=(8192, 8192),
              timeout=600, retries=0, max_containers=1, scaledown_window=2)
def probe_decode_concurrency(run_id, contract_json):
    from beatfcn.audio.mert2 import CELL_SAMPLES
    from beatfcn.train.cuda_pipeline import decoded_audio_from_receipt, load_raw_index
    contract = json.loads(contract_json)
    verify_beatfcn_sources(contract)
    if contract.get('probe') != 'mert_decode_concurrency_v1':
        raise ValueError('Decode concurrency probe identity changed')
    worker_counts = contract.get('worker_counts')
    if not worker_counts or not all(isinstance(count, int) and count > 0
            for count in worker_counts):
        raise ValueError('Decode probe worker counts must be a nonempty positive int list')
    blocks = contract.get('blocks')
    if blocks:
        if len(blocks) != len(worker_counts) or not all(
                isinstance(block, list) and block for block in blocks):
            raise ValueError('Decode probe blocks must align with worker counts')
        phase_jobs = blocks
    else:
        phase_jobs = [contract['jobs']] * len(worker_counts)
    jobs = [job for block in phase_jobs for job in block]
    if not jobs or len({job['audio_sha256'] for job in jobs}) != len(jobs):
        raise ValueError('Decode probe jobs must be nonempty and unique')
    raw_index = load_raw_index(DATA, contract['raw_content_sha256'],
        contract['raw_manifest_sha256'])
    temporary_root = Path('/tmp') / 'mert-decode-concurrency-probe'
    temporary_root.mkdir(parents=True, exist_ok=True)

    def decode_one(job):
        started = time.monotonic()
        _, original, wave = decoded_audio_from_receipt(job, raw_index, DATA, temporary_root)
        # The legacy mel10 receipt is the authoritative decoded-sample clock.
        if 4 * (len(wave) // CELL_SAMPLES) != job['expected_mel_frames']:
            raise ValueError('Decoded sample clock differs from authoritative mel receipt: '
                + job['audio_sha256'])
        _cleanup(original)
        return dict(source=job['source'], wall_s=time.monotonic() - started,
            audio_seconds=job['audio_sec'])

    phases = []
    for round_index, (workers, block) in enumerate(zip(worker_counts, phase_jobs)):
        ordered = list(block)
        random.Random(20261031 + round_index).shuffle(ordered)
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='decode-probe') as pool:
            values = list(pool.map(decode_one, ordered))
        wall = time.monotonic() - started
        phase = dict(round=round_index, workers=workers, jobs=len(values),
            wall_s=wall, jobs_per_second=len(values) / wall,
            audio_seconds_per_second=sum(value['audio_seconds'] for value in values) / wall,
            aggregate_task_s=sum(value['wall_s'] for value in values),
            by_source={})
        for source in ('osu', 'phira'):
            selected = [value for value in values if value['source'] == source]
            phase['by_source'][source] = dict(jobs=len(selected),
                jobs_per_second=len(selected) / wall,
                aggregate_task_s=sum(value['wall_s'] for value in selected))
        phases.append(phase)
        print(dict(stage='decode-probe', **phase), flush=True)
    return dict(run_id=run_id, probe='mert_decode_concurrency_v1', phases=phases,
        disjoint_blocks=bool(blocks), optimizer_updates=0, diagnostic_only=True,
        gpu_calls=0, full_corpus_encoding_authorized=False, training_authorized=False)
