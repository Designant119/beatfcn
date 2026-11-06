"""True batched MERT2 forward over equal-padded-length windows.

Only host/kernel scheduling changes. Window geometry, tail handling, retained
layers, feature clocks and per-song stitch equations mirror FrozenMert2.encode.
"""
from dataclasses import asdict
import math

import numpy as np
import torch

from .mert2 import (
    CELL_SAMPLES, PREPROCESS_RECIPE, REPO, REVISION, SR, Mert2Spec, pad_tail, windows)


def _metadata(model, wave, chunks, receipts, spec):
    return dict(repo=REPO, revision=REVISION, spec=asdict(spec), sample_rate=SR,
        real_samples=len(wave), nominal_first_center_ms=15, measured_acoustic_offset_ms=None,
        full_song_context=len(chunks) == 1, windows=receipts,
        encoded_samples=sum(item['padded_samples'] for item in receipts),
        fine_mel_recipe=PREPROCESS_RECIPE)


@torch.inference_mode()
def encode_windows_batch(model, waves, *, window_batch=4):
    """Encode a small group of songs, batching only identical padded lengths."""
    if type(window_batch) is not int or not 1 <= window_batch <= 16:
        raise ValueError('MERT window batch must be an integer in [1,16]')
    spec = model.spec
    waves = [np.asarray(wave, dtype=np.float32) for wave in waves]
    if not waves or any(wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all()
                        for wave in waves):
        raise ValueError('Expected nonempty finite mono waveforms')
    chunks_by_job = [windows(len(wave), spec) for wave in waves]
    groups = {}
    for job, (wave, chunks) in enumerate(zip(waves, chunks_by_job)):
        for chunk, (start, end) in enumerate(chunks):
            padded = pad_tail(wave[start:end])
            groups.setdefault(len(padded), []).append(dict(job=job, chunk=chunk,
                start=start, end=end, padded=padded))
    kept_by_job = [[] for _ in waves]
    ids_by_job = [[] for _ in waves]
    receipts_by_job = [[] for _ in waves]
    for length in sorted(groups):
        entries = sorted(groups[length], key=lambda item: (item['job'], item['chunk']))
        for offset in range(0, len(entries), window_batch):
            batch = entries[offset:offset + window_batch]
            signal = torch.from_numpy(np.stack([item['padded'] for item in batch])).to(model.device)
            output = model.model(input_values=signal, output_hidden_states=True)
            features = torch.stack([output.hidden_states[layer - 1] for layer in spec.layers])
            for index, entry in enumerate(batch):
                job = entry['job']
                chunk = entry['chunk']
                start = entry['start']
                end = entry['end']
                wave = waves[job]
                chunks = chunks_by_job[job]
                values = features[:, index]
                ids = np.arange(values.shape[1]) * CELL_SAMPLES + start
                nominal = ids + 360
                left = 0 if chunk == 0 else (chunks[chunk - 1][1] + start) // 2
                right = len(wave) if chunk + 1 == len(chunks) else (end + chunks[chunk + 1][0]) // 2
                mask = ((nominal >= left) & ((nominal < right) | (chunk + 1 == len(chunks)))
                    & (ids < len(wave)))
                keep = torch.from_numpy(np.flatnonzero(mask)).to(model.device)
                values = values.index_select(1, keep).float().cpu().numpy().astype(np.float16)
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite batched MERT2 output')
                kept_by_job[job].append((chunk, values))
                ids_by_job[job].append((chunk, ids[mask]))
                receipts_by_job[job].append((chunk, dict(start_sample=start, end_sample=end,
                    padded_samples=length, kept_frames=int(mask.sum()),
                    retain_interval_samples=[left, right])))
            del signal, output, features
    results = []
    for job, wave in enumerate(waves):
        kept = [value for _, value in sorted(kept_by_job[job], key=lambda item: item[0])]
        frame_ids = [value for _, value in sorted(ids_by_job[job], key=lambda item: item[0])]
        receipts = [value for _, value in sorted(receipts_by_job[job], key=lambda item: item[0])]
        ids = np.concatenate(frame_ids)
        expected = np.arange(math.ceil(len(wave) / CELL_SAMPLES)) * CELL_SAMPLES
        if not np.array_equal(ids, expected):
            raise ValueError('Gap/overlap in batched stitched feature clock')
        results.append(dict(layers=np.concatenate(kept, axis=1), times_s=(ids + 360) / SR,
            intervals_s=np.stack((ids, np.minimum(ids + CELL_SAMPLES, len(wave))), axis=-1) / SR,
            metadata=_metadata(model, wave, chunks_by_job[job], receipts, spec)))
    return results


def feature_parity(reference, candidate):
    """Compare two feature payloads without hiding float16 quantization differences."""
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError('Batch/single feature shapes differ')
    difference = np.abs(reference.astype(np.float32) - candidate.astype(np.float32))
    exact = reference == candidate
    return dict(shape=list(reference.shape),
        max_abs=float(difference.max()) if difference.size else 0.,
        mean_abs=float(difference.mean()) if difference.size else 0.,
        float16_exact_fraction=float(exact.mean()) if exact.size else 1.,
        finite=bool(np.isfinite(candidate).all()))
