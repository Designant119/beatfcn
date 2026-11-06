"""Run the pinned Ensomi release on one audio file, with no chart-label inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

# The external release explicitly requests unrestricted loading. Override that at
# the documented PyTorch boundary; do not silently retry unsafe deserialization.
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '1'
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(value):
    a = np.asarray(value)
    return dict(shape=list(a.shape), dtype=str(a.dtype), nan=int(np.isnan(a).sum()),
                inf=int(np.isinf(a).sum()), min=float(a.min()), max=float(a.max()),
                mean=float(a.mean(dtype=np.float64)), std=float(a.std(dtype=np.float64)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    for name, digest in plan['input_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'frozen input changed: {name}')
    torch.set_num_threads(plan['threads'])
    torch.set_num_interop_threads(1)
    from ensomi_model.system.make import System
    from ensomi_model.audio_rows.decoder import lead_seconds
    started = time.time()
    result = dict(id=plan['id'], complete=False, model_updates=0, new_cloud_spend_usd=0,
                  external_pretrained=True, started_at_unix=started,
                  plan_sha256=sha(args.plan), requests=[])
    destination = ROOT / plan['result']

    def save():
        destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')

    save()
    model_dir = ROOT / plan['model_dir']
    config = json.loads((model_dir / 'system-offline.json').read_text())
    assert (model_dir / config['beatthis']).is_file(), 'offline checkpoint absent'
    t0 = time.monotonic()
    # The verified rows checkpoint stores numeric NumPy calibration arrays as
    # well as tensors. Permit only those numeric container/dtype constructors.
    numeric_types = [type(np.dtype(x)) for x in ('float16', 'float32', 'float64', 'int32', 'int64', 'uint8', 'bool')]
    with torch.serialization.safe_globals([np.ndarray, np.dtype, np.core.multiarray._reconstruct, *numeric_types]):
        system = System.load(config, model_dir)
    result['model_load_s'] = time.monotonic() - t0
    result['model_health'] = {}
    for name, model in [('rows', system.rows.model), ('arranger', system.r2),
                        ('beatthis', system.rows.listener.frames.model)]:
        parameters = list(model.parameters())
        result['model_health'][name] = dict(parameters=sum(p.numel() for p in parameters),
            finite=all(bool(torch.isfinite(p).all()) for p in parameters),
            training_mode=model.training, devices=sorted({str(p.device) for p in parameters}))
        assert result['model_health'][name]['finite'] and not model.training
        assert result['model_health'][name]['devices'] == ['cpu']
    audio = ROOT / plan['audio']
    result['ffmpeg_to_native_clock_lead_s'] = lead_seconds(audio, 'ffmpeg')
    # Evaluation is frozen to the decoded-audio clock. Any nonzero native lead
    # requires an explicit new contract before generation, never a fitted shift.
    if result['ffmpeg_to_native_clock_lead_s'] != 0:
        raise ValueError('nonzero native/decoded lead requires a separately frozen clock contract')
    with torch.inference_mode():
        t0 = time.monotonic()
        prepared = system.rows.prepare(audio)
        result['audio_preparation_s'] = time.monotonic() - t0
        song = prepared['song']
        result['input_health'] = dict(features=stats(song.features), segments=stats(song.segments),
                                     song_ms=song.song_ms, listener_seconds=song.seconds)
        assert np.isfinite(song.features).all() and np.isfinite(song.segments).all()
        np.savez_compressed(ROOT / plan['audio_feature_output'], features=song.features,
                            segments=song.segments, song_ms=np.array(song.song_ms))
        save()
        for request in plan['requests']:
            print(json.dumps({'phase': 'generate', 'request': request}), flush=True)
            t0 = time.monotonic()
            receipt = system.make(audio, request['star'], ROOT / request['output'],
                                  prepared=prepared, seed=plan['seed'], name=request['label'],
                                  title=plan['title'], artist=plan['artist'])
            result['requests'].append(dict(label=request['label'], star=request['star'],
                wall_s=time.monotonic() - t0, receipt=receipt,
                osz_sha256=sha(ROOT / request['output'])))
            save()
    result.update(complete=True, ended_at_unix=time.time(), wall_s=time.time() - started)
    save()
    print(json.dumps({k: result[k] for k in ('complete', 'model_load_s', 'audio_preparation_s', 'wall_s')}))


if __name__ == '__main__':
    main()
