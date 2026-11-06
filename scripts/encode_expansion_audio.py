"""D55: encode independent expansion songs with the unchanged D48 frontend."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '1'
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(a):
    a = np.asarray(a)
    finite = a[np.isfinite(a)]
    return dict(shape=list(a.shape), dtype=str(a.dtype), nan=int(np.isnan(a).sum()),
        inf=int(np.isinf(a).sum()), min=float(finite.min()) if finite.size else None,
        max=float(finite.max()) if finite.size else None,
        mean=float(finite.mean(dtype=np.float64)) if finite.size else None,
        std=float(finite.std(dtype=np.float64)) if finite.size else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    output = ROOT / plan['result']
    if output.exists():
        raise FileExistsError('previous D55 inference result exists')
    for path, expected in plan['inference_input_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('inference input drift: ' + path)
    inputs = json.loads((ROOT / plan['audio_inputs']).read_text())
    torch.set_num_threads(plan['threads'])
    torch.set_num_interop_threads(1)
    torch.manual_seed(plan['seed'])
    np.random.seed(plan['seed'])
    from ensomi_model.audio_rows.audio import Listener
    from ensomi_model.audio_rows.decoder import lead_seconds
    start = time.time()
    result = dict(id='D55', complete=False, plan_sha256=sha(args.plan),
        model_updates=0, device='cpu', new_cloud_spend_usd=0,
        label_inputs=False, started_at_unix=start, songs=[])

    def save():
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')

    save()
    t0 = time.monotonic()
    listener = Listener(ROOT / plan['checkpoint'], device='cpu')
    original_loader = listener.load
    model = listener.frames.model
    parameters = list(model.parameters())
    result['model_health'] = dict(parameters=sum(p.numel() for p in parameters),
        finite=all(bool(torch.isfinite(p).all()) for p in parameters),
        training=model.training, devices=sorted({str(p.device) for p in parameters}))
    assert result['model_health']['finite'] and not model.training
    assert result['model_health']['devices'] == ['cpu']
    result['model_load_s'] = time.monotonic() - t0
    arrays = ROOT / plan['array_dir']
    arrays.mkdir(parents=True, exist_ok=True)
    for entry in inputs['rows']:
        print(json.dumps(dict(phase='listen', chart_id=entry['chart_id'])), flush=True)
        record = dict(chart_id=entry['chart_id'], audio=entry['audio'],
                      audio_sha256=entry['audio_sha256'], complete=False)
        t0 = time.monotonic()
        try:
            def measured_loader(path):
                try:
                    wave, sample_rate = original_loader(path)
                except Exception as exc:
                    record['beat_audio_loader'] = dict(fallback_to_mel=True,
                        error_type=type(exc).__name__, error=str(exc))
                    raise
                record['beat_audio_loader'] = dict(fallback_to_mel=False,
                    sample_rate=sample_rate, waveform=stats(wave))
                return wave, sample_rate
            listener.load = measured_loader
            audio = ROOT / entry['audio']
            if sha(audio) != entry['audio_sha256']:
                raise ValueError('selected audio changed')
            ffmpeg_lead = lead_seconds(audio, 'ffmpeg')
            mpg123_lead = lead_seconds(audio, 'mpg123')
            with torch.inference_mode():
                song = listener.listen(audio)
            record.update(ffmpeg_lead_s=ffmpeg_lead, mpg123_lead_s=mpg123_lead,
                listener_song_ms=float(song.song_ms), listener_seconds=song.seconds,
                features=stats(song.features), segments=stats(song.segments),
                beat_probability=stats(song.features[:, -2]),
                downbeat_probability=stats(song.features[:, -1]))
            if not np.isfinite(song.features).all() or not np.isfinite(song.segments).all():
                raise ValueError('nonfinite audio features or segments')
            if (song.features.ndim != 2 or song.features.shape[1] != 130
                    or song.segments.ndim != 2 or song.segments.shape[1] != 2
                    or not len(song.segments) or np.any(np.diff(song.segments[:, 0]) <= 0)
                    or np.any(song.segments[:, 1] <= 0)):
                raise ValueError('invalid feature or timing geometry')
            decoded_segments = song.segments.copy()
            decoded_segments[:, 0] -= ffmpeg_lead * 1000
            relative = Path(plan['array_dir']) / (entry['chart_id'].replace(':', '-') + '.npz')
            if (ROOT / relative).exists():
                raise FileExistsError('previous audio features exist')
            np.savez_compressed(ROOT / relative, features=song.features,
                native_segments_ms=song.segments, decoded_segments_ms=decoded_segments,
                decoded_song_ms=np.array(song.song_ms - mpg123_lead * 1000))
            record.update(complete=True, arrays=str(relative), arrays_sha256=sha(ROOT / relative))
        except Exception as exc:
            # Keep the exact failed song in the denominator. Never replace a
            # difficult file or silently retry with target-derived timing.
            record.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        record['wall_s'] = time.monotonic() - t0
        result['songs'].append(record)
        save()
        print(json.dumps({k: record[k] for k in ('chart_id', 'complete', 'wall_s')}), flush=True)
    result.update(complete=True, success_count=sum(s['complete'] for s in result['songs']),
        failed_count=sum(not s['complete'] for s in result['songs']),
        wall_s=time.time() - start, ended_at_unix=time.time())
    save()
    print(json.dumps({k: result[k] for k in ('complete', 'success_count', 'failed_count', 'wall_s')}))


if __name__ == '__main__':
    main()
