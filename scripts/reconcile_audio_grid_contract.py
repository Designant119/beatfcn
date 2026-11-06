"""Apply the pinned external generator's label-free grid contract to saved audio predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generator_segments(raw_segments_ms, native_song_ms, ffmpeg_lead_s):
    # Listener returns pre-correction segments. Both released generation entry
    # points call this exact function before constructing note rows. Its fixed
    # lag/tempo rules are release parameters, not fitted to our target charts.
    from ensomi_model.audio_rows.grid import corrected_grid
    _, native = corrected_grid(raw_segments_ms, native_song_ms)
    decoded = native[:, :2].copy()
    decoded[:, 0] -= ffmpeg_lead_s * 1000
    return native, decoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, digest in plan['coordinate_input_sha256'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('grid contract input drift: ' + path)
    result_path = ROOT / plan['coordinate_result']
    if result_path.exists():
        raise FileExistsError('coordinate conversion already exists')
    inference = json.loads((ROOT / plan['result']).read_text())
    if not inference['complete']:
        raise ValueError('unfinished audio inference')
    result = dict(id='D48-coordinate-contract', complete=False, created_at_unix=time.time(),
        inference_sha256=sha(ROOT / plan['result']), plan_sha256=sha(args.plan),
        label_inputs=False, additional_model_forwards=0, model_updates=0, songs=[])
    for record in inference['songs']:
        if not record['complete']:
            continue
        if sha(ROOT / record['arrays']) != record['arrays_sha256']:
            raise ValueError('saved predictions changed')
        a = np.load(ROOT / record['arrays'], allow_pickle=False)
        native, decoded = generator_segments(a['native_segments_ms'],
            record['listener_song_ms'], record['ffmpeg_lead_s'])
        result['songs'].append(dict(chart_id=record['chart_id'],
            source_arrays_sha256=record['arrays_sha256'],
            raw_segments_ms=a['native_segments_ms'].tolist(),
            generator_native_segments_ms=native.tolist(), decoded_segments_ms=decoded.tolist(),
            decoded_duration_us=round(float(a['decoded_song_ms']) * 1000)))
    result.update(complete=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(converted=len(result['songs']), additional_model_forwards=0)))


if __name__ == '__main__':
    main()
