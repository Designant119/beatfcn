"""Assemble the D54 song panel using the frozen D48 feature and clock contract."""
import argparse
from collections import Counter
from pathlib import Path
import json
import time

import numpy as np

from scripts.prepare_action_state_batch import sha, stats


ROOT = Path(__file__).resolve().parents[1]


def assemble_inputs(rows):
    local, global_audio, global_times = [], [], []
    for row in rows:
        path = ROOT / row['source_features']
        assert sha(path) == row['source_features_sha256'], row['id']
        with np.load(path, allow_pickle=False) as arrays:
            audio = arrays['features'].astype(np.float32)
        assert audio.ndim == 2 and audio.shape[1] == 130 and np.isfinite(audio).all()
        start = row['source_clip_start_us']
        assert start % 10_000 == 0
        excerpt = audio[start // 10_000:start // 10_000 + 3200]
        assert excerpt.shape == (3200, 130), row['id']
        local.append(excerpt)
        boundaries = np.linspace(0, len(audio), 65, dtype=np.int64)
        assert np.all(np.diff(boundaries) > 0)
        global_audio.append(np.stack([audio[a:b].mean(axis=0, dtype=np.float64)
            for a, b in zip(boundaries, boundaries[1:])]).astype(np.float32))
        global_times.append(np.array([(a + b - 1) * 5000 + 20_000 - start
            for a, b in zip(boundaries, boundaries[1:])], dtype=np.int64))
    n = len(rows)
    result = dict(audio=np.stack(local), song_audio=np.stack(global_audio),
        song_times_us=np.stack(global_times),
        audio_times_us=np.broadcast_to(np.arange(3200, dtype=np.int64) * 10_000 + 20_000, (n, 3200)).copy(),
        mode=np.array([r['mode'] == 'phira' for r in rows], dtype=np.int64),
        core_bounds_us=np.tile(np.array([8_000_000, 16_000_000], dtype=np.int64), (n, 1)),
        duration_us=np.full(n, 32_000_000, dtype=np.int64),
        condition_values=np.zeros((n, 8), dtype=np.float32),
        condition_known=np.zeros((n, 8), dtype=bool))
    result['audio_valid'] = result['audio_times_us'] < result['duration_us'][:, None]
    return result


def source_targets(row):
    chart = json.loads((ROOT / row['canonical']).read_text())
    assert sha(ROOT / row['canonical']) == row['canonical_sha256']
    shift, clip, start = row['clock_shift_us'], row['source_clip_start_us'], row['source_core_start_us']
    expected = Counter((e['onset_us'] + shift - clip, e['end_us'] + shift - clip, e['type'],
        e['location']['lane'] if row['mode'] == 'mania_4k' else -1)
        for e in chart['events'] if start <= e['onset_us'] + shift < start + 8_000_000)
    actual = Counter((e['onset_us'], e['end_us'], e['type'], e['lane']) for e in row['target_events'])
    assert actual == expected, row['id']
    assert all(8_000_000 <= e[0] < 16_000_000 and e[1] < 32_000_000 for e in actual)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, expected in plan['input_sha256'].items():
        assert sha(ROOT / path) == expected, path
    output = ROOT / plan['output_dir']
    destination = ROOT / plan['result']
    assert not output.exists() and not destination.exists()
    started = time.time()
    panel = json.loads((ROOT / plan['source_panel']).read_text())
    inference = json.loads((ROOT / plan['audio_inference']).read_text())
    assert inference['complete'] and inference['success_count'] == 30 and inference['failed_count'] == 0
    features = {r['chart_id']: r for r in inference['songs']}
    rows = []
    for part, items in [('train', panel['rows']), ('heldout', panel['heldout_rows'])]:
        for original in items:
            row = dict(original, partition=part)
            if row.get('origin') == 'new_independent_song':
                feature = features[row['chart_id']]
                assert feature['audio_sha256'] == row['audio_sha256']
                row.update(source_features=feature['arrays'], source_features_sha256=feature['arrays_sha256'])
            source_targets(row)
            rows.append(row)
    assert len(rows) == 41 and len({r['audio_sha256'] for r in rows}) == 41
    inputs = assemble_inputs(rows)
    assert all(np.isfinite(a).all() for a in inputs.values())
    old = json.loads((ROOT / plan['original_manifest']).read_text())
    original_rows = {r['id']: i for i, r in enumerate(old['rows'])}
    identical = []
    with np.load(ROOT / plan['original_inputs'], allow_pickle=False) as before:
        assert set(inputs) == set(before.files)
        for i, row in enumerate(rows):
            if row['id'] not in original_rows:
                continue
            j = original_rows[row['id']]
            assert all(np.array_equal(inputs[k][i], before[k][j]) for k in inputs), row['id']
            assert all(row[k] == v for k, v in old['rows'][j].items()), row['id']
            identical.append(row['id'])
    assert len(identical) == 11
    output.mkdir(parents=True)
    np.savez_compressed(output / 'inputs.npz', **inputs)
    manifest = dict(id='d55-independent-song-tensors', rows=rows, inputs='inputs.npz',
        frame_center_offset_us=20_000, frame_hop_us=10_000,
        condition_fields=old['condition_fields'], training_ready=False,
        train_ids=[r['id'] for r in rows if r['partition'] == 'train'],
        heldout_ids=[r['id'] for r in rows if r['partition'] == 'heldout'],
        scope='Finite diagnosis only. Labels remain outside the blank-generation input arrays.')
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    report = dict(id='D55-tensors', complete=True, passed=True, model_updates=0, new_cloud_spend_usd=0,
        plan_sha256=sha(args.plan), rows=len(rows), train_songs=36, heldout_songs=5,
        source_targets_exact=41, original_inputs_labels_exact=identical,
        tensor_statistics={k: stats(v) for k, v in inputs.items()}, wall_s=time.time() - started,
        artifacts_sha256={str(p.relative_to(ROOT)): sha(p) for p in output.iterdir()})
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ['passed', 'rows', 'train_songs', 'heldout_songs', 'wall_s']}))


if __name__ == '__main__':
    main()
