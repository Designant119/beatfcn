"""Independent receipt, integer-time reconstruction and coverage checks for D48."""
from bisect import bisect_left
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plan_path = ROOT / 'bench/rhythm-v2/d48-audio-only-time-coordinate-audit-v3.json'
    plan = json.loads(plan_path.read_text())
    report_path = ROOT / plan['audit_result']
    report = json.loads(report_path.read_text())
    panel = json.loads((ROOT / plan['panel']).read_text())
    inference = json.loads((ROOT / plan['result']).read_text())
    identities = dict(plan['inference_input_sha256'])
    identities.update(plan['coordinate_input_sha256'])
    identities.update(plan['audit_input_sha256'])
    for name, expected in identities.items():
        if sha(ROOT / name) != expected:
            raise AssertionError('identity mismatch: ' + name)
    assert report['complete'] and report['coordinate_protocol'] == 'released_generator_corrected_grid'
    assert report['plan_sha256'] == sha(plan_path)
    assert report['inference_sha256'] == sha(ROOT / plan['result'])
    assert report['coordinate_result_sha256'] == sha(ROOT / plan['coordinate_result'])
    assert inference['success_count'] == report['completed_charts'] == len(panel['rows']) == 12
    assert len({r['audio_sha256'] for r in panel['rows']}) == 12
    assert inference['model_updates'] == report['new_model_updates'] == 0
    assert not report['model_promotion'] and report['label_changes'] == 0
    records = {row['chart_id']: row for row in panel['rows']}
    checked = 0
    for item in report['songs']:
        assert sha(ROOT / item['residual_arrays']) == item['residual_arrays_sha256']
        stored = np.load(ROOT / item['residual_arrays'], allow_pickle=False)
        chart = json.loads((ROOT / records[item['chart_id']]['canonical']).read_text())
        assert np.array_equal(stored['native_onset_us'], [e['onset_us'] for e in chart['events']])
        assert np.array_equal(stored['native_end_us'], [e['end_us'] for e in chart['events']])
        for view in ('raw_native', 'frozen_model'):
            offset = 0 if view == 'raw_native' else item['frozen_model_shift_us']
            times = [int(t) + offset for t in stored['native_onset_us']]
            for grid_name in plan['subdivisions']:
                grid = stored[f'{view}_{grid_name}_grid_us'].tolist()
                predicted_errors = stored[f'{view}_{grid_name}_head_error_us'].tolist()
                errors = []
                for t, error in zip(times, predicted_errors):
                    index = bisect_left(grid, t)
                    candidates = grid[max(index - 1, 0): min(index + 1, len(grid))]
                    nearest = min(candidates, key=lambda value: (abs(value - t), value))
                    assert t - nearest == error
                    errors.append(t - nearest)
                    checked += 1
                score = item['views'][view]['grids'][grid_name]['metrics']['note_heads']
                assert score['n'] == len(times)
                assert abs(score['within_20ms'] - sum(abs(e) <= 20000 for e in errors) / len(errors)) < 1e-12
        # The serialization-only revision must have left all old computations
        # unchanged, even though that intermediate coordinate was later rejected.
        name = item['chart_id'].replace(':', '-') + '.npz'
        old = np.load(ROOT / 'artifacts/d48-audio-grid/residuals' / name, allow_pickle=False)
        repaired = np.load(ROOT / 'artifacts/d48-audio-grid/residuals-v2' / name, allow_pickle=False)
        assert old.files == repaired.files
        assert all(np.array_equal(old[k], repaired[k]) for k in old.files)
    for song in inference['songs']:
        assert song['features']['shape'][1] == 130
        assert song['features']['nan'] == song['features']['inf'] == 0
        for field in ('beat_probability', 'downbeat_probability'):
            assert 0 <= song[field]['min'] <= song[field]['max'] <= 1
        assert abs(song['features']['shape'][0] * 10 - song['listener_song_ms']) < 50
    p1_sha = sha(ROOT / 'beatfcn/model/p1.py')
    assert p1_sha == 'c86c6fafa5b9c1fc911903d95c3bc22ed1c56b7c4c8f425ab969c4e35bf99364'
    receipt = dict(complete=True, verified_at_unix=time.time(), identities=len(identities),
        charts=12, unique_audio=12, independent_nearest_point_checks=checked,
        serialization_revision_arrays_identical=True, feature_health_passed=True,
        label_changes=0, p1_sha256=p1_sha, model_promotion=False,
        source_sha256={str(report_path.relative_to(ROOT)): sha(report_path),
                       str(plan_path.relative_to(ROOT)): sha(plan_path)})
    (ROOT / 'reports/d48_verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
