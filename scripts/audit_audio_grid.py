"""Measure audio-only coordinate coverage without changing any chart labels."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from scripts.audio_grid_geometry import grid_points, residuals, error_summary, drag_burst_indices

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measured(times, grid, controls):
    error, nearest = residuals(times, grid)
    base = error_summary(error)
    shifted = [error_summary(residuals(times, points)[0]) for points in controls]
    base['phase_controls'] = shifted
    base['phase_excess_20ms'] = (float(base['within_20ms'] - np.mean([s['within_20ms'] for s in shifted]))
                                if len(times) else None)
    return base, error, nearest


def screen_passed(score, gate):
    if score['within_20ms'] is None or score['phase_excess_20ms'] is None:
        return False
    return bool(score['within_20ms'] >= gate['minimum_coverage_20ms']
                and score['phase_excess_20ms'] >= gate['minimum_phase_excess_20ms'])


def quantization_damage(onsets, ends, holds, grid):
    head = residuals(onsets, grid)[1]
    tail = residuals(ends[holds], grid)[1]
    distinct = len(np.unique(onsets))
    collapsed = distinct - len(np.unique(head))
    return dict(distinct_onsets=distinct, collapsed_distinct_onsets=collapsed,
        collapsed_distinct_onset_fraction=collapsed / distinct if distinct else None,
        holds=int(holds.sum()), nonpositive_holds=int(np.sum(tail <= head[holds])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, digest in plan['audit_input_sha256'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('audit input changed: ' + path)
    output = ROOT / plan['audit_result']
    if output.exists():
        raise FileExistsError('previous audit result exists')
    panel = json.loads((ROOT / plan['panel']).read_text())
    inference = json.loads((ROOT / plan['result']).read_text())
    if not inference['complete'] or inference['plan_sha256'] != plan.get('inference_plan_sha256', sha(args.plan)):
        raise ValueError('incomplete or mismatched inference')
    by_chart = {r['chart_id']: r for r in inference['songs']}
    coordinates = None
    if plan.get('coordinate_result'):
        coordinates = json.loads((ROOT / plan['coordinate_result']).read_text())
        if (not coordinates['complete'] or coordinates['inference_sha256'] != sha(ROOT / plan['result'])
                or coordinates['plan_sha256'] != sha(args.plan)):
            raise ValueError('coordinate conversion identity mismatch')
        coordinates = {r['chart_id']: r for r in coordinates['songs']}
    all_errors = defaultdict(list)
    all_shifted = defaultdict(lambda: [[], [], []])
    macro = defaultdict(list)
    report = dict(id='D48', complete=False, created_at_unix=time.time(),
        plan_sha256=sha(args.plan), inference_sha256=sha(ROOT / plan['result']),
        panel_sha256=sha(ROOT / plan['panel']), label_changes=0, new_model_updates=0,
        model_promotion=False, points_are_audio_only=True, songs=[], failures=[],
        coordinate_protocol='released_generator_corrected_grid' if coordinates is not None else 'listener_raw_intermediate',
        coordinate_result_sha256=sha(ROOT / plan['coordinate_result']) if coordinates is not None else None)
    arrays_dir = ROOT / plan['audit_array_dir']
    arrays_dir.mkdir(parents=True, exist_ok=True)
    for row in panel['rows']:
        source = by_chart.get(row['chart_id'])
        if not source or not source['complete']:
            report['failures'].append(dict(chart_id=row['chart_id'], source=source))
            continue
        if sha(ROOT / source['arrays']) != source['arrays_sha256']:
            raise ValueError('audio array drift')
        song = np.load(ROOT / source['arrays'], allow_pickle=False)
        segments = song['decoded_segments_ms']
        duration_us = round(float(song['decoded_song_ms']) * 1000)
        if coordinates is not None:
            conversion = coordinates[row['chart_id']]
            if conversion['source_arrays_sha256'] != source['arrays_sha256']:
                raise ValueError('coordinate source array mismatch')
            segments = np.array(conversion['decoded_segments_ms'], dtype=np.float64)
            if conversion['decoded_duration_us'] != duration_us:
                raise ValueError('coordinate duration mismatch')
        chart = json.loads((ROOT / row['canonical']).read_text())
        events = chart['events']
        if any(e.get('fake') for e in events):
            raise ValueError('fake in canonical playable events')
        head_native = np.array([e['onset_us'] for e in events], dtype=np.int64)
        end_native = np.array([e['end_us'] for e in events], dtype=np.int64)
        types = np.array([e['type'] for e in events])
        holds = types == 'hold'
        if np.any(end_native < head_native) or np.any(end_native[holds] <= head_native[holds]):
            raise ValueError('invalid canonical duration')
        burst = drag_burst_indices(events, **plan['drag_burst_proxy'])
        entry = {k: row[k] for k in ('chart_id', 'title', 'mode', 'tier', 'types', 'split',
            'native_difficulty', 'difficulty_evidence', 'measured_controls', 'offset_decision',
            'frozen_model_shift_us', 'fakes_excluded')}
        entry.update(duration_s=duration_us / 1e6, segments_ms=segments.tolist(),
            clock_lead_s=source['ffmpeg_lead_s'], views={},
            drag_burst_notes=len(burst), drag_burst_distinct_onsets=len(np.unique(head_native[burst])))
        detail = dict(native_onset_us=head_native, native_end_us=end_native,
                      types=types, drag_burst_indices=burst)
        for view, shift in [('raw_native', 0), ('frozen_model', row['frozen_model_shift_us'])]:
            heads, ends = head_native + shift, end_native + shift
            data = {'note_heads': heads, 'distinct_onset_groups': np.unique(heads),
                **{f'{t}_heads': heads[types == t] for t in ('tap', 'hold', 'flick', 'drag')},
                'hold_ends': ends[holds], 'drag_burst_heads': heads[burst],
                'drag_burst_distinct_onsets': np.unique(heads[burst])}
            view_record = dict(shift_us=shift, outside_audio_heads=int(np.sum((heads < 0) | (heads > duration_us))),
                outside_audio_hold_ends=int(np.sum((ends[holds] < 0) | (ends[holds] > duration_us))), grids={})
            for name, denominators in plan['subdivisions'].items():
                grid = grid_points(segments, duration_us, denominators)
                controls = [grid_points(segments, duration_us, denominators, phase_shift_us=offset)
                            for offset in plan['phase_controls_us']]
                metrics = {}
                for target, times in data.items():
                    score, error, nearest = measured(times, grid, controls)
                    metrics[target] = score
                    if not np.array_equal(nearest + error, times):
                        raise ValueError('coordinate residual reconstruction failed')
                    for group in ('all', row['mode'], row['mode'] + '/' + row['tier']):
                        key = (group, view, name, target)
                        all_errors[key].append(error)
                        macro[key].append(score)
                        for index, points in enumerate(controls):
                            all_shifted[key][index].append(residuals(times, points)[0])
                detail[f'{view}_{name}_grid_us'] = grid
                detail[f'{view}_{name}_head_error_us'] = residuals(heads, grid)[0]
                detail[f'{view}_{name}_hold_end_error_us'] = residuals(ends[holds], grid)[0]
                worst = np.argsort(-abs(detail[f'{view}_{name}_head_error_us']), kind='stable')[:8]
                bad_cases = [dict(event_id=events[i]['event_id'], type=events[i]['type'],
                    onset_us=int(heads[i]), end_us=int(ends[i]),
                    residual_us=int(detail[f'{view}_{name}_head_error_us'][i])) for i in worst]
                view_record['grids'][name] = dict(point_count=len(grid), points_per_second=len(grid) * 1e6 / duration_us,
                    metrics=metrics, hypothetical_snap_damage=quantization_damage(heads, ends, holds, grid),
                    worst_heads=bad_cases)
            entry['views'][view] = view_record
        relative = Path(plan['audit_array_dir']) / (row['chart_id'].replace(':', '-') + '.npz')
        if (ROOT / relative).exists():
            raise FileExistsError('previous audit residual arrays exist')
        np.savez_compressed(ROOT / relative, **detail)
        entry.update(residual_arrays=str(relative), residual_arrays_sha256=sha(ROOT / relative))
        report['songs'].append(entry)
    aggregates = {}
    for key, errors in all_errors.items():
        score = error_summary(np.concatenate(errors))
        shifted = [error_summary(np.concatenate(parts)) for parts in all_shifted[key]]
        score['phase_controls'] = shifted
        score['phase_excess_20ms'] = (score['within_20ms'] - float(np.mean([s['within_20ms'] for s in shifted]))
                                    if score['n'] else None)
        defined = [s for s in macro[key] if s['n']]
        score.update(charts_with_support=len(defined),
            chart_macro_within_20ms=float(np.mean([s['within_20ms'] for s in defined])) if defined else None,
            chart_macro_phase_excess_20ms=float(np.mean([s['phase_excess_20ms'] for s in defined])) if defined else None)
        aggregates['|'.join(key)] = score
    report['aggregates'] = aggregates
    gate = plan['coordinate_screen']
    screens = []
    for row in report['songs']:
        score = row['views'][gate['clock']]['grids'][gate['grid']]['metrics'][gate['target']]
        screens.append(dict(chart_id=row['chart_id'], mode=row['mode'],
            passed=screen_passed(score, gate)))
    mode_pass = {mode: sum(s['passed'] for s in screens if s['mode'] == mode) for mode in ('mania_4k', 'phira')}
    report['coordinate_screen'] = dict(per_chart=screens, mode_pass=mode_pass,
        passed=(not report['failures'] and sum(mode_pass.values()) >= gate['minimum_charts']
                and min(mode_pass.values()) >= gate['minimum_per_mode']),
        scope='Engineering screen for an auxiliary coordinate; not a generation quality gate or a hard grid approval.')
    report.update(complete=True, expected_charts=len(panel['rows']), completed_charts=len(report['songs']),
        all_integer_microsecond_roundtrips_exact=True, model_quality_gain_proven=False)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(completed=report['completed_charts'], failures=len(report['failures']),
                         screen=report['coordinate_screen']), ensure_ascii=False))


if __name__ == '__main__':
    main()
