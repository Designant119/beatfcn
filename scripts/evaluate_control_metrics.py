"""Score generated songs with control-response metrics (CPU only).

Reads per-song generation JSONs, joins them to the frozen tape targets, and
reports curve following (when a requested curve is embedded), note-type
distribution, empty-readout rate, and the existing event-level F1s.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from beatfcn.events.canonical import Event, LaneLoc, LineLoc
from beatfcn.model.evaluation import score_events
from beatfcn.train.control_metrics import curve_metrics, readout_metrics, type_metrics

ROOT = Path(__file__).resolve().parents[1]
TYPES = ('tap', 'hold', 'drag', 'flick')


def load_songs(path):
    path = Path(path)
    if path.is_dir():
        return [json.loads(file.read_text()) for file in sorted(path.glob('*.json'))]
    value = json.loads(path.read_text())
    for key in ('songs', 'results', 'generation'):
        if isinstance(value, dict) and isinstance(value.get(key), list):
            return value[key]
    return [value]


def tape_charts(tape_path):
    tape = json.loads(Path(tape_path).read_text())
    return {chart['chart_id']: chart for chart in tape['charts']}


def target_events(chart):
    with np.load(ROOT / chart['target_path'], allow_pickle=False) as archive:
        return [dict(onset_us=int(onset), end_us=int(end), type=TYPES[int(kind)],
                     lane=int(lane))
                for onset, end, kind, lane in zip(archive['onset_us'], archive['end_us'],
                                                  archive['kind'], archive['lane'])]


def as_events(rows, mode):
    result = []
    for index, row in enumerate(rows):
        location = (LaneLoc(int(row.get('lane', 0)))
                    if mode == 'mania_4k' else LineLoc('0', 'above', 0.))
        result.append(Event(f'record:{index}', int(row['onset_us']), int(row.get('end_us', row['onset_us'])),
                            row['type'], location))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation', type=Path, required=True)
    parser.add_argument('--tape', default='reports/d94_expanded_target_tape_v1.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    charts = tape_charts(ROOT / args.tape)
    songs = load_songs(args.generation)
    records = []
    for song in songs:
        chart = charts.get(song.get('chart_id'))
        if chart is None:
            records.append(dict(chart_id=song.get('chart_id'), skipped='chart_not_in_tape'))
            continue
        truth = target_events(chart)
        predicted = as_events(song.get('events', []), song.get('mode', chart['mode']))
        truth_events = as_events(truth, song.get('mode', chart['mode']))
        scores = score_events(predicted, truth_events, song.get('mode', chart['mode']))
        types = type_metrics(predicted, truth_events)
        record = dict(chart_id=song.get('chart_id'), complete=song.get('complete'),
            note_f1_50ms=scores['note_50ms']['f1'],
            onset_f1_50ms=scores['onset_50ms']['f1'],
            density_ratio=scores['density_ratio'],
            types=types, readout=readout_metrics([predicted]))
        if song.get('requested_curve') is not None and song.get('curve_start_us') is not None:
            record['curve'] = curve_metrics(predicted, song['requested_curve'],
                int(song['curve_start_us']), cell_us=int(song.get('curve_cell_us', 500_000)),
                scale=float(song.get('curve_scale', 16.0)))
        if song.get('cores'):
            cores = []
            for core in song['cores']:
                if core.get('requested_curve') is None:
                    continue
                start = int(core['curve_start_us'])
                end = int(core['end_us'])
                local = [row for row in predicted if start <= row.onset_us < end]
                cores.append(dict(start_us=start, end_us=end, **curve_metrics(
                    local, core['requested_curve'], start,
                    cell_us=int(core.get('curve_cell_us', 500_000)),
                    scale=float(core.get('curve_scale', 16.0)))))
            if cores:
                record['cores'] = cores
        records.append(record)
    scored = [r for r in records if 'note_f1_50ms' in r and r['note_f1_50ms'] is not None]
    def mean(key):
        values = [r[key] for r in scored if r.get(key) is not None]
        return float(np.mean(values)) if values else None
    summary = dict(songs=len(records), scored=len(scored),
        mean_note_f1_50ms=mean('note_f1_50ms'),
        mean_onset_f1_50ms=mean('onset_f1_50ms'),
        mean_density_ratio=mean('density_ratio'),
        nonempty_rate=float(np.mean([r['readout']['nonempty_rate'] for r in scored])) if scored else None,
        mean_type_coverage=float(np.mean([r['types']['macro_type_recall'] for r in scored])) if scored else None,
        mean_type_js=float(np.mean([r['types']['js_divergence'] for r in scored])) if scored else None,
        curve=[r['curve'] for r in scored if 'curve' in r],
        core_curve=[c for r in scored for c in r.get('cores', [])])
    result = dict(generation=str(args.generation), tape=args.tape,
                  summary=summary, songs=records)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
