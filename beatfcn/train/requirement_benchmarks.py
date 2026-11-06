"""Requirement benchmarks: continuation, condition response, human similarity."""
import numpy as np

from beatfcn.events.canonical import Event, LaneLoc, LineLoc
from beatfcn.model.evaluation import quality_metrics, score_events
from beatfcn.train.control_metrics import density_request_metrics, readout_metrics, type_metrics

TASK_TYPES = ('free', 'curve', 'continuation', 'infill')


def as_events(rows, mode):
    result = []
    for index, row in enumerate(rows):
        location = (LaneLoc(int(row.get('lane', 0)))
                    if mode == 'mania_4k' else LineLoc('0', 'above', 0.))
        result.append(Event(f'bench:{index}', int(row['onset_us']),
            int(row.get('end_us', row['onset_us'])), row['type'], location))
    return result


def _quality(pred, truth, mode, start_us, end_us):
    try:
        return quality_metrics(pred, truth,
            [dict(start_us=start_us, end_us=end_us, state='editable')], mode)
    except (ValueError, KeyError):
        return None


def requirement_metrics(predicted, truth, mode, start_us, end_us, *,
                        requested_curve=None, task='free'):
    if task not in TASK_TYPES:
        raise ValueError('Unknown benchmark task')
    pred_events = as_events(predicted, mode)
    truth_events = as_events(truth, mode)
    scores = score_events(pred_events, truth_events, mode)
    result = dict(task=task, start_us=start_us, end_us=end_us,
        note_f1_50ms=scores['note_50ms']['f1'],
        onset_f1_50ms=scores['onset_50ms']['f1'],
        onset_f1_10ms=scores['onset_10ms']['f1'],
        density_ratio=scores['density_ratio'],
        types=type_metrics(pred_events, truth_events),
        readout=readout_metrics([pred_events]),
        human_similarity=_quality(pred_events, truth_events, mode, start_us, end_us))
    if requested_curve:
        result['curve'] = density_request_metrics(pred_events, requested_curve)
    return result


def summarize(records):
    records = list(records)
    def mean(key):
        values = [r[key] for r in records if r.get(key) is not None]
        return float(np.mean(values)) if values else None
    def nested(path):
        values = []
        for record in records:
            value = record
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if value is not None:
                values.append(value)
        return float(np.mean(values)) if values else None
    return dict(tasks=len(records),
        mean_note_f1_50ms=mean('note_f1_50ms'),
        mean_onset_f1_50ms=mean('onset_f1_50ms'),
        mean_density_ratio=mean('density_ratio'),
        nonempty_rate=nested(('readout', 'nonempty_rate')),
        mean_type_coverage=nested(('types', 'macro_type_count_coverage')),
        mean_type_js=nested(('types', 'js_divergence')),
        mean_onset_error_50ms=nested(('human_similarity', 'onset_error_50ms', 'match_rate')),
        mean_ioi_kl=nested(('human_similarity', 'ioi_kl', 'kl_nats')),
        mean_density_pearson=nested(('human_similarity', 'density_curve', 'pearson_r')))
