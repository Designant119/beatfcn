"""Segment diagnostics complement atomic drag metrics; temporal clusters are not gestures."""
import numpy as np
from .sample import match_f1


def score_drag_segments(pred, truth, cases):
    rows = []
    for case in cases:
        if case['semantic_status'] not in ('temporal_candidate', 'human_verified'):
            raise ValueError('drag case needs an explicit evidence status')
        a, b = case['span_us']
        if not a < b:
            raise ValueError('invalid drag case interval')
        p = [e for e in pred if e.type == 'drag' and a <= e.onset_us < b]
        t = [e for e in truth if e.type == 'drag' and a <= e.onset_us < b]
        tp, npred, ntrue, _ = match_f1([e.onset_us for e in p], [e.onset_us for e in t], 20_000)
        row = {'case_id': case['case_id'], 'semantic_status': case['semantic_status'],
               'pred': npred, 'true': ntrue, 'atomic_f1_20ms': 2 * tp / (npred + ntrue) if npred + ntrue else None,
               'count_ratio': npred / ntrue if ntrue else None, 'start_error_ms': None, 'end_error_ms': None,
               'longest_gap_ms': None, 'operation_quality': 'not_measured_without_actions',
               'visual_quality': 'not_measured', 'musical_validity': 'requires_context_and_human_calibration'}
        if p and t:
            row['start_error_ms'] = (min(e.onset_us for e in p) - min(e.onset_us for e in t)) / 1000
            row['end_error_ms'] = (max(e.onset_us for e in p) - max(e.onset_us for e in t)) / 1000
        onsets = sorted({e.onset_us for e in p})
        if len(onsets) > 1:
            row['longest_gap_ms'] = float(np.diff(onsets).max() / 1000)
        rows.append(row)
    return {'schema': 'drag_segment_diagnostics_v1', 'cases': rows,
            'policy': 'No gesture truth inferred from IOI; multiplicity retained; missing submetrics remain missing.'}
