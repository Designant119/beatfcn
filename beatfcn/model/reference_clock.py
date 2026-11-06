"""Reference uncertainty and reconstruction diagnostics, never model acceptance.

These helpers do not shift events, change the primary 20 ms score or infer user
conditions from a hidden reference. A metadata offset is an estimate, not truth.
"""
import math

from .edit_utility import edit_utility
from .sample import match_f1

TIMING_TOLERANCES_MS = (5, 10, 20, 25, 30, 50)


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def reference_clock(row, *, tolerance_ms=20):
    if not _finite_number(tolerance_ms) or tolerance_ms < 0:
        raise ValueError('timing tolerance must be finite and nonnegative')
    decision = row.get('offset_decision') or {}
    estimator = row.get('offset') or {}
    required = ('c_ms', 'r_ms', 'shift_ms')
    missing = [key for key in required if decision.get(key) is None]
    invalid = [key for key in required if decision.get(key) is not None and
               not _finite_number(decision[key])]
    flags = []
    if missing:
        flags.append('missing_clock_metadata')
    if invalid:
        flags.append('invalid_clock_metadata')
    residual = shift = None
    if not missing and not invalid:
        residual = decision['r_ms'] - decision['shift_ms']
        shift = decision['c_ms'] + decision['shift_ms']
        if abs(residual) > tolerance_ms:
            flags.append('remaining_residual_exceeds_match_tolerance')
    if decision.get('action') not in ('keep', 'shift'):
        flags.append('reference_clock_not_accepted_by_ingest')
    if row.get('offset_disagreement'):
        flags.append('other_charts_disagree_on_clock')
    confidence = estimator.get('conf')
    if not _finite_number(confidence):
        confidence = None
        flags.append('estimator_confidence_unknown')
    elif confidence < 3:
        flags.append('estimator_low_confidence')
    if estimator.get('edge'):
        flags.append('estimator_search_edge')
    if estimator.get('status') != 'ok':
        flags.append('offset_estimate_unavailable')
    return dict(schema='beatfcn.reference_clock.v1', chart_id=row.get('chart_id'),
        timing_tolerance_ms=tolerance_ms, model_view_shift_ms=shift,
        unapplied_residual_ms=residual, estimator_confidence=confidence,
        status='uncertain' if flags else 'metadata_within_tolerance', flags=flags,
        missing_fields=missing, invalid_fields=invalid, within_song_stability_verified=False,
        automatic_correction_applied=False,
        limitation='Ingest metadata estimates, not human timing ground truth; within-song stability is unverified.')


def benchmark_evidence(pred, truth, mode, row):
    if row.get('mode') != mode:
        raise ValueError('reference metadata mode differs from evaluated events')
    predicted_heads = sorted({e.onset_us for e in pred})
    true_heads = sorted({e.onset_us for e in truth})
    timing = {}
    for tolerance in TIMING_TOLERANCES_MS:
        matched = match_f1(predicted_heads, true_heads, tolerance*1000)[0]
        denominator = len(predicted_heads)+len(true_heads)
        timing[str(tolerance)] = dict(matched=matched, predicted=len(predicted_heads),
            reference=len(true_heads), f1=2*matched/denominator if denominator else 1.)
    return dict(schema='beatfcn.benchmark_evidence.v1', reference_clock=reference_clock(row),
        timing_f1_by_tolerance_ms=timing,
        reference_edit_proxy=edit_utility(pred, truth, mode),
        primary_score_changed=False, automatic_correction_applied=False,
        product_acceptance=False)
