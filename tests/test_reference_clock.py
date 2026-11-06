from copy import deepcopy
import json
import unittest

from beatfcn.events.canonical import Event, LaneLoc
from beatfcn.model.reference_clock import benchmark_evidence, reference_clock


def row(residual=25, applied=0):
    return dict(chart_id='synthetic', mode='mania_4k', offset_decision=dict(c_ms=14,
        r_ms=residual, shift_ms=applied, action='shift' if applied else 'keep'),
        offset=dict(status='ok', conf=3.5, edge=False))


class ReferenceClockTests(unittest.TestCase):
    def test_retained_residual_is_flagged_without_changing_prediction_or_reference(self):
        truth = [Event('t', 100000, 100000, 'tap', LaneLoc(0))]
        pred = [Event('p', 125000, 125000, 'tap', LaneLoc(0))]
        before = deepcopy((pred, truth))
        result = benchmark_evidence(pred, truth, 'mania_4k', row())
        self.assertEqual(result['timing_f1_by_tolerance_ms']['20']['f1'], 0)
        self.assertEqual(result['timing_f1_by_tolerance_ms']['25']['f1'], 1)
        self.assertIn('remaining_residual_exceeds_match_tolerance', result['reference_clock']['flags'])
        self.assertEqual((pred, truth), before)
        self.assertFalse(result['automatic_correction_applied'])

    def test_applied_residual_is_not_counted_twice(self):
        result = reference_clock(row(42, 42))
        self.assertEqual(result['model_view_shift_ms'], 56)
        self.assertEqual(result['unapplied_residual_ms'], 0)
        self.assertEqual(result['status'], 'metadata_within_tolerance')
        self.assertFalse(result['within_song_stability_verified'])

    def test_missing_and_invalid_metadata_never_become_zero_uncertainty(self):
        for value in (None, float('inf'), float('nan'), True):
            item = row(); item['offset_decision']['r_ms'] = value
            result = reference_clock(item)
            self.assertEqual(result['status'], 'uncertain')
            self.assertIsNone(result['unapplied_residual_ms'])
            json.dumps(result, allow_nan=False)
        self.assertIn('estimator_confidence_unknown', reference_clock({})['flags'])

    def test_type_collapse_and_chord_loss_remain_visible_at_perfect_timing(self):
        truth = [Event('a',100000,200000,'hold',LaneLoc(0)),
                 Event('b',100000,200000,'hold',LaneLoc(1))]
        pred = [Event('p',100000,100000,'tap',LaneLoc(0))]
        result = benchmark_evidence(pred, truth, 'mania_4k', row(0))
        self.assertEqual(result['timing_f1_by_tolerance_ms']['20']['f1'], 1)
        self.assertEqual(result['reference_edit_proxy']['full_note']['matched'], 0)
        self.assertEqual(result['reference_edit_proxy']['full_note']['true'], 2)
        self.assertLess(result['reference_edit_proxy']['full_note']['saved_fraction'], 0)
        self.assertFalse(result['product_acceptance'])

    def test_empty_chart_and_wrong_metadata_do_not_claim_utility(self):
        truth = [Event('a',100000,100000,'tap',LaneLoc(0))]
        result = benchmark_evidence([], truth, 'mania_4k', row(0))
        self.assertEqual(result['timing_f1_by_tolerance_ms']['20']['f1'], 0)
        self.assertEqual(result['reference_edit_proxy']['full_note']['saved_fraction'], 0)
        with self.assertRaises(ValueError):
            benchmark_evidence([], truth, 'phira', row(0))
