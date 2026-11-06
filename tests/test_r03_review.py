import copy
import unittest

from scripts.review_r03 import compare


class R03ReviewTests(unittest.TestCase):
    def test_control_failure_or_type_regression_cannot_be_hidden_by_mean_f1(self):
        plan = {'control_gate': {'fields_required': 13}, 'quality_gate': {
            'macro_full_note_f1_gain_over_initial_at_least': .03,
            'macro_full_note_f1_gain_over_matched_density_arm_at_least': .02,
            'max_per_mode_type_recall_drop_from_initial': .03,
            'min_non_tap_prediction_count_ratio_to_initial_if_present': .6}}
        before = dict(complete=True, macro_full_note_f1=.2, types={'phira:hold': {'recall': .5, 'pred': 10}})
        candidate = dict(complete=True, macro_full_note_f1=.4, types=copy.deepcopy(before['types']),
                         passed_fields=13, all_requests_complete_legal=True)
        good = compare(before, before, candidate, plan)
        self.assertTrue(good['short_window_gate_passed'])
        self.assertFalse(good['product_promotion'])
        for change in [dict(passed_fields=12), dict(all_requests_complete_legal=False),
                       dict(types={'phira:hold': {'recall': .4, 'pred': 10}}),
                       dict(types={'phira:hold': {'recall': .5, 'pred': 1}})]:
            self.assertFalse(compare(before, before, {**candidate, **change}, plan)['short_window_gate_passed'])


if __name__ == '__main__':
    unittest.main()
