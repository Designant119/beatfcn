import unittest

import torch

from scripts.audit_local_sequence_conditions import field_summary
from scripts.audit_local_sequence_audio import AUDIO_KEYS, swap_audio


class ConditionFieldGateTest(unittest.TestCase):
    @staticmethod
    def row(delta=3., hits=(True, True), identical=False):
        return dict(complete=True, identical_token_stream=identical, direction_delta=delta,
                    minimum_delta=2., in_band=list(hits), independent_legality=[[], []])

    def test_direction_and_both_band_support_required(self):
        rows = [self.row(), self.row(), self.row(), self.row(delta=0, hits=(False, False))]
        self.assertTrue(field_summary(rows)['sentinel_passed'])
        rows[2]['in_band'] = [True, False]
        self.assertFalse(field_summary(rows)['sentinel_passed'])

    def test_missing_and_empty_identical_outputs_cannot_pass(self):
        self.assertFalse(field_summary([self.row()]*3)['sentinel_passed'])
        self.assertFalse(field_summary([self.row(delta=None, identical=True)]*4)['sentinel_passed'])

    def test_illegality_overrides_scalar_response(self):
        rows = [self.row() for _ in range(4)]
        rows[0]['independent_legality'][1] = [{'code':'outside_requested_scope'}]
        self.assertFalse(field_summary(rows)['sentinel_passed'])

    def test_plain_model_audio_swap_preserves_non_audio_inputs(self):
        inputs = {key:torch.arange(12).reshape(4,3) for key in AUDIO_KEYS}
        inputs.update(mode=torch.ones(4),controls=torch.arange(24).reshape(4,6),reference=torch.ones(4,1,16))
        swapped = swap_audio(inputs,[1,2,3,0])
        for key,value in inputs.items():
            self.assertTrue(torch.equal(swapped[key],value[[1,2,3,0]] if key in AUDIO_KEYS else value))


if __name__ == '__main__':
    unittest.main()
