import unittest

import torch

from scripts.audit_local_candidate_audio import AUDIO_KEYS, swap_audio


class AudioSwapTest(unittest.TestCase):
    def test_every_audio_stream_clock_and_mask_moves_without_condition_or_reference(self):
        inputs = {key: torch.arange(12).reshape(4, 3) for key in AUDIO_KEYS}
        inputs.update(mode=torch.ones(4), controls=torch.arange(16).reshape(4, 4),
                      reference=torch.arange(8).reshape(4, 2))
        original = {key:value.clone() for key,value in inputs.items()}
        donors = [1,2,3,0]
        altered = swap_audio(inputs, donors)
        for key in inputs:
            torch.testing.assert_close(altered[key], inputs[key][donors] if key in AUDIO_KEYS else inputs[key])
            torch.testing.assert_close(inputs[key], original[key])
        altered['controls'][0,0] = -999
        torch.testing.assert_close(inputs['controls'], original['controls'])

    def test_self_donors_and_nonpermutations_rejected(self):
        for donors in ([0,1,2,3],[1,1,3,0],[1,2,3]):
            with self.subTest(donors=donors), self.assertRaises(ValueError):
                swap_audio({'mode':torch.ones(4)}, donors)
