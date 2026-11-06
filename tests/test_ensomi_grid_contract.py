"""Run in the isolated pinned external environment; no weights or inference needed."""
import unittest
import importlib.util
import numpy as np
from scripts.reconcile_audio_grid_contract import generator_segments


@unittest.skipUnless(importlib.util.find_spec('ensomi_model'), 'requires isolated pinned external environment')
class EnsomiGridContractTests(unittest.TestCase):
    def test_release_lag_and_decoder_lead_are_both_applied(self):
        native, decoded = generator_segments([[100, 500]], 10000, .012)
        np.testing.assert_allclose(native, [[70, 500, 4]])
        np.testing.assert_allclose(decoded, [[58, 500]])

    def test_fast_tempo_is_folded_to_release_canonical_beat(self):
        native, decoded = generator_segments([[100, 300]], 10000, 0)
        np.testing.assert_allclose(native, [[70, 600, 4]])
        np.testing.assert_allclose(decoded, [[70, 600]])

    def test_octave_equivalent_segments_use_the_longest_segment(self):
        native, decoded = generator_segments([[100, 500], [1100, 250]], 5000, 0)
        np.testing.assert_allclose(native, [[1070, 500, 4]])
        np.testing.assert_allclose(decoded, [[1070, 500]])


if __name__ == '__main__':
    unittest.main()
