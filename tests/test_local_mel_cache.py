import unittest

import numpy as np
import torch

from beatfcn.audio.local_mel_cache import complete_song_mel, legacy_prefix_comparison
from beatfcn.ingest.pipeline import mel10


class LocalMelCacheTests(unittest.TestCase):
    def test_whole_cells_preserve_production_values_exactly(self):
        torch.set_num_threads(2)
        wave = np.random.default_rng(7).normal(0, .1, 960*8).astype(np.float32)
        result = complete_song_mel(wave)
        np.testing.assert_array_equal(result['mel'], mel10(wave))
        self.assertTrue(result['mel'].flags.c_contiguous)
        self.assertTrue(result['valid'].all())
        self.assertEqual(result['padding_samples'], 0)

    def test_final_partial_frame_retains_audio_and_truthful_mask(self):
        wave = np.zeros(960*8+137, dtype=np.float32)
        wave[-137:] = np.random.default_rng(8).normal(0, .1, 137)
        result = complete_song_mel(wave)
        self.assertEqual(result['mel'].shape, (36, 128))
        self.assertEqual(result['valid'].sum(), 33)
        self.assertEqual(result['frame_start_samples'][32], 7680)
        self.assertEqual(result['frame_end_samples'][32], len(wave))
        np.testing.assert_array_equal(result['frame_start_samples'][33:], len(wave))
        np.testing.assert_array_equal(result['frame_end_samples'][33:], len(wave))
        silence = complete_song_mel(np.zeros_like(wave))
        self.assertFalse(np.array_equal(result['mel'][32], silence['mel'][32]))
        self.assertTrue(legacy_prefix_comparison(mel10(wave), result)['interior_exact_equal'])

    def test_reject_invalid_waveform_and_wrong_legacy_clock(self):
        for bad in (np.zeros((2, 960)), np.zeros(392), np.full(960, np.nan)):
            with self.assertRaises(ValueError):
                complete_song_mel(bad)
        result = complete_song_mel(np.zeros(960, dtype=np.float32))
        with self.assertRaises(ValueError):
            legacy_prefix_comparison(np.zeros((8, 128)), result)
