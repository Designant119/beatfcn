import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from beatfcn.train.full_database import commit_feature, existing_feature_receipt, feature_payloads


class FullDatabaseFeatureTests(unittest.TestCase):
    class Encoder:
        revision = 'test-revision'
        spec = None

        def encode(self, wave):
            samples = len(wave)
            frames = (samples + 959) // 960
            values = np.zeros((2, frames, 1024), dtype=np.float16)
            values[:, :, 0] = np.linspace(0, 1, frames * 2, dtype=np.float16).reshape(2, frames)
            return dict(layers=values, times_s=np.arange(frames) * .04 + .015,
                intervals_s=np.stack((np.arange(frames) * .04,
                    np.minimum((np.arange(frames) + 1) * .04, samples / 24000)), axis=-1),
                metadata=dict(revision=self.revision, real_samples=samples))

    def test_payload_clock_commit_and_existing_receipt_are_lossless(self):
        wave = np.linspace(-1, 1, 960, dtype=np.float32)
        encoder = self.Encoder()
        features, mel, expected = feature_payloads(wave, encoder)
        self.assertEqual(expected, 1)
        self.assertEqual(features['layers'].shape, (2, 1, 1024))
        self.assertEqual(mel['mel'].shape, (4, 128))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);original=root/'original.npy';original.write_bytes(b'original')
            receipt = commit_feature('a' * 64, original, root / 'out',
                root / 'tmp', features, mel, expected)
            self.assertTrue(receipt['complete'])
            self.assertEqual(set(receipt['files']),
                {'L12.npy', 'L23.npy', 'times_s.npy', 'intervals_s.npy', 'mel10_complete.npy'})
            prior = existing_feature_receipt(root / 'out' / 'features' / ('a' * 64), encoder)
            self.assertEqual(prior['files'], receipt['files'])
            self.assertTrue((root / 'out' / 'features' / ('a' * 64) / 'receipt.json').exists())

    def test_changed_committed_feature_is_rejected(self):
        wave = np.zeros(960, dtype=np.float32)
        encoder = self.Encoder()
        features, mel, expected = feature_payloads(wave, encoder)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);original=root/'original.npy';original.write_bytes(b'original')
            commit_feature('b' * 64, original, root / 'out',
                root / 'tmp', features, mel, expected)
            path = root / 'out' / 'features' / ('b' * 64) / 'L12.npy'
            values = np.load(path, allow_pickle=False)
            values[0, 1] = 1
            with path.open('wb') as stream:
                np.save(stream, values, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, 'Committed feature changed'):
                existing_feature_receipt(root / 'out' / 'features' / ('b' * 64), encoder)


if __name__ == '__main__':
    unittest.main()
