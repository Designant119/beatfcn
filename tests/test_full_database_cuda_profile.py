import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import wave
import zipfile

import numpy as np

from beatfcn.audio.cuda_mel import complete_song_mel_cuda, mel_parity
from beatfcn.audio.local_mel_cache import complete_song_mel
from beatfcn.cloud.full_database_app import verify_feature_folder
from beatfcn.train.cuda_pipeline import (
    commit_feature_hashing, decoded_audio_from_receipt, load_raw_index)
from beatfcn.train.full_database import digest, feature_payloads
from tests.test_full_database_mert_pilot_runtime import Encoder


class CudaProfileTests(unittest.TestCase):
    def test_cuda_mel_cpu_mode_is_exact_with_legacy_complete_recipe(self):
        for samples in (960, 1000, 24000):
            wave = np.linspace(-1, 1, samples, dtype=np.float32)
            reference = complete_song_mel(wave)['mel']
            candidate = complete_song_mel_cuda(wave, device='cpu')['mel']
            result = mel_parity(reference, candidate)
            self.assertEqual(result['max_abs'], 0.)
            self.assertEqual(result['float16_exact_fraction'], 1.)

    def test_receipt_direct_zip_decode_skips_raw_rehash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_path = root / 'input.wav'
            with wave.open(str(audio_path), 'wb') as stream:
                stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(24000)
                stream.writeframes(np.zeros(24000, dtype=np.int16).tobytes())
            payload = audio_path.read_bytes()
            audio = hashlib.sha256(payload).hexdigest()
            archive = root / 'raw/phira/zips/test.zip'
            archive.parent.mkdir(parents=True)
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.write(audio_path, 'song.wav')
            record = dict(audio_sha256=audio, source='phira',
                archive='raw/phira/zips/test.zip', archive_member='song.wav',
                payload_bytes=len(payload), payload_sha256=audio, verified=True,
                manifest_sha256='m')
            content = root / 'full-database-v1/raw-content.jsonl'
            content.parent.mkdir(parents=True)
            content.write_text(json.dumps(record) + '\n')
            index = load_raw_index(root, hashlib.sha256(content.read_bytes()).hexdigest(), 'm')
            decoded, original, values = decoded_audio_from_receipt(
                dict(audio_sha256=audio), index, root, root / 'tmp')
            self.assertEqual(decoded, audio)
            self.assertEqual(len(values), 24000)
            self.assertTrue(np.isfinite(values).all())

    def test_single_pass_write_hash_matches_rehash_and_remote_recovery(self):
        wave = np.linspace(-1, 1, 1000, dtype=np.float32)
        encoder = Encoder()
        features, mel, expected = feature_payloads(wave, encoder)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / 'original.audio';original.write_bytes(b'original')
            folder = root / 'out' / 'features' / ('b' * 64)
            result = commit_feature_hashing('b' * 64, original, root / 'out',
                root / 'tmp', features, mel, expected)
            for name, info in result['files'].items():
                self.assertEqual(info['sha256'], digest(folder / name))
            recovered = verify_feature_folder(folder,
                dict(audio_sha256='b' * 64, expected_mel_frames=4), encoder.revision)
            self.assertEqual(recovered['complete_tail_frames'], 2)


if __name__ == '__main__':
    unittest.main()
