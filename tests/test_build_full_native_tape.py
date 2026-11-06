import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from beatfcn.audio.mert2 import SR

ROOT = Path(__file__).resolve().parents[1]


def chart(root, features, name, audio, split, mode='phira'):
    folder = root/name
    folder.mkdir()
    onsets = np.arange(0, 9_000_000, 250_000, dtype=np.int64)
    values = np.stack([onsets, onsets, np.zeros_like(onsets), np.full_like(onsets, -1 if mode == 'phira' else 0)], 1)
    np.save(folder/'filtered.npy', values)
    (folder/'receipt.json').write_text(json.dumps(dict(chart_id=f'phira:{name}', audio_sha256=audio,
        split=split, mode=mode, canonical_sha256='c'*64, component='a:'+audio)))
    (features/audio).mkdir(exist_ok=True)
    (features/audio/'receipt.json').write_text(json.dumps(dict(metadata=dict(real_samples=10*SR))))


class FullNativeTapeTests(unittest.TestCase):
    def run_builder(self, *extra):
        return subprocess.run([sys.executable, '-m', 'scripts.build_full_native_tape',
            '--targets-root', str(self.targets), '--features-root', str(self.features),
            '--output', str(self.output), '--workers', '2', *extra],
            cwd=ROOT, capture_output=True, text=True, check=True)

    def test_parallel_resume_and_split_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self.targets, self.features, self.output = tmp/'t', tmp/'f', tmp/'o'
            self.targets.mkdir(); self.features.mkdir()
            chart(self.targets, self.features, '1', 'a'*64, 'train')
            chart(self.targets, self.features, '2', 'b'*64, 'val')
            chart(self.targets, self.features, '3', 'a'*64, 'val')  # cross-split audio
            self.run_builder('--limit', '1')
            self.assertFalse((self.output/'tape.json').exists())
            self.assertTrue((self.output/'tape-limit1.json').exists())
            self.run_builder()
            tape = json.loads((self.output/'tape.json').read_text())
            self.assertEqual([c['chart_id'] for c in tape['charts']], ['phira:1', 'phira:2'])
            self.assertEqual(tape['skipped'][0]['reason'], 'audio_cross_split')
            first = (self.output/'phira-1-chart.json').stat().st_mtime_ns
            self.run_builder()  # resume: verified sidecar is reused, not rewritten
            self.assertEqual((self.output/'phira-1-chart.json').stat().st_mtime_ns, first)
            again = json.loads((self.output/'tape.json').read_text())
            self.assertEqual(again['charts'], tape['charts'])


if __name__ == '__main__':
    unittest.main()
