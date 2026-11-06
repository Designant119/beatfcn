import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.freeze_learning_watch_panel import digest, freeze_panel


class LearningPanelTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        charts = []
        for index, mode in enumerate(('phira', 'mania_4k')):
            target = root / f'{index}.npz'
            np.savez(target, kind=np.array([0, 1]))
            charts.append(dict(chart_id=str(index), audio_sha256=str(index), component=str(index),
                split='val', mode=mode, target_path=str(target), target_sha256=digest(target),
                requests=[dict(target_start=0, target_stop=2, request=dict(known_carry=[{'id': 'public'}])),
                          dict(target_start=2, target_stop=2, request={})]))
        path = root / 'tape.json'
        path.write_text(json.dumps(dict(complete=True, failures=[], charts=charts)))
        return root, path, charts

    def test_selection_keeps_song_identity_silence_and_public_carry(self):
        with tempfile.TemporaryDirectory() as directory:
            root, tape, _ = self.fixture(directory)
            result = freeze_panel(tape, root, digest(tape), 'a'*64, per_mode=1)
            self.assertEqual(result['distinct_positive_songs'], 2)
            self.assertEqual(len(result['samples']), 4)
            positives = [s for s in result['samples'] if s['case_role'] == 'positive']
            self.assertTrue(all(s['request']['known_carry'] == [{'id': 'public'}] for s in positives))
            self.assertFalse(result['protocol_ready'])

    def test_training_audio_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, tape, charts = self.fixture(directory)
            training = dict(charts[0], split='train')
            tape.write_text(json.dumps(dict(complete=True, failures=[], charts=charts+[training])))
            with self.assertRaisesRegex(ValueError, 'overlaps training'):
                freeze_panel(tape, root, digest(tape), 'a'*64, per_mode=1)

    def test_target_replacement_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, tape, charts = self.fixture(directory)
            Path(charts[0]['target_path']).write_bytes(b'replaced')
            with self.assertRaisesRegex(ValueError, 'Target identity'):
                freeze_panel(tape, root, digest(tape), 'a'*64, per_mode=1)
