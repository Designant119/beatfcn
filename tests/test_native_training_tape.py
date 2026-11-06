import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from beatfcn.events.canonical import Event, LineLoc
from scripts.prepare_native_training_tape import prepare_native_tape


class NativeTrainingTapeTests(unittest.TestCase):
    def test_native_clock_strict_filter_duplicates_and_cross_window_hold(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            loc = LineLoc('0', 'above', 0.)
            events = [Event(str(i), 100_000, 100_000, 'tap', loc) for i in range(12)]
            events += [Event('drag', 200_000, 200_000, 'drag', loc),
                Event('hold', 7_000_000, 8_500_000, 'hold', loc),
                Event('tap', 8_100_000, 8_100_000, 'tap', loc)]
            document = dict(mode='phira', source_format='rpe', format_version=1,
                events=[e.to_json() for e in events])
            path = root/'canonical.json'
            path.write_text(json.dumps(document))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            chart = dict(chart_id='phira:1', canonical_sha256=digest, duration_us=9_000_000,
                mode='phira', component='song', split='train', stratum='HD', audio_sha256='a'*64,
                offset_decision=dict(c_ms=17, shift_ms=20))
            (root/'source.json').write_text(json.dumps(dict(complete=True, failures=[], charts=[chart])))
            (root/'receipt.json').write_text(json.dumps(dict(complete=True, failures=[],
                files=[dict(sha256=digest, local='canonical.json')])))
            result = prepare_native_tape(root, 'source.json', 'receipt.json', root/'new')
            self.assertTrue(result['complete'])
            self.assertEqual(result['removals'][0]['removed_indices'], [12])
            chart = result['charts'][0]
            with np.load(root/chart['target_path']) as targets:
                self.assertEqual(int(targets['onset_us'][0]), 100_000)
                self.assertEqual(int((targets['onset_us'] == 100_000).sum()), 12)
                self.assertEqual(len(targets['onset_us']), 14)
            self.assertEqual(chart['requests'][1]['incoming_holds'], 1)
            self.assertEqual(chart['requests'][1]['event_count'], 1)
            with self.assertRaises(FileExistsError):
                prepare_native_tape(root, 'source.json', 'receipt.json', root/'new')


if __name__ == '__main__':
    unittest.main()
