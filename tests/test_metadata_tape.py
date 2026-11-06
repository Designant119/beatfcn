import hashlib
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import torch

from beatfcn.train.tape import MetadataTape, policy_digest
from scripts.condition_cohort_probe import ResidentCohorts
from beatfcn.train.pinned import PinnedRing


class MetadataTapeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.policy = {'request_mix': {'from_scratch': 1.0}}
        batches = [({'_audio': {'keys': ['a', 'b']}, '_packed': {'torch.int64': torch.arange(8) + i},
                     '_provenance': [{'seed': i}]}, {}) for i in range(3)]
        path = self.root / 'part-0000.pt'
        torch.save(batches, path)
        content = path.read_bytes()
        manifest = {'schema': 'beatfcn.metadata_tape.v1', 'complete': True, 'batches': 3,
                    'rows_sha256': 'data', 'policy_sha256': policy_digest(self.policy), 'micro_batch': 2,
                    'parts': [{'file': path.name, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest(), 'batches': 3}]}
        (self.root / 'manifest.json').write_text(json.dumps(manifest))

    def load(self, **changes):
        args = dict(rows_sha256='data', policy=self.policy, micro_batch=2)
        return MetadataTape(self.root, **(args | changes))

    def test_resume_preserves_values_and_provenance_without_mutating_tape(self):
        tape = self.load(skip_batches=1)
        first, _ = next(iter(tape))
        self.assertEqual(first['_provenance'][0]['seed'], 1)
        self.assertTrue(torch.equal(first['_packed']['torch.int64'], torch.arange(8) + 1))
        first['_packed'] = {'replaced_by_pinned_slot': True}
        self.assertIn('torch.int64', next(iter(tape))[0]['_packed'])
        self.assertEqual(len(list(tape)), 2)

    def test_identity_and_resume_mismatches_are_rejected(self):
        for changes in ({'rows_sha256': 'other'}, {'policy': {}}, {'micro_batch': 4}, {'skip_batches': 3}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.load(**changes)

    def test_corrupt_or_partial_tape_cannot_enter_training(self):
        part = self.root / 'part-0000.pt'
        part.write_bytes(part.read_bytes()[:-1] + b'x')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            self.load()
        path = self.root / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['complete'] = False
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.load()

    def test_compressed_tape_preserves_exact_batches_and_checks_decoded_identity(self):
        path = self.root / 'manifest.json'
        manifest = json.loads(path.read_text())
        part = manifest['parts'][0]
        original = (self.root / part['file']).read_bytes()
        compressed = gzip.compress(original, compresslevel=1, mtime=0)
        (self.root / part['file']).write_bytes(compressed)
        part.update(codec='gzip', raw_bytes=len(original), raw_sha256=part['sha256'],
                    bytes=len(compressed), sha256=hashlib.sha256(compressed).hexdigest())
        path.write_text(json.dumps(manifest))
        loaded = list(self.load())
        self.assertEqual(len(loaded), 3)
        for i, (batch, _) in enumerate(loaded):
            self.assertTrue(torch.equal(batch['_packed']['torch.int64'], torch.arange(8) + i))
        part['raw_bytes'] -= 1
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'decoded.*integrity'):
            self.load()

    def test_preallocated_slots_preserve_batches_without_growth_or_cross_slot_aliases(self):
        ring = PinnedRing(depth=2, pin=False)
        self.assertEqual(ring.preallocate({'i': torch.arange(8)}), 128)
        grown = ring.stats['grown']
        first, _ = ring.take({'i': torch.arange(5)})
        second, _ = ring.take({'i': torch.arange(8) + 10})
        self.assertTrue(torch.equal(first['i'], torch.arange(5)))
        self.assertTrue(torch.equal(second['i'], torch.arange(8) + 10))
        ring.take({'i': torch.arange(3) + 20})
        self.assertEqual(ring.stats['grown'], grown)
        self.assertTrue(torch.equal(second['i'], torch.arange(8) + 10))
        with self.assertRaisesRegex(ValueError, 'fresh'):
            ring.preallocate({'i': torch.arange(8)})


class ResidentCohortTests(unittest.TestCase):
    def test_order_and_pins_survive_cohort_boundaries_without_background_loads(self):
        events, pins = [], set()
        class Pool:
            def request(self, keys):
                self.pending = True
                pins.update(keys)
                events.append(('request', tuple(sorted(keys))))
            def wait_idle(self):
                self.pending = False
                events.append(('ready',))
            def release(self, keys):
                pins.difference_update(keys)
                events.append(('release',))
        pool = Pool()
        batches = [({'_audio': {'keys': keys}, 'identity': i}, {})
                   for i, keys in enumerate([['a'], ['b'], ['a'], ['c'], ['b'], ['d']])]
        actual = []
        for batch, _ in ResidentCohorts(batches, pool, max_songs=2, max_batches=3,
                                       before_load=lambda: events.append(('drain',))):
            self.assertFalse(pool.pending)
            self.assertTrue(set(batch['_audio']['keys']) <= pins)
            actual.append(batch['identity'])
        self.assertEqual(actual, list(range(6)))
        self.assertFalse(pins)
        self.assertEqual([x for x in events if x[0] == 'request'],
                         [('request', ('a', 'b')), ('request', ('b', 'c')), ('request', ('d',))])

    def test_closing_early_releases_the_current_cohort(self):
        from unittest.mock import Mock
        pool = Mock()
        rows = [({'_audio': {'keys': ['a']}}, {})] * 4
        iterator = iter(ResidentCohorts(rows, pool, max_songs=1, max_batches=4))
        next(iterator)
        iterator.close()
        pool.release.assert_called_once_with({'a'})


if __name__ == '__main__':
    unittest.main()
