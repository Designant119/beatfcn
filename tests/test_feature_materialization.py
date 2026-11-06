import importlib.util
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location('feature_materialization', Path(__file__).resolve().parents[1] / 'scripts/materialize_features.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class MaterializationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sha = 'a'*64
        source = self.root / 'source' / 'taps/v4_i8_L9-16-24' / self.sha
        source.mkdir(parents=True)
        entries = {}
        for i, name in enumerate(M.NAMES):
            arr = np.arange(327, dtype=np.float32 if name=='S20.npy' else np.int8).reshape(109, 3)
            np.save(source/name, arr)
            entries[name] = dict(path=str((source/name).relative_to(self.root/'source')), bytes=(source/name).stat().st_size)
        self.inv = dict(complete=True, songs={self.sha: dict(files=entries)}, feature_bytes=sum(e['bytes'] for e in entries.values()))

    def run_copy(self, **kw):
        return M.materialize(self.inv, self.root/'source', self.root/'local', deadline=time.perf_counter()+5, **kw)

    def test_exact_original_arrays_and_local_only_lookup(self):
        result = self.run_copy()
        self.assertTrue(result['complete'])
        for path in M.local_paths(self.root/'local', self.inv, self.sha):
            original = self.root/'source'/self.inv['songs'][self.sha]['files'][path.name]['path']
            self.assertEqual(path.read_bytes(), original.read_bytes())
            np.testing.assert_array_equal(np.load(path), np.load(original))
        with self.assertRaises(ValueError):
            M.local_paths(self.root/'local', self.inv, 'b'*64)
        (self.root/'local'/self.sha/'L9.npy').unlink()
        with self.assertRaises(ValueError):
            M.local_paths(self.root/'local', self.inv, self.sha)

    def test_short_source_fails_without_remote_fallback(self):
        p=self.root/'source'/self.inv['songs'][self.sha]['files']['L9.npy']['path']
        p.write_bytes(p.read_bytes()[:-1])
        with self.assertRaises((ValueError, TimeoutError)):
            self.run_copy()

    def test_deadline_capacity_and_untrusted_paths_fail_before_use(self):
        with self.assertRaises(ValueError):
            self.run_copy(max_bytes=1)
        with self.assertRaises(TimeoutError):
            M.materialize(self.inv,self.root/'source',self.root/'local',deadline=time.perf_counter()-1)
        self.inv['songs'][self.sha]['files']['L9.npy']['path']='../outside.npy'
        with self.assertRaises(ValueError):
            self.run_copy()


if __name__ == '__main__':
    unittest.main()
