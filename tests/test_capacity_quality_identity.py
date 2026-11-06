import tempfile
import unittest
from pathlib import Path

from scripts.compare_local_capacity_quality import verify_sources
from scripts.run_local_song_baseline import sha


class CapacityQualityIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.code = {'scripts/run_local_data_scale.py': 'original execution',
                     'beatfcn/model/evaluation.py': 'unchanged scoring'}
        for path, value in self.code.items():
            for prefix in ('', 'snapshot/'):
                p = self.root / (prefix + path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(value)
        (self.root / 'target').write_text('frozen targets')
        self.plan = dict(source_snapshot='snapshot',
            inputs_sha256={'target': sha(self.root / 'target')},
            code_sha256={p: sha(self.root / p) for p in self.code})

    def test_only_archived_closed_control_runner_may_differ(self):
        (self.root / 'scripts/run_local_data_scale.py').write_text('closed')
        result = verify_sources(self.plan, 'control', self.root)
        self.assertEqual(result['historical_non_scoring_changes'], ['scripts/run_local_data_scale.py'])
        with self.assertRaises(ValueError):
            verify_sources(self.plan, 'candidate', self.root)

    def test_scoring_and_targets_cannot_change(self):
        for path in ('target', 'beatfcn/model/evaluation.py'):
            original = (self.root / path).read_text()
            (self.root / path).write_text('changed')
            with self.assertRaises(ValueError):
                verify_sources(self.plan, 'control', self.root)
            (self.root / path).write_text(original)

    def test_archive_change_is_not_excused_by_closed_runner(self):
        (self.root / 'snapshot/scripts/run_local_data_scale.py').write_text('changed archive')
        with self.assertRaises(ValueError):
            verify_sources(self.plan, 'control', self.root)


if __name__ == '__main__':
    unittest.main()
