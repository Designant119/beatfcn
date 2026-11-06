import tempfile
import unittest
from pathlib import Path

from scripts.compare_local_data_scale_quality import verify_sources
from scripts.run_local_song_baseline import sha


class HistoricalQualityIdentityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root/'archive'
        self.canonical = 'beatfcn/events/canonical.py'
        self.parser = 'beatfcn/events/phira.py'
        self.scorer = 'beatfcn/model/evaluation.py'
        self.old = {
            self.canonical: "TYPES=('tap','hold')\ndef build_model_view():\n    return 4\n",
            self.parser: 'ADMISSION=4\n',
            self.scorer: 'SCORING=20\n',
        }
        for name, text in self.old.items():
            for base in (self.root, self.archive):
                p = base/name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text)
        (self.root/'targets.json').write_text('{}')
        self.plan = dict(source_snapshot='archive',
                         inputs_sha256={'targets.json': sha(self.root/'targets.json')},
                         code_sha256={name: sha(self.root/name) for name in self.old})

    def test_only_archived_control_admission_may_change(self):
        (self.root/self.canonical).write_text(self.old[self.canonical].replace('return 4', 'return 10'))
        (self.root/self.parser).write_text('ADMISSION=10\n')
        proof = verify_sources(self.plan, 'control', self.root)
        self.assertTrue(proof['archived_code_verified'])
        self.assertEqual(set(proof['historical_non_scoring_changes']), {self.canonical, self.parser})
        with self.assertRaisesRegex(ValueError, 'Candidate source changed'):
            verify_sources(self.plan, 'candidate', self.root)

    def test_scoring_definitions_must_remain_equal(self):
        (self.root/self.canonical).write_text(self.old[self.canonical].replace("'hold'", "'drag'"))
        with self.assertRaisesRegex(ValueError, 'scoring definitions changed'):
            verify_sources(self.plan, 'control', self.root)

    def test_archive_and_scoring_dependency_are_not_bypassed(self):
        (self.archive/self.parser).write_text('CORRUPT=True\n')
        with self.assertRaisesRegex(ValueError, 'archive changed'):
            verify_sources(self.plan, 'control', self.root)
        (self.archive/self.parser).write_text(self.old[self.parser])
        (self.root/self.scorer).write_text('SCORING=50\n')
        with self.assertRaisesRegex(ValueError, 'scoring dependency changed'):
            verify_sources(self.plan, 'control', self.root)

    def test_frozen_targets_still_required(self):
        (self.root/'targets.json').write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
            verify_sources(self.plan, 'control', self.root)


if __name__ == '__main__':
    unittest.main()
