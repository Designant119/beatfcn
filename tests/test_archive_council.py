import json
from pathlib import Path
import tempfile
import unittest

from scripts.archive_council import REL, append_event, check, digest, snapshot, write_json


def fixture_archive(root):
    base = root / REL
    base.mkdir(parents=True)
    source = root / 'reports/council-test'
    source.mkdir()
    (source / 'sol.md').write_text('First full opinion.\n')
    receipt = base / 'receipt.json'
    write_json(receipt, {'result': {'structuredContent': {
        'taskId': 'finished-task', 'hasPendingChildRuns': False,
        'status': 'completed', 'model': 'test-model', 'options': [],
    }}})
    raw = base / 'input.json'
    write_json(raw, {'result': {'structuredContent': {'items': [
        {'itemId': 'prompt-1', 'text': 'Full input', 'textTruncated': False}
    ]}}})
    prompt = base / 'prompt.txt'
    prompt.write_text('Full input')
    attempt = dict(id='R-S', round_id='R', label='Sol', outcome='completed',
        task_id='finished-task', model='test-model', options=[], effort='ultra',
        report='reports/council-test/sol.md', receipt_files=[str(receipt.relative_to(root))],
        prompt_files=[str(prompt.relative_to(root)), str(raw.relative_to(root))],
        prompt_complete=True)
    pending = dict(id='R-A', round_id='R', label='Argon', outcome='pending',
        model='test-model', options=[], report=None, receipt_files=[], prompt_files=[])
    write_json(base / 'manifest.json', dict(schema_version=1, recording_basis='test',
        rounds=[dict(id='R', title='Test', source_dir='reports/council-test', context='Test')],
        attempts=[attempt, pending]))
    row = dict(event_id='D.1', id='D', title='Keep alternative', attempt_id='R-S',
        source=attempt['report'], source_sha256=digest((source / 'sol.md').read_bytes()),
        locator='First paragraph', status='deferred', recommendation='An alternative',
        rationale='Try later', evidence=[attempt['report']], revisit='When evidence changes',
        recorded_at='2026-10-27', recording_basis='test', context='R')
    (base / 'decisions.jsonl').write_text(json.dumps(row) + '\n')
    return base, source, row


class ArchiveCouncilTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_source_versions_preserve_history_and_pending_is_not_an_opinion(self):
        _, source, _ = fixture_archive(self.root)
        _, first = snapshot(self.root)
        report = next(v for v in first['versions'] if v['source'].endswith('/sol.md'))
        archived = self.root / report['object']
        self.assertEqual(archived.read_text(), 'First full opinion.\n')
        _, unchanged = snapshot(self.root)
        self.assertEqual(unchanged, first)
        (source / 'sol.md').write_text('Revised opinion.\n')
        _, second = snapshot(self.root)
        self.assertEqual(len(second['versions']), len(first['versions']) + 1)
        self.assertEqual(archived.read_text(), 'First full opinion.\n')
        self.assertEqual(check(self.root)['attempts'], {'completed': 1, 'pending': 1})
        self.assertEqual(check(self.root)['recommendation_count'], 1)

    def test_corrupt_archive_is_rejected_instead_of_overwritten(self):
        fixture_archive(self.root)
        _, catalog = snapshot(self.root)
        original = self.root / catalog['versions'][0]['object']
        original.write_text('corrupt')
        with self.assertRaisesRegex(ValueError, 'Corrupt historical object'):
            snapshot(self.root)
        self.assertEqual(original.read_text(), 'corrupt')

    def test_decisions_append_and_invalid_event_preserves_log(self):
        base, _, row = fixture_archive(self.root)
        snapshot(self.root)
        before = (base / 'decisions.jsonl').read_bytes()
        with self.assertRaisesRegex(ValueError, 'Missing decision field'):
            append_event(self.root, {**row, 'event_id': 'D.2', 'revisit': ''})
        self.assertEqual((base / 'decisions.jsonl').read_bytes(), before)
        append_event(self.root, {**row, 'event_id': 'D.2', 'status': 'adopted',
                               'rationale': 'New evidence triggered recovery'})
        self.assertTrue((base / 'decisions.jsonl').read_bytes().startswith(before))
        snapshot(self.root)
        self.assertEqual(check(self.root)['decision_events'], 2)
        self.assertEqual(check(self.root)['statuses'], {'adopted': 1})
        (base / 'decisions.jsonl').write_bytes(before)
        with self.assertRaisesRegex(ValueError, 'Decision history was rewritten'):
            snapshot(self.root)

    def test_partial_prompt_cannot_be_marked_complete(self):
        base, _, _ = fixture_archive(self.root)
        (base / 'prompt.txt').write_text('Only part of input')
        snapshot(self.root)
        with self.assertRaisesRegex(ValueError, 'Prompt pagination incomplete'):
            check(self.root)


if __name__ == '__main__':
    unittest.main()
