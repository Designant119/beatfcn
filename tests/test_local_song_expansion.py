import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import unittest

from beatfcn.events.canonical import Issue
from beatfcn.events.phira import parse_phira
from scripts.prepare_local_song_expansion_v2 import raw_issue_audit


class NativeStackAdmission(unittest.TestCase):
    def chart(self, count):
        fixture = Path(__file__).parent / 'fixtures/rpe_multibpm_synthetic.json'
        doc = json.loads(fixture.read_text())
        note = deepcopy(doc['judgeLineList'][0]['notes'][3])
        doc['judgeLineList'] = [doc['judgeLineList'][0]]
        doc['judgeLineList'][0]['notes'] = [deepcopy(note) for _ in range(count)]
        return parse_phira(json.dumps(doc).encode())

    def test_four_artistic_drags_keep_all_objects(self):
        chart = self.chart(4)
        before = chart.events
        result = raw_issue_audit(chart)
        self.assertEqual(result['maximum_source_stack'], 4)
        self.assertEqual(result['intentional_duplicate_issues'], 3)
        self.assertEqual(chart.events, before)
        self.assertEqual(len(chart.events), 4)
        self.assertTrue(all(e.type == 'drag' for e in chart.events))

    def test_deep_native_stack_is_not_a_fake_heuristic(self):
        from beatfcn.train.data import check_stacks
        chart = self.chart(32)
        self.assertFalse(chart.quarantined)
        result = raw_issue_audit(chart)
        self.assertEqual(result['maximum_source_stack'], 32)
        self.assertIsNone(result['maximum_allowed_source_stack'])
        self.assertEqual(check_stacks(chart.events), 32)
        self.assertEqual(len(chart.events), 32)

    def test_duplicate_does_not_hide_other_quarantine(self):
        chart = self.chart(2)
        chart = replace(chart, issues=chart.issues + (
            Issue('ambiguous_is_fake', 'quarantine', 'unknown fake flag'),))
        with self.assertRaisesRegex(ValueError, 'ambiguous_is_fake'):
            raw_issue_audit(chart)

    def test_stack_exception_is_phira_only(self):
        chart = replace(self.chart(2), mode='mania_4k', issues=(
            Issue('duplicate_object', 'quarantine', 'invalid mania object'),))
        with self.assertRaisesRegex(ValueError, 'duplicate_object'):
            raw_issue_audit(chart)


if __name__ == '__main__':
    unittest.main()
