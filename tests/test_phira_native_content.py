import json
import unittest

from scripts.audit_phira_native_content import inspect_chart


class NativeContentAccounting(unittest.TestCase):
    def chart(self, notes, **line):
        return json.dumps(dict(META={'offset': 0}, BPMList=[dict(startTime=[0, 0, 1], bpm=120)],
            judgeLineList=[dict(notes=notes, **line)])).encode()

    def note(self, **changes):
        return dict(dict(type=4, startTime=[1, 0, 1], endTime=[1, 0, 1],
                         positionX=0, isFake=0), **changes)

    def test_fake_and_duplicate_artistic_drags_have_distinct_destinations(self):
        result = inspect_chart(self.chart([self.note(), self.note(), self.note(isFake=1)]))
        self.assertTrue(all(result['checks'].values()))
        self.assertEqual(result['canonical_types'], {'drag': 2})
        self.assertEqual(result['separated_fake_ids'], ['rpe:0:2'])
        self.assertEqual(result['status'], 'static_screen_only')
        self.assertFalse(result['native_playability_verified'])

    def test_visual_fields_never_block_real_notes(self):
        for fields in ({'alpha': 0}, {'alpha': -1}, {'speed': 0}, {'size': 0},
                       {'visibleTime': 0}, {'alpha': 0, 'speed': 0, 'size': 0}):
            with self.subTest(fields=fields):
                result = inspect_chart(self.chart([self.note(**fields)]))
                self.assertEqual(result['status'], 'static_screen_only')
                self.assertEqual(result['counts']['canonical_events'], 1)
                self.assertEqual(result['review_signals'], {})
                self.assertEqual(result['removed_nonfake_events'], 0)
                self.assertTrue(all(result['checks'].values()))

    def test_attached_ui_and_line_animation_are_not_interchangeable(self):
        result = inspect_chart(self.chart([self.note()], attachUI='score'))
        self.assertIn('nonfake_on_attached_ui', result['visual_observations'])
        self.assertEqual(result['review_signals'], {})
        self.assertEqual(result['status'], 'static_screen_only')
        ordinary = inspect_chart(self.chart([self.note()], extended={'textEvents': []}))
        self.assertEqual(ordinary['status'], 'static_screen_only')
        fake = inspect_chart(self.chart([self.note(isFake=1)], attachUI='score'))
        self.assertEqual(fake['counts']['fake_only_lines'], 1)
        self.assertEqual(fake['review_signals'], {})

    def test_ambiguous_fake_flags_cannot_be_certified(self):
        result = inspect_chart(self.chart([self.note(isFake=2)]))
        self.assertEqual(result['status'], 'parse_or_accounting_quarantine')
        self.assertEqual(result['unknown_fake_ids'], ['rpe:0:0'])
        self.assertFalse(result['checks']['all_objects_accounted'])


if __name__ == '__main__':
    unittest.main()
