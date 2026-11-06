import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from beatfcn.train.data import load_manifest, selected_for_training


class PhiraTrainingSelection(unittest.TestCase):
    def row(self, **changes):
        return dict(dict(source='phira', difficulty_label='EZ Lv.3', rating=.75,
                    rating_count=50, status='parsed', kept=True, split='train'), **changes)

    def test_rating_and_vote_boundaries_both_apply(self):
        for votes in [50, 51, 100, 10000, '50']:
            row = self.row()
            row['rating_count'] = votes
            with self.subTest(votes=votes):
                self.assertTrue(selected_for_training(row))
        row = self.row()
        row.pop('rating_count')
        self.assertFalse(selected_for_training(row))
        row['rating'] = .749999
        row['rating_count'] = 100000
        self.assertFalse(selected_for_training(row))

    def test_invalid_or_conflicting_vote_counts_fail_closed(self):
        for votes in [None, 0, 49, 49.999, 50.1, True, 'invalid', float('nan'), float('inf'), -1]:
            with self.subTest(votes=votes):
                self.assertFalse(selected_for_training(self.row(rating=.99, rating_count=votes)))
        row = self.row(); row.pop('rating_count'); row['ratingCount'] = 50
        self.assertTrue(selected_for_training(row))
        row['rating_count'] = 49
        self.assertFalse(selected_for_training(row))

    def test_missing_or_invalid_scores_cannot_be_admitted(self):
        for value in [None, '', 'invalid', float('nan'), float('inf'), -1, 1.01]:
            row = self.row()
            row['rating'] = value
            with self.subTest(value=value):
                self.assertFalse(selected_for_training(row))
        row['rating'] = '0.85'
        self.assertTrue(selected_for_training(row))

    def test_nonstandard_gameplay_tags_remain_outside_training(self):
        for label in ['SP Lv.3', '4K Lv.3', 'ADOFAI Lv.3', '', None]:
            row = self.row()
            row['difficulty_label'] = label
            row['rating'] = 1.0
            with self.subTest(label=label):
                self.assertFalse(selected_for_training(row))

    def test_loader_still_requires_technical_and_split_eligibility(self):
        rows = [dict(self.row(), chart_id='accepted'),
                dict(self.row(), chart_id='quarantined', kept=False,
                     quarantine_reasons=['invalid_time']),
                dict(self.row(), chart_id='heldout', split='val')]
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / 'manifest/charts.jsonl'
            manifest.parent.mkdir()
            manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.assertEqual([row['chart_id'] for row in load_manifest(folder, split='train')],
                             ['accepted'])

    def test_mania_approval_rules_are_unchanged(self):
        for approval, accepted in [(1, True), (2, True), (4, False), (0, False)]:
            self.assertEqual(selected_for_training(dict(source='osu', approved=approval)), accepted)

    def test_expansion_entrypoint_uses_both_gates_without_stable_bypass(self):
        from data_expand.select_ids import main
        rows = [dict(id=1, name='First', level='EZ Lv.3', difficulty=3,
                     reviewed=True, tags=['regular'], file='https://example.org/1',
                     rating=.75, ratingCount=50),
                dict(id=2, name='Second', level='EZ Lv.3', difficulty=3,
                     reviewed=True, tags=['regular'], file='https://example.org/2',
                     rating=.749, ratingCount=10000, stable=True),
                dict(id=3, name='Third', level='EZ Lv.3', difficulty=3,
                     reviewed=True, tags=['regular'], file='https://example.org/3',
                     rating=.99, ratingCount=49, stable=True)]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            catalog, existing, output = (folder / name for name in ['catalog.jsonl', 'old.txt', 'out.txt'])
            catalog.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            existing.write_text('')
            with patch('sys.argv', ['select_ids.py', str(catalog), str(existing), '10', str(output)]):
                with contextlib.redirect_stdout(io.StringIO()):
                    main()
            self.assertEqual(output.read_text().split(), ['1'])


if __name__ == '__main__':
    unittest.main()
