import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('condition_summary', Path(__file__).parents[1] / 'scripts/summarize_condition_benchmark.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ConditionSummaryTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{'id': f'{song}-{kind}', 'checkpoint': 'model', 'chart_id': song, 'song': song, 'mode': 'phira',
                      'request_kind': 'density_' + kind, 'complete': True, 'legality': [], 'requested': band,
                      'achieved': value, 'minimum_from_observed': 0.}
                     for song in ('a', 'b') for kind, band, value in
                     (('low', [1., 1.5], 1.25), ('mid', [3., 4.], 3.5), ('high', [6., 8.], 7.))]
        self.protocol = {'expected': [{k: r[k] for k in ('id', 'checkpoint', 'chart_id', 'song', 'mode', 'request_kind')}
                                      for r in self.rows]}

    def test_missing_and_infeasible_songs_remain_in_requested_denominator(self):
        result = module.summarize(self.protocol, self.rows[:-1])
        control = result['controls']['model/phira/window']
        self.assertFalse(result['complete'])
        self.assertEqual((result['expected'], result['received']), (6, 5))
        self.assertEqual((result['not_recorded'], result['failed_recorded'], result['complete_legal']), (1, 0, 5))
        self.assertEqual((control['requested_songs'], control['complete_feasible_sweeps']), (2, 1))
        self.assertEqual((control['positive_monotonic_songs'], control['in_band']), (1, 3))
        rows = copy.deepcopy(self.rows)
        rows[0]['minimum_from_observed'] = 2.
        control = module.summarize(self.protocol, rows)['controls']['model/phira/window']
        self.assertEqual((control['requested_songs'], control['complete_feasible_sweeps']), (2, 1))

    def test_duplicates_are_not_silently_counted_as_more_songs(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            module.summarize(self.protocol, self.rows + self.rows[:1])

    def test_control_improvement_is_paired_by_song_and_requires_both_complete_sweeps(self):
        rows = []
        for label in ('r02-density-supervised', 'r02-density-contrastive'):
            for row in self.rows:
                r = dict(row, id=label + row['id'], checkpoint=label)
                if label.endswith('supervised'):
                    r['achieved'] = 3.5
                rows.append(r)
        protocol = {'expected': [{k: r[k] for k in ('id', 'checkpoint', 'chart_id', 'song', 'mode', 'request_kind')} for r in rows]}
        rows[-1]['complete'] = False
        result = module.summarize(protocol, rows)
        delta = result['paired_control_changes']['r02-density-contrastive-minus-r02-density-supervised/phira/window']
        self.assertEqual(delta['high_minus_low']['songs'], 1)
        self.assertEqual(delta['high_minus_low']['mean'], 5.75)
        self.assertEqual(delta['monotonic_fraction']['mean'], 1.)
        self.assertAlmostEqual(delta['in_band_fraction']['mean'], 2 / 3)
        self.assertFalse(result['complete'])
        self.assertEqual((result['not_recorded'], result['failed_recorded']), (0, 1))


if __name__ == '__main__':
    unittest.main()
