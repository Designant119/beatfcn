import unittest
import numpy as np
from beatfcn.events.requests import build_request
from beatfcn.model.condition_probe import control_views, condition_inventory, screen_control, SWEEP_BANDS
from beatfcn.model import features as F
from events_helpers import synthetic_mania, synthetic_phira, DURATION


class ConditionProbeTests(unittest.TestCase):
    def test_all_supported_fields_are_in_inventory(self):
        self.assertEqual(set(condition_inventory()['numeric_fields']), set(F.CONTROL_FIELDS))

    def test_control_changes_without_targets_context_or_audio_changes(self):
        chart = synthetic_mania()
        sample = build_request(chart, kind='middle_holes', seed=7, audio_id='a', audio_duration_us=DURATION)
        variants = control_views(sample, 'density_notes_per_second')
        c0 = F.choose_core(sample, np.random.default_rng(7), DURATION)
        cores = [F.build_core(v, chart.events, c0, DURATION) for v in variants]
        for v, core in zip(variants, cores):
            self.assertEqual(v.hidden, sample.hidden)
            self.assertEqual(v.evidence, sample.evidence)
            self.assertEqual(v.model_view, sample.model_view)
            for key in ('audio_id', 'regions', 'observed_events', 'references'):
                self.assertEqual(v.request[key], sample.request[key])
            self.assertEqual(set(v.request['controls']), {'allowed_types', 'density_notes_per_second'})
            np.testing.assert_array_equal(core.known, cores[0].known)
            np.testing.assert_array_equal(core.refs, cores[0].refs)
            self.assertEqual(repr(core.groups), repr(cores[0].groups))
        self.assertEqual(len({tuple(c.controls) for c in cores}), 3)

    def test_mode_and_invalid_band_are_rejected(self):
        sample = build_request(synthetic_phira(), kind='from_scratch', seed=7, audio_id='a', audio_duration_us=DURATION)
        with self.assertRaises(ValueError):
            control_views(sample, 'jack_ratio')
        with self.assertRaises(ValueError):
            control_views(sample, 'density_notes_per_second', [[2., 9.]])

    def test_all_controls_stop_flat_response_without_promoting_partial_or_illegal_results(self):
        for field, inventory in condition_inventory()['numeric_fields'].items():
            mode = inventory['modes'][0]
            songs = ['a', 'b', 'c', 'd']
            rows = [{'song': song, 'requested': band, 'achieved': .25, 'complete': True,
                     'legality': [], 'feasible': True} for song in songs for band in SWEEP_BANDS[field]]
            result = screen_control(field, mode, rows, songs)
            self.assertEqual(result['decision'], 'stop_futility')
            self.assertFalse(result['promotion'])
            self.assertEqual(screen_control(field, mode, rows[:-1], songs)['decision'], 'insufficient_coverage')
            rows[-1]['feasible'] = False
            self.assertEqual(screen_control(field, mode, rows, songs)['complete_feasible_sweeps'], 3)
            rows[-1]['feasible'] = True
            for row in rows:
                row['achieved'] = sum(row['requested']) / 2
            result = screen_control(field, mode, rows, songs)
            self.assertEqual(result['decision'], 'quality_review_required')
            self.assertFalse(result['promotion'])
            rows[-1]['legality'] = ['bad']
            self.assertEqual(screen_control(field, mode, rows, songs)['decision'], 'insufficient_coverage')


if __name__ == '__main__':
    unittest.main()
