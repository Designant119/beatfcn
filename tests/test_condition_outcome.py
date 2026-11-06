import unittest

from beatfcn.events.canonical import Event, LaneLoc, LineLoc
from beatfcn.train.condition_outcome import interval_violation, score_condition_outcome


def event(t, kind='tap', end=None, lane=None):
    return Event(str(t), t, t if end is None else end, kind,
                 LaneLoc(lane) if lane is not None else LineLoc('1', 'above', 0.))


class ConditionOutcomeTests(unittest.TestCase):
    def score(self, events, bands, **kwargs):
        return score_condition_outcome('phira', events, (0, 8_000_000), bands,
                                       complete=kwargs.get('complete', True),
                                       legality_errors=kwargs.get('legality_errors', []))

    def test_empty_cannot_satisfy_nonzero_density_or_undefined_ratio(self):
        result = self.score([], {'density_notes_per_second': [2., 3.], 'hold_head_ratio': [0., .02]})
        self.assertFalse(result['all_requested_satisfied'])
        self.assertLess(result['reward'], -1.)
        self.assertFalse(result['fields']['hold_head_ratio']['defined'])
        self.assertEqual(result['fields']['hold_head_ratio']['violation'], 2.)

    def test_real_silence_and_unknown_are_not_forced_to_emit(self):
        result = self.score([], {'density_notes_per_second': [0., .5]})
        self.assertEqual(result['reward'], 0.)
        self.assertTrue(result['all_requested_satisfied'])
        unknown = self.score([], {})
        self.assertIsNone(unknown['reward'])
        self.assertFalse(unknown['all_requested_satisfied'])

    def test_exclusive_upper_boundary_has_nonzero_penalty(self):
        self.assertEqual(interval_violation('density_notes_per_second', 3., [2., 3.])['violation'], 1.)
        self.assertEqual(interval_violation('drag_ratio', 1., [.8, 1.])['violation'], 0.)
        self.assertGreater(interval_violation('density_notes_per_second', 10., [2., 3.])['violation'],
                           interval_violation('density_notes_per_second', 4., [2., 3.])['violation'])

    def test_incoming_hold_uses_union_and_excludes_outside_heads(self):
        rows = [event(0, 'hold', 5_000_000), event(3_000_000, 'hold', 6_000_000), event(8_000_000)]
        result = score_condition_outcome('phira', rows, (2_000_000, 8_000_000),
            {'hold_occupancy_ratio': [.6, .8]}, complete=True, legality_errors=[])
        self.assertAlmostEqual(result['realized']['hold_occupancy_ratio'], 4/6)
        self.assertEqual(result['audit']['heads'], 1)
        self.assertTrue(result['all_requested_satisfied'])

    def test_drag_multiplicity_preserved_and_other_fields_cannot_compensate(self):
        e = event(1000, 'drag')
        result = self.score([e, e], {'drag_ratio': [.8, 1.], 'chord_ratio': [.8, 1.],
                                    'hold_head_ratio': [.8, 1.]})
        self.assertEqual(result['audit']['heads'], 2)
        self.assertEqual(result['realized']['drag_ratio'], 1.)
        self.assertEqual(result['realized']['chord_ratio'], 1.)
        self.assertFalse(result['all_requested_satisfied'])
        self.assertLess(result['reward'], 0.)

    def test_all_tap_penalized_for_requested_flick(self):
        result = self.score([event(1000), event(2000)], {'flick_ratio': [.2, .3]})
        self.assertLess(result['reward'], -1.)
        self.assertEqual(result['audit']['positive_tap_gaps_lt10ms_fraction'], 1.)
        self.assertIsNone(result['quality_passed'])

    def test_incomplete_or_illegal_cannot_receive_satisfied_reward(self):
        for flags in [dict(complete=False), dict(legality_errors=['locked note removed'])]:
            result = self.score([], {'density_notes_per_second': [0., .5]}, **flags)
            self.assertEqual(result['reward'], -2.)
            self.assertFalse(result['all_requested_satisfied'])

    def test_mania_jack_and_inapplicable_fields(self):
        rows = [event(0, lane=2), event(1_000_000, lane=2)]
        result = score_condition_outcome('mania_4k', rows, (0, 8_000_000),
            {'jack_ratio': [.8, 1.]}, complete=True, legality_errors=[])
        self.assertEqual(result['reward'], 0.)
        with self.assertRaisesRegex(ValueError, 'inapplicable'):
            self.score([], {'jack_ratio': [0., .02]})

    def test_errors_are_not_silently_masked(self):
        with self.assertRaises(ValueError):
            interval_violation('drag_ratio', float('nan'), [.8, 1.])
        with self.assertRaises(ValueError):
            self.score([], {'drag_ratio': [.25, .75]})
        with self.assertRaises(ValueError):
            self.score([event(10, 'hold', 5)], {})


if __name__ == '__main__':
    unittest.main()
