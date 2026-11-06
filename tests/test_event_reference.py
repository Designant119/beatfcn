from dataclasses import replace
import unittest

import numpy as np

from beatfcn.events.canonical import Event, LaneLoc, LineLoc
from beatfcn.train.event_reference import VisibleReference, encode_reference


class EventReferenceTests(unittest.TestCase):
    def arguments(self):
        return dict(mode='phira', audio_sha256='a'*64, duration_us=20_000_000,
                    origin_us=8_000_000, unknown_regions=((8_000_000, 16_000_000),))

    def source(self, events=()):
        return VisibleReference('human-chart', 'phira', 'a'*64, 'human_input',
                                ((0, 8_000_000),), tuple(events))

    def test_blank_is_exactly_the_frozen_local_loader_contract(self):
        inputs, provenance = encode_reference(**self.arguments())
        expected = np.zeros((1, 16), np.float32)
        expected[0, 1] = 8_000_000/1_200_000_000; expected[0, 15] = 1
        np.testing.assert_array_equal(inputs['reference'], expected)
        np.testing.assert_array_equal(inputs['reference_times'], [0.])
        self.assertEqual(provenance[0]['role'], 'unknown')

    def test_known_silence_is_distinct_from_unknown(self):
        inputs, _ = encode_reference(**self.arguments(), sources=(self.source(),))
        self.assertEqual(inputs['reference'].shape, (2, 16))
        self.assertEqual(inputs['reference'][1, 14], 1)
        self.assertEqual(inputs['reference'][1, 15], 0)
        self.assertEqual(inputs['reference'][1, 11], 1)

    def test_visible_hold_keeps_known_tail_and_does_not_round(self):
        event = Event('hold', 7_000_123, 10_000_987, 'hold', LineLoc('1', 'above', 0.))
        inputs, provenance = encode_reference(**self.arguments(), sources=(self.source((event,)),))
        self.assertEqual(provenance[-1]['interval_us'], [7_000_123, 10_000_987])
        self.assertEqual(inputs['reference'][-1, 3], 1)
        self.assertEqual(inputs['reference'][-1, 10], 1)

    def test_hidden_target_and_audio_swap_are_rejected(self):
        event = Event('hidden', 9_000_000, 9_000_000, 'drag', LineLoc('1', 'above', 0.))
        with self.assertRaisesRegex(ValueError, 'outside'):
            encode_reference(**self.arguments(), sources=(self.source((event,)),))
        with self.assertRaisesRegex(ValueError, 'identity'):
            encode_reference(**self.arguments(), sources=(replace(self.source(), audio_sha256='b'*64),))
        with self.assertRaisesRegex(ValueError, 'overlaps unknown'):
            encode_reference(**self.arguments(), sources=(replace(self.source(), regions=((0, 10_000_000),)),))

    def test_artistic_drag_multiplicity_is_preserved_and_source_marked(self):
        event = Event('a', 7_000_000, 7_000_000, 'drag', LineLoc('1', 'above', 0.))
        source = replace(self.source((event, replace(event, event_id='b'))), role='model_history')
        inputs, provenance = encode_reference(**self.arguments(), sources=(source,))
        self.assertEqual(int(inputs['reference'][:, 10].sum()), 2)
        self.assertEqual(int(inputs['reference'][:, 12].sum()), 3)
        self.assertEqual([p['event_id'] for p in provenance[-2:]], ['a', 'b'])

    def test_capacity_does_not_silently_remove_reference_events(self):
        with self.assertRaisesRegex(ValueError, 'capacity'):
            encode_reference(**self.arguments(), sources=(self.source(),), max_rows=1)

    def test_mania_lane_identity_and_wrong_location(self):
        args = dict(self.arguments(), mode='mania_4k')
        event = Event('key', 1000, 1000, 'tap', LaneLoc(3))
        source = replace(self.source((event,)), mode='mania_4k')
        inputs, _ = encode_reference(**args, sources=(source,))
        self.assertEqual(inputs['reference'][-1, 9], 1)
        with self.assertRaisesRegex(ValueError, 'mode/region'):
            encode_reference(**args, sources=(replace(source, events=(replace(event, location=LineLoc('0', 'above', 0.)),)),))


if __name__ == '__main__':
    unittest.main()
