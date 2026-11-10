import unittest

import numpy as np

from beatfcn.model.event_canvas import RhythmEvent as E
from beatfcn.model.full_event_codec import EventRequest, EventState, EventVocabulary, NEW
from beatfcn.model.full_event_codec import encode_events as absolute_encode
from beatfcn.model.relative_event_codec import RelativeHeadState, encode_events, teacher_sequence


class RelativeHeadCodecTest(unittest.TestCase):
    def roundtrip(self, events, request):
        tokens = encode_events(events, request)
        absolute = absolute_encode(events, request)
        state = RelativeHeadState(request)
        old = EventState(request)
        for a, b in zip(absolute, tokens, strict=True):
            self.assertEqual(old.vocab.decision(a)[0], state.vocab.decision(b)[0])
            old.apply(a)
            state.apply(b)
            self.assertEqual(old.events, state.events)
            self.assertEqual(old.last_onset, state.last_onset)
        self.assertTrue(state.done)
        self.assertEqual(sorted(state.events), sorted(events))
        seq = teacher_sequence(tokens, request)
        self.assertTrue(np.isfinite(seq['times']).all())
        self.assertTrue(seq['allowed'][np.arange(len(tokens)), seq['targets']].all())
        return tokens, seq

    def test_phira_all_types_multiplicity_copy_and_cross_window_hold(self):
        request = EventRequest('phira', 8_000_123, 16_000_001, 30_000_000,
            copy_times=(8_000_123, 9_000_500), known_carry=(E(1_000_000, 20_000_000, 'hold'),))
        events = [E(8_000_123, 8_000_123, 'drag'), E(9_000_500, 25_000_000, 'hold'),
            E(9_001_000, 9_001_000, 'tap'), E(9_001_000, 9_001_000, 'tap'),
            E(15_999_000, 15_999_000, 'flick'), E(16_000_000, 16_000_000, 'drag')]
        self.roundtrip(events, request)

    def test_mania_incoming_occupancy_and_multiple_hold_tails(self):
        request = EventRequest('mania_4k', 8_000_000, 16_000_000, 25_000_000,
            known_carry=(E(1_000_000, 12_000_000, 'hold', 0),))
        events = [E(8_000_000, 20_000_000, 'hold', 1), E(8_000_000, 11_000_000, 'hold', 2),
            E(8_000_000, 8_000_000, 'tap', 3), E(12_000_000, 12_000_000, 'tap', 0),
            E(15_000_000, 18_000_000, 'hold', 2)]
        self.roundtrip(events, request)

    def test_non_grid_start_long_rest_and_empty_request(self):
        req = EventRequest('phira', 21_123_456, 31_777_789, 40_000_000)
        self.roundtrip([], req)
        tokens, seq = self.roundtrip([E(21_124_000, 21_124_000, 'tap'),
            E(30_999_000, 30_999_000, 'drag')], req)
        state = RelativeHeadState(req)
        queries = []
        for token in tokens:
            if state.phase in ('head_cell', 'head_ms'):
                queries.append(state.query()[1])
            state.apply(token)
        self.assertAlmostEqual(queries[0], -.000456)
        self.assertTrue(any(t > 8 for t in queries))

    def test_relative_coordinates_recover_every_allowed_physical_ms(self):
        # Exhaust a short request across the 2s/40ms boundaries, including non-ms COPY base.
        req = EventRequest('phira', 3_987_321, 4_111_234, 8_000_000, copy_times=(4_000_500,))
        for prefix in ([], [E(4_000_500, 4_000_500, 'drag')]):
            for onset in range(3_988_000 if not prefix else 4_001_000, 4_112_000, 1000):
                self.roundtrip(prefix+[E(onset, onset, 'tap')], req)

    def test_tail_and_copy_tokens_do_not_change(self):
        req = EventRequest('phira', 7_000_000, 10_000_000, 20_000_000, copy_times=(7_011_999,))
        events = [E(7_011_999, 19_999_000, 'hold'), E(9_001_000, 9_001_000, 'flick')]
        v = EventVocabulary()
        old, new = absolute_encode(events, req), encode_events(events, req)
        for a,b in zip(old,new,strict=True):
            field,value=v.decision(a)
            if not field.startswith('head_') or field == 'head_block' and value >= v.blocks:
                self.assertEqual(a,b)

    def test_invalid_subms_permission_and_lane_conflict_still_reject(self):
        req = EventRequest('phira', 8_000_000, 9_000_000, 10_000_000)
        with self.assertRaises(ValueError):
            encode_events([E(8_000_001, 8_000_001, 'tap')],req)
        req = EventRequest('mania_4k',8_000_000,9_000_000,10_000_000,
            known_carry=(E(1_000_000,9_000_000,'hold',0),))
        with self.assertRaises(ValueError):
            encode_events([E(8_100_000,8_100_000,'tap',0)],req)

    def test_hold_only_final_sample_still_disallows_zero_length_tail(self):
        req = EventRequest('phira',7_998_500,8_000_000,8_000_000,allowed_types=('hold',))
        self.roundtrip([E(7_999_000,8_000_000,'hold')],req)
        state = RelativeHeadState(req)
        state.apply(state.vocab.token('group',NEW))
        self.assertEqual(state._heads_ms(),(1,1))


if __name__ == '__main__':
    unittest.main()
