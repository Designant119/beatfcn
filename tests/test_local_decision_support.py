import unittest

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest, encode_events
from scripts.audit_local_decision_support import inspect_request, summarize


class DecisionSupportTest(unittest.TestCase):
    def test_partial_cells_multiplicity_copy_and_crossing_tail(self):
        request = EventRequest('phira', 10_500, 85_000, 200_000, copy_times=(10_500,))
        events = [RhythmEvent(10_500, 10_500, 'drag', -1),
                  RhythmEvent(10_500, 160_000, 'hold', -1),
                  RhythmEvent(84_000, 84_000, 'flick', -1)]
        result = inspect_request(encode_events(events, request), request, events)
        self.assertEqual((result['cells'], result['occupied_cells'], result['cells_with_multiple_heads']), (3, 2, 1))
        self.assertEqual(result['max_heads_per_cell'], 2)
        self.assertEqual(result['outgoing_holds'], 1)
        self.assertTrue(result['partial_first_cell'] and result['partial_last_cell'])
        self.assertEqual(result['forced_fields']['member'], 2)

    def test_forced_bos_has_no_classifier_weight(self):
        request = EventRequest('mania_4k', 40_000, 80_000, 200_000,
            known_carry=tuple(RhythmEvent(0, 160_000, 'hold', lane) for lane in range(4)))
        result = inspect_request(encode_events([], request), request, [])
        self.assertEqual((result['tokens'], result['choice_tokens'], result['bos_support_size']), (1, 0, 1))
        result.update(record_index=0, component='song')
        aggregate = summarize([result])
        self.assertIsNone(aggregate['original_constant_bos_end_optimum'])
        self.assertEqual(aggregate['bos_forced_requests'], 1)

    def test_empty_request_is_retained(self):
        request = EventRequest('phira', 0, 80_000, 200_000)
        empty = inspect_request(encode_events([], request), request, [])
        events = [RhythmEvent(40_000, 40_000, 'tap', -1)]
        nonempty = inspect_request(encode_events(events, request), request, events)
        for i, row in enumerate((empty, nonempty)):
            row.update(record_index=i, component='song')
        aggregate = summarize([empty, nonempty])
        self.assertEqual(aggregate['empty_requests'], 1)
        self.assertEqual(aggregate['hypothetical_equal_bos_decision_end_fraction'], .5)
        self.assertGreater(aggregate['original_constant_bos_end_optimum'], .5)
        self.assertEqual(aggregate['empty_cell_fraction'], .75)

    def test_mismatched_target_multiset_rejected(self):
        request = EventRequest('phira', 0, 80_000, 200_000)
        events = [RhythmEvent(40_000, 40_000, 'tap', -1)]
        with self.assertRaisesRegex(ValueError, 'target multiset'):
            inspect_request(encode_events(events, request), request, [])


if __name__ == '__main__':
    unittest.main()
