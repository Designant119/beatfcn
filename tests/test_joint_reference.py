import unittest

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest
from beatfcn.train.joint_reference import joint_reference_inputs


class JointReferenceTests(unittest.TestCase):
    def setUp(self):
        self.request=EventRequest('phira',40000,80000,100000)

    def encode(self,visible,regions=(),role='human_input'):
        return joint_reference_inputs(self.request,'a'*64,visible,regions,source_id='authorized-test',role=role)

    def test_empty_complete_regions_still_encode_known_silence(self):
        x,p=self.encode([],((0,40000),(80000,100000)))
        self.assertEqual(int(x['reference'][:,14].sum()),2)
        self.assertEqual(int(x['reference'][:,15].sum()),1)
        self.assertEqual(int(x['reference'][:,10].sum()),0)
        self.assertEqual(len(p),3)

    def test_duplicates_and_incoming_hold_tail_remain_visible(self):
        e=[RhythmEvent(1000,90000,'hold'),RhythmEvent(20000,20000,'drag'),RhythmEvent(20000,20000,'drag')]
        x,p=self.encode(e,((0,40000),))
        self.assertEqual(int(x['reference'][:,10].sum()),3)
        self.assertEqual(len({v['event_id'] for v in p if v.get('event_id')}),3)
        self.assertEqual(next(v['interval_us'][1] for v in p if v.get('event_id') and v['interval_us'][0]==1000),90000)

    def test_sparse_positive_reference_does_not_mark_missing_events_as_silence(self):
        x,p=self.encode([RhythmEvent(50000,50000,'flick')],role='model_history')
        self.assertEqual(int(x['reference'][:,14].sum()),0)
        self.assertEqual(int(x['reference'][:,15].sum()),1)
        self.assertEqual(int(x['reference'][:,12].sum()),1)
        self.assertEqual(int(x['reference'][:,11].sum()),0)

    def test_overlapping_known_region_is_rejected(self):
        with self.assertRaises(ValueError):self.encode([],((30000,50000),))
