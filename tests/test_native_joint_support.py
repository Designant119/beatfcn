from collections import Counter
from dataclasses import replace
import unittest

import numpy as np

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest
from beatfcn.train.native_joint_support import NativeJointSupport,NativeMultiset


class NativeJointSupportTests(unittest.TestCase):
    def setUp(self):
        self.codec=NativeJointSupport()
        self.request=EventRequest('phira',0,80000,200000,copy_times=(1501,20000))

    def test_multiset_COPY_native_alias_short_intervals_and_cross_window_tail(self):
        events=[RhythmEvent(1501,90000,'hold'),RhythmEvent(20000,20000,'tap'),
            RhythmEvent(20000,20000,'tap'),RhythmEvent(21000,21000,'drag'),RhythmEvent(79999,79999,'flick')]
        request=replace(self.request,copy_times=self.request.copy_times+(79999,))
        coded=self.codec.encode(events,request)
        self.assertEqual(Counter(self.codec.decode(coded,request)),Counter(events))
        self.assertEqual(coded.members[0].delta_ms,89)
        a,v=self.codec.addresses(request)
        self.assertEqual(int(((a==20000)&v).sum()),1)
        self.assertEqual(coded.counts.tolist(),[4,1])

    def test_fractional_request_origin_and_partial_last_cell(self):
        r=EventRequest('phira',1501,41502,90000,copy_times=(1501,41501))
        events=[RhythmEvent(1501,2000,'hold'),RhythmEvent(2000,2000,'tap'),RhythmEvent(41501,41501,'drag')]
        self.assertEqual(Counter(self.codec.decode(self.codec.encode(events,r),r)),Counter(events))
        a,v=self.codec.addresses(r)
        self.assertTrue(bool((a[v]>=r.start_us).all() and (a[v]<r.end_us).all()))

    def test_translation_preserves_addresses_and_relative_tail_labels(self):
        e=[RhythmEvent(1501,90000,'hold'),RhythmEvent(20000,20000,'flick')];shift=2_000_000
        r=replace(self.request,start_us=shift,end_us=self.request.end_us+shift,
            duration_us=self.request.duration_us+shift,copy_times=tuple(t+shift for t in self.request.copy_times))
        a=self.codec.encode(e,self.request);b=self.codec.encode([replace(v,onset_us=v.onset_us+shift,end_us=v.end_us+shift) for v in e],r)
        self.assertEqual(a.members,b.members);np.testing.assert_array_equal(a.counts,b.counts)

    def test_empty_is_valid_and_support_does_not_depend_on_targets(self):
        empty=self.codec.encode([],self.request);self.assertEqual(self.codec.decode(empty,self.request),[])
        a=self.codec.addresses(self.request)
        self.codec.encode([RhythmEvent(1000,1000,'tap')],self.request)
        b=self.codec.addresses(self.request)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)

    def test_capacity_and_unauthorized_COPY_fail_explicitly(self):
        events=[RhythmEvent(1000,1000,'tap')]*65
        with self.assertRaisesRegex(ValueError,'multiplicity overflow'):self.codec.encode(events,self.request)
        with self.assertRaises(ValueError):self.codec.encode([RhythmEvent(1502,1502,'tap')],self.request)
        bad=replace(self.request,copy_times=tuple(range(1001,1018)))
        with self.assertRaisesRegex(ValueError,'COPY address overflow'):self.codec.addresses(bad)

    def test_decoder_refuses_missing_members_overflow_and_bad_tail(self):
        e=self.codec.encode([RhythmEvent(1000,80000,'hold')],self.request)
        with self.assertRaises(ValueError):self.codec.decode(NativeMultiset(e.counts,()),self.request)
        counts=e.counts.copy();counts[0]=65
        with self.assertRaises(ValueError):self.codec.decode(NativeMultiset(counts,e.members),self.request)
        with self.assertRaises(ValueError):self.codec.decode(NativeMultiset(e.counts,(replace(e.members[0],delta_ms=0),)),self.request)
