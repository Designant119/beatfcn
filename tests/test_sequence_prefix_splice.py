import unittest

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest, EventState
from scripts.audit_sequence_prefix_splice import histories


class PrefixSpliceTest(unittest.TestCase):
    def test_real_anchor_keeps_multiset_and_long_hold_tail(self):
        request=EventRequest('phira',0,8000000,12000000)
        truth=[RhythmEvent(1000000,1000000,'tap'),RhythmEvent(3000000,9000000,'hold'),
               RhythmEvent(3000000,3000000,'drag'),RhythmEvent(5000000,5000000,'flick')]
        generated=[RhythmEvent(500000,500000,'tap'),RhythmEvent(2999000,2999000,'tap')]
        result=histories(truth,generated,request)
        self.assertEqual(result['anchor_us'],3000000)
        states=[]
        for prefix in result['prefixes']:
            s=EventState(request)
            for token in prefix:s.apply(token)
            states.append(s)
        self.assertEqual([e for e in states[1].events if e.onset_us==3000000],
                         [e for e in states[0].events if e.onset_us==3000000])
        self.assertIn(RhythmEvent(3000000,9000000,'hold'),states[1].events)

    def test_missing_real_anchor_is_explicit(self):
        with self.assertRaisesRegex(ValueError,'Insufficient'):
            histories([],[],EventRequest('phira',0,8000000,12000000))


if __name__=='__main__':unittest.main()
