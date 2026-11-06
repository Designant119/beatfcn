import unittest
import numpy as np

from beatfcn.model.event_canvas import ClockLayout, RhythmEvent
from beatfcn.model.group_canvas import encode_group_targets
from scripts.audit_draft_exposure import saved_payload, canvas_payload, draft_stats, lost_holds


class DraftExposureTest(unittest.TestCase):
    def test_ambiguous_group_ownership_remains_unresolved(self):
        e=dict(onset_us=100000,end_us=200000,type='hold',lane=0)
        saved=dict(member_counts=[[1,1]],raw_group_heads=[[[100,-1],[100,-1]]],events=[e,e],round=1)
        layout=ClockLayout(0,400000)
        _,_,payload,owners,issues=saved_payload(saved,layout)
        self.assertEqual(payload,{})
        self.assertEqual(len(issues),1)
        core={'target_events':[e],'identity':{'source_clip_start_us':0}}
        lost=lost_holds(core,saved,{'events':[],'round':2},layout,np.ones((1,2),bool),owners)
        self.assertEqual(lost[0]['multiplicity'],1)
        self.assertIsNone(lost[0]['was_visible'])

    def test_masked_payload_does_not_count_as_visible_and_tail_is_not_clipped(self):
        labels=encode_group_targets([RhythmEvent(100000,9000000,'hold',0)],
                                    ClockLayout(0,400000),'mania_4k',groups=2)
        stats=draft_stats(labels.multiplicity,labels.heads,np.zeros((1,2),bool),canvas_payload(labels),labels)
        self.assertEqual(stats['visible_notes'],0)
        self.assertEqual(stats['types'],{})
        stats=draft_stats(labels.multiplicity,labels.heads,np.ones((1,2),bool),canvas_payload(labels),labels)
        self.assertEqual(stats['hold_duration_us'],[8900000])
        self.assertEqual(stats['long_holds_ge8s'],1)
        self.assertEqual(stats['wrong_resolved_visible_true_slots'],0)


if __name__=='__main__':unittest.main()
