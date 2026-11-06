import math
from types import SimpleNamespace
import unittest

import numpy as np

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest
from beatfcn.train.native_quality_metrics import teacher_scores
from scripts.run_native_cell_recovery import output_null


class QualityMetricTests(unittest.TestCase):
    def test_binary_proper_score_keeps_negative_examples(self):
        steps=[dict(cell=0,rank=0,kind_allowed=[True]*4,type_probability=[.4,.2,.2,.2],event=dict(onset_us=0,end_us=0,type='tap'))]
        target=dict(counts=np.array([1]),members=[[SimpleNamespace(kind=0)]])
        r=teacher_scores(steps,target,EventRequest('phira',0,100000,100000),[[.1,.9]],[1,1,1,1])
        self.assertFalse(r['type_binary']['drag'][0]['positive'])
        self.assertAlmostEqual(r['type_binary']['drag'][0]['nll'],-math.log(.8))
        self.assertAlmostEqual(r['count_nll'],-math.log(.9))

    def test_invalid_null_hold_stays_in_denominator_but_cannot_match_tail(self):
        request=EventRequest('phira',0,1000000,1000000)
        predicted=[RhythmEvent(0,1000,'hold')]
        target=[RhythmEvent(999000,1000000,'hold')]
        result=output_null(predicted,target,request,[999500])[0]
        self.assertEqual(result['onset10'],1.)
        self.assertEqual(result['invalid_endpoint_events'],1)
        self.assertEqual(result['types']['hold']['pred'],1)
        self.assertEqual(result['types']['hold']['tp'],0)
        self.assertEqual(result['full50'],0.)
        self.assertEqual(predicted[0].end_us,1000)


if __name__=='__main__':unittest.main()
