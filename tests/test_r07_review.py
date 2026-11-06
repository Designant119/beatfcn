from copy import deepcopy
import json
from pathlib import Path
import unittest

from scripts.review_r07 import decide


class R07ReviewTests(unittest.TestCase):
    def setUp(self):
        self.gate=json.loads((Path(__file__).resolve().parents[1]/'bench/rhythm-v2/r07-blank-context-v1.json').read_text())['quality_gate']
        self.control=dict(complete=True,macro_full_note_f1=.1,macro_timing_point_f1=.4,
            per_mode_full_note_f1={'mania_4k':.1,'phira':.1},types={'phira:hold':{'recall':.3,'pred':10}},
            factual=[{'first_onset_error_ms':1000.}]*8,control_fields={'mania_4k:density':{'passed':False}})
        self.candidate=deepcopy(self.control)
        self.candidate.update(macro_full_note_f1=.15,macro_timing_point_f1=.5,first_onset_matches=4,
            empty_windows=0,aggregate_count_ratio=.8,all_requests_complete_legal=True,passed_fields=0,
            factual=[{'first_onset_error_ms':10.}]*4+[{'first_onset_error_ms':500.}]*4)

    def test_gain_is_only_research_retention(self):
        verdict=decide(self.control,self.candidate,self.gate)
        self.assertTrue(verdict['short_window_gate_passed'])
        self.assertFalse(verdict['product_promotion'])

    def test_adding_early_false_notes_cannot_win_by_first_onset_alone(self):
        self.candidate['macro_full_note_f1']=.09
        self.candidate['aggregate_count_ratio']=2.
        verdict=decide(self.control,self.candidate,self.gate)
        self.assertFalse(verdict['short_window_gate_passed'])
        self.assertIn('full-note gain below threshold',verdict['reasons'])
        self.assertIn('note overproduction',verdict['reasons'])

    def test_type_collapse_and_lost_control_are_rejected(self):
        self.candidate['types']['phira:hold']['pred']=0
        self.control['control_fields']['mania_4k:density']['passed']=True
        verdict=decide(self.control,self.candidate,self.gate)
        self.assertFalse(verdict['short_window_gate_passed'])
        self.assertIn('phira:hold predictions collapsed',verdict['reasons'])
        self.assertIn('previously working condition lost',verdict['reasons'])

    def test_more_empty_windows_or_late_first_points_fail(self):
        self.candidate.update(empty_windows=3,first_onset_matches=1,factual=[{'first_onset_error_ms':900.}]*5)
        verdict=decide(self.control,self.candidate,self.gate)
        self.assertFalse(verdict['short_window_gate_passed'])
        self.assertIn('first-onset match gate failed',verdict['reasons'])
        self.assertIn('first-onset error did not fall sufficiently',verdict['reasons'])


if __name__=='__main__': unittest.main()
