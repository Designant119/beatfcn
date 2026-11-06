from copy import deepcopy
import json
from pathlib import Path
import unittest

from scripts.review_r05 import decide


class R05ReviewTests(unittest.TestCase):
    def setUp(self):
        self.gate=json.loads((Path(__file__).resolve().parents[1]/'bench/rhythm-v2/r05-direct-condition-v1.json').read_text())['quality_gate']
        self.control=dict(complete=True,macro_full_note_f1=.1,macro_timing_point_f1=.4,
            per_mode_full_note_f1={'mania_4k':.1,'phira':.1},types={'phira:hold':{'recall':.3,'pred':10}})
        self.candidate=deepcopy(self.control)
        fields=['mania_4k:density_notes_per_second','mania_4k:hold_head_ratio','phira:chord_ratio','phira:drag_ratio']
        self.candidate.update(first_onset_matches=0,empty_windows=2,aggregate_count_ratio=.5,
            all_requests_complete_legal=True,passed_fields=4,control_fields={k:{'passed':True} for k in fields})

    def test_four_strict_controls_are_research_retention_only(self):
        result=decide(self.control,self.candidate,self.gate)
        self.assertTrue(result['short_window_gate_passed'])
        self.assertFalse(result['product_promotion'])

    def test_four_fields_only_one_mode_do_not_pass(self):
        self.candidate['control_fields']={f'mania_4k:{k}':{'passed':True} for k in
            ['density_notes_per_second','peak_notes_per_second','hold_head_ratio','chord_ratio']}
        result=decide(self.control,self.candidate,self.gate)
        self.assertFalse(result['short_window_gate_passed'])
        self.assertIn('control gains do not cover both modes',result['reasons'])

    def test_structure_only_response_and_note_collapse_are_rejected(self):
        self.candidate['control_fields']={f'{m}:{k}':{'passed':True} for m in ('mania_4k','phira')
            for k in ('density_notes_per_second','chord_ratio')}
        self.candidate['types']['phira:hold']['pred']=0
        result=decide(self.control,self.candidate,self.gate)
        self.assertFalse(result['short_window_gate_passed'])
        self.assertIn('control gains must include type and structure',result['reasons'])
        self.assertIn('phira:hold predictions collapsed',result['reasons'])

    def test_condition_gain_does_not_override_music_or_legality_regression(self):
        self.candidate.update(macro_timing_point_f1=.3,all_requests_complete_legal=False)
        result=decide(self.control,self.candidate,self.gate)
        self.assertFalse(result['short_window_gate_passed'])
        self.assertIn('timing-point gain below threshold',result['reasons'])


if __name__=='__main__':unittest.main()
