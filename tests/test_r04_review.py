from copy import deepcopy
import json
from pathlib import Path
import unittest

from scripts.review_r04 import decide, summarize


ROOT = Path(__file__).resolve().parents[1]


class R04ReviewTests(unittest.TestCase):
    def setUp(self):
        self.gate=json.loads((ROOT/'bench/rhythm-v2/r04-initial-loss-v1.json').read_text())['quality_gate']
        self.control=dict(complete=True,macro_full_note_f1=.1,macro_timing_point_f1=.3,
            per_mode_full_note_f1={'mania_4k':.1,'phira':.1},
            types={'phira:hold':{'recall':.4,'pred':10},'mania_4k:tap':{'recall':.1,'pred':10}})
        self.candidate=deepcopy(self.control)
        self.candidate.update(macro_full_note_f1=.2,macro_timing_point_f1=.4,first_onset_matches=3,
            empty_windows=2,aggregate_count_ratio=1.25,all_requests_complete_legal=True,passed_fields=0)

    def test_partial_objective_gain_cannot_claim_conditions_or_product_completion(self):
        result=decide(self.control,self.candidate,self.gate)
        self.assertTrue(result['short_window_gate_passed'])
        self.assertEqual(result['control_fields_passed'],0)
        self.assertFalse(result['product_promotion'])

    def test_aggregate_gain_does_not_hide_type_collapse(self):
        self.candidate['types']['phira:hold']={'recall':.35,'pred':5}
        result=decide(self.control,self.candidate,self.gate)
        self.assertFalse(result['short_window_gate_passed'])
        self.assertIn('phira:hold recall regressed',result['reasons'])
        self.assertIn('phira:hold predictions collapsed',result['reasons'])

    def test_first_note_improvement_does_not_override_other_failures(self):
        self.candidate.update(first_onset_matches=8,empty_windows=3,aggregate_count_ratio=1.26,
                              all_requests_complete_legal=False)
        result=decide(self.control,self.candidate,self.gate)
        self.assertEqual(len(result['reasons']),3)
        self.assertFalse(result['short_window_gate_passed'])

    def test_existing_control_recomputes_and_fails_as_its_own_intervention(self):
        directory=ROOT/'reports/r03_window_raw'
        control=summarize(json.loads((directory/'r03-window-density.benchmark.json').read_text()),
                          json.loads((directory/'panel.json').read_text()))
        self.assertEqual(control['first_onset_matches'],0)
        self.assertEqual(control['empty_windows'],2)
        self.assertAlmostEqual(control['macro_full_note_f1'],.08677979579618923)
        result=decide(control,control,self.gate)
        self.assertFalse(result['short_window_gate_passed'])
        self.assertIn('first-onset match gate failed',result['reasons'])


if __name__=='__main__': unittest.main()
