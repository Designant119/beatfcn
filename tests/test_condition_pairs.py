from dataclasses import replace
import unittest

import numpy as np

from beatfcn.events.canonical import Event,LineLoc
from beatfcn.events import controls as C
from beatfcn.events.validate import validate_request
from beatfcn.model import features as F
from beatfcn.model.batch import SongInfo
from beatfcn.train.condition_pairs import blank_context_excerpt,assert_same_generation_inputs,without_conditions
from events_helpers import _chart,DURATION


class ConditionPairTests(unittest.TestCase):
    def examples(self):
        loc=LineLoc('0','above',0.)
        target_a=[Event('a1',9_000_000,9_000_000,'tap',loc),Event('a2',10_000_000,11_000_000,'hold',loc)]
        target_b=[Event('b1',9_200_000,9_200_000,'drag',loc),Event('b2',10_200_000,10_200_000,'flick',loc),
                  Event('b3',11_000_000,11_000_000,'tap',loc)]
        outside=[Event('outside',1_000_000,1_000_000,'tap',loc)]
        samples=[];examples=[]
        for target in (target_a,target_b):
            chart=_chart('phira',outside+target)
            sample=blank_context_excerpt(chart,8_000_000,DURATION,'a'*64,4)
            samples.append(sample)
            examples.append(F.build_core(sample,chart.events,8_000_000,DURATION,stage_a_arm='legacy'))
        return samples,examples

    def test_real_excerpt_contract_keeps_targets_and_has_all_condition_labels(self):
        samples,examples=self.examples()
        for sample,ex,n in zip(samples,examples,(2,3)):
            self.assertEqual(validate_request(sample.request,scene_line_ids=set()),[])
            self.assertEqual(len(sample.hidden),n)
            self.assertEqual(sample.request['observed_events'],[])
            self.assertEqual(set(sample.request['controls'])-{'allowed_types'},set(C.MODE_FIELDS['phira']))
            self.assertEqual(ex.known.shape[0],0)
            self.assertEqual(float(ex.blk_counts.sum()),0.)
            self.assertEqual(F.check_targets(ex),[])

    def test_initial_inputs_identical_except_conditions_and_catch_context_leak(self):
        _,examples=self.examples()
        args=(SongInfo('same',750,15),(9,16,24),24,32,32)
        result=assert_same_generation_inputs(*examples,*args)
        self.assertTrue(result['non_condition_inputs_equal'])
        self.assertIn('drag_ratio',result['changed_fields'])
        self.assertTrue(np.all(without_conditions(examples[0]).controls==-1))
        changed=replace(examples[1],blk_counts=examples[1].blk_counts+1)
        with self.assertRaisesRegex(ValueError,'non-condition input'):
            assert_same_generation_inputs(examples[0],changed,*args)

    def test_boundary_hold_rejects_whole_excerpt(self):
        loc=LineLoc('0','above',0.)
        for onset,end in ((7_000_000,9_000_000),(15_000_000,17_000_000)):
            chart=_chart('phira',[Event('cross',onset,end,'hold',loc)])
            with self.assertRaisesRegex(ValueError,'boundary hold'):
                blank_context_excerpt(chart,8_000_000,DURATION,'a'*64,4)


if __name__=='__main__':unittest.main()
