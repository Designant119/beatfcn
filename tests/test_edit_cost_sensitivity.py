import itertools
import random
import unittest

from beatfcn.events.canonical import Event,LaneLoc,LineLoc
from scripts.audit_edit_cost_sensitivity import minimum_assignment,cost_proxy


class EditCostSensitivityTests(unittest.TestCase):
    def test_assignment_agrees_with_bruteforce_including_negative_costs(self):
        rng=random.Random(22)
        for n in range(1,5):
            for _ in range(12):
                costs=[[rng.randrange(-8,20)/4 for j in range(n+2)] for i in range(n)]
                selected=minimum_assignment(costs)
                expected=min(sum(costs[i][j] for i,j in enumerate(cols)) for cols in itertools.permutations(range(n+2),n))
                self.assertEqual(len(set(selected)),n)
                self.assertEqual(sum(costs[i][j] for i,j in enumerate(selected)),expected)

    def test_wrong_lane_and_type_are_separate_edits(self):
        pred=[Event('p',100000,100000,'tap',LaneLoc(0))]
        truth=[Event('t',100000,300000,'hold',LaneLoc(1))]
        r=cost_proxy(pred,truth,'mania_4k',.25,0)
        self.assertEqual(r['operations'],dict(insert=0,delete=0,type=1,lane=1,duration=1))
        self.assertEqual(r['edit_cost'],.75)
        r=cost_proxy(pred,truth,'mania_4k',1,0)
        self.assertEqual(r['edit_cost'],2)

    def test_chords_are_one_to_one_and_phira_positions_are_not_scored(self):
        truth=[Event(str(i),100000,100000,'drag',LineLoc('0','above',float(i))) for i in range(2)]
        r=cost_proxy(truth[:1],truth,'phira',.25,.1)
        self.assertEqual(r['reused'],1);self.assertEqual(r['operations']['insert'],1)
        self.assertAlmostEqual(r['edit_cost'],1.1)
        reversed_notes=[Event('p',100000,100000,'drag',LineLoc('different','below',99.))]
        self.assertEqual(cost_proxy(reversed_notes,truth[:1],'phira',1,0)['edit_cost'],0)

    def test_silence_and_wrong_timing_do_not_become_free_benefit(self):
        truth=[Event('t',100000,100000,'tap',LaneLoc(0))]
        pred=[Event('p',121000,121000,'tap',LaneLoc(0))]
        self.assertEqual(cost_proxy([],truth,'mania_4k',.25,0)['saved_fraction'],0)
        self.assertEqual(cost_proxy(pred,truth,'mania_4k',.25,0)['saved_fraction'],-1)


if __name__=='__main__':unittest.main()
