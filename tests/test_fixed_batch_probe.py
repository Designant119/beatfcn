import copy
import math
import unittest

import torch

from beatfcn.model import p1
from beatfcn.model.edit_utility import edit_utility
from beatfcn.events.canonical import Event, LaneLoc
from beatfcn.train.probe import tensor_summary, capture_decisions, metrics_json, assess_memorisation, assess_generation
from test_p1_model import CFG, batch


class FixedBatchProbeTests(unittest.TestCase):
    def test_stats_preserve_nonfinite_evidence_and_handle_empty(self):
        result=tensor_summary(torch.tensor([1.,3.,math.nan,math.inf,-math.inf]),chunk_size=2)
        self.assertEqual((result['finite'],result['nan'],result['posinf'],result['neginf']),(2,1,1,1))
        self.assertEqual((result['min'],result['max'],result['mean'],result['std']),(1.,3.,2.,1.))
        self.assertIsNone(tensor_summary(torch.empty(0))['mean'])

    def test_real_forward_capture_does_not_change_loss_and_excludes_padding(self):
        torch.manual_seed(21)
        model=p1.P1Model(CFG).eval()
        bt=batch(2,seed=3)
        original=p1.masked_ce
        with torch.no_grad():
            expected,_=model(bt)
            with capture_decisions() as decisions:
                loss,met=model(bt)
        self.assertIs(p1.masked_ce,original)
        torch.testing.assert_close(loss,expected)
        measured=metrics_json(loss,met)
        self.assertEqual(decisions['type']['count'],measured['heads']['type']['count'])
        self.assertEqual(sum(decisions['type']['truth_counts']),int((bt['grp_n']*bt['grp_valid']).sum()))
        self.assertEqual(sum(map(sum,decisions['type']['confusion_truth_rows_pred_columns'])),decisions['type']['count'])
        with self.assertRaisesRegex(RuntimeError,'stop'):
            with capture_decisions():raise RuntimeError('stop')
        self.assertIs(p1.masked_ce,original)

    def test_acceptance_rejects_bad_heads_and_incomplete_updates(self):
        gates=dict(final_loss_at_most=.1,reduction_at_least=.95,per_head_accuracy_at_least=.95)
        history=[dict(step=0,loss=10.,illegal=0,heads={}),
                 dict(step=100,loss=.01,illegal=0,heads={'type':dict(count=30,accuracy=1.)})]
        gradients=[dict(step=i,finite=True) for i in range(1,101)]
        self.assertTrue(assess_memorisation(history,gradients,gates,100)['passed'])
        for accuracy in (.9,math.nan,None):
            bad=copy.deepcopy(history);bad[-1]['heads']['type']['accuracy']=accuracy
            self.assertFalse(assess_memorisation(bad,gradients,gates,100)['passed'])
        self.assertFalse(assess_memorisation(history,gradients[:-1],gates,100)['passed'])
        self.assertFalse(assess_memorisation(history[:-1],gradients,gates,100)['passed'])

    def test_free_generation_gate_uses_hold_tail_and_detects_missing_core(self):
        truth=[Event('t',1_000_000,2_000_000,'hold',LaneLoc(0))]
        wrong=[Event('p',1_000_000,2_500_000,'hold',LaneLoc(0))]
        record=dict(chart_id='train:1',c0_us=0,complete=True,legal=True,
                    scores={'edit_utility':edit_utility(wrong,truth,'mania_4k')})
        self.assertFalse(assess_generation([record],1,.9)['passed'])
        record['scores']['edit_utility']=edit_utility(truth,truth,'mania_4k')
        self.assertTrue(assess_generation([record],1,.9)['passed'])
        self.assertFalse(assess_generation([record],2,.9)['passed'])
        self.assertFalse(assess_generation([record,record],2,.9)['passed'])
        record['complete']=False
        self.assertFalse(assess_generation([record],1,.9)['passed'])


if __name__=='__main__':unittest.main()
