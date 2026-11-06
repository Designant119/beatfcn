from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from beatfcn.model.anchor_mark_chain import AnchorMarkChain,mark_loss
from beatfcn.model.canvas_diffusion import legal_tail_nll
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.train.anchor_marks import anchor_inputs,nearest,teacher_example


class AnchorMarkTests(unittest.TestCase):
    @staticmethod
    def row():
        rng=np.random.default_rng(7)
        inputs=dict(local=rng.normal(size=(2,9,1,1024)).astype(np.float32),local_times=np.arange(9)*.01,
            local_valid=np.ones(9,bool),mel=rng.normal(size=(9,128)).astype(np.float32),mel_times=np.arange(9)*.01,
            mel_valid=np.ones(9,bool),song=rng.normal(size=(3,1024)).astype(np.float32),song_valid=np.ones(3,bool),
            reference=np.zeros((1,16),np.float32),reference_valid=np.zeros(1,bool),controls=np.zeros((8,6),np.float32))
        request=SimpleNamespace(start_us=0,end_us=80000,duration_us=100000,max_events=4096,copy_times=(1501,))
        return dict(request=request,inputs=inputs,targets=[RhythmEvent(1501,90000,'hold'),
            RhythmEvent(20000,20000,'flick'),RhythmEvent(20000,20000,'tap')])

    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(5)

    def test_features_do_not_depend_on_hidden_type_or_tail_and_keep_duplicates(self):
        row=self.row();x,y,heads=teacher_example(row)
        alternative={**row,'targets':[RhythmEvent(h,h,'drag') for h in heads]}
        z=anchor_inputs(alternative,heads)
        for key in x:torch.testing.assert_close(x[key],z[key])
        self.assertEqual(heads,[1501,20000,20000]);self.assertEqual(y['kind'].tolist(),[[1,0,2]])
        self.assertEqual(y['delta_ms'][0].item(),89)
        self.assertEqual((heads[0]//1000+y['delta_ms'][0].item())*1000,90000)

    def test_clock_translation_keeps_relative_targets_and_features(self):
        row=self.row();x,y,heads=teacher_example(row);offset=2000000
        q=row['request'];moved={**row,'request':SimpleNamespace(**{**vars(q),'start_us':q.start_us+offset,
            'end_us':q.end_us+offset,'duration_us':q.duration_us+offset,'copy_times':(1501+offset,)}),
            'targets':[RhythmEvent(e.onset_us+offset,e.end_us+offset,e.type) for e in row['targets']]}
        z,t,_=teacher_example(moved)
        for key in x:torch.testing.assert_close(x[key],z[key])
        torch.testing.assert_close(y['delta_ms'],t['delta_ms'])

    def test_legal_joint_tail_normalizer_matches_bruteforce_and_has_gradients(self):
        b=torch.randn(3,requires_grad=True);c=torch.randn(50,requires_grad=True);m=torch.randn(40,requires_grad=True)
        lo,hi,target=1,4567,2055;values=torch.arange(lo,hi+1)
        scores=b[values//2000]+c[(values%2000)//40]+m[values%40]
        expected=torch.logsumexp(scores,0)-scores[target-lo]
        actual=legal_tail_nll(b,c,m,lo,hi,target)
        torch.testing.assert_close(actual,expected);actual.backward()
        self.assertTrue(all(bool(torch.isfinite(v.grad).all()) for v in (b,c,m)))

    def test_nearest_tie_uses_earlier_and_teacher_loss_is_finite(self):
        np.testing.assert_array_equal(nearest(np.array([0.,2.,4.]),np.ones(3,bool),np.array([1.,3.])),[0,1])
        x,y,_=teacher_example(self.row());model=AnchorMarkChain();loss,parts=mark_loss(model,x,y);loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)));self.assertEqual(parts['events'],3);self.assertEqual(parts['holds'],1)
        self.assertLess(sum(p.numel() for p in model.parameters()),2000000)
