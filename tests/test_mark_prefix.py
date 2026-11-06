import unittest

import numpy as np
import torch

from beatfcn.model.anchor_mark_chain import AnchorMarkChain,mark_loss
from beatfcn.train.mark_prefix import prefix_outputs,self_prefix_loss,_PrefixForward
from tests import test_anchor_mark_chain as fixtures


class MarkPrefixTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(5)
        from beatfcn.train.anchor_marks import teacher_example
        self.x,self.y,_=teacher_example(fixtures.AnchorMarkTests.row());self.model=AnchorMarkChain()
        self.u=np.array([.02,.5,.99])

    def test_teacher_replay_matches_original_heads_and_joint_loss(self):
        expected=self.model.teacher(self.x,self.y['kind'])
        actual,_=prefix_outputs(self.model,self.x,replay_ids=self.y['kind'])
        for name in expected:torch.testing.assert_close(actual[name],expected[name],atol=1e-6,rtol=1e-6)
        a,_=mark_loss(self.model,self.x,self.y)
        b,_=mark_loss(_PrefixForward(self.model,None,self.y['kind']),self.x,self.y)
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)

    def test_self_prefix_does_not_read_labels_and_inputs_follow_sampled_ids(self):
        proxy=_PrefixForward(self.model,self.u)
        a=proxy.teacher(self.x,self.y['kind']);b=proxy.teacher(self.x,(self.y['kind']+1)%4)
        for key in a:torch.testing.assert_close(a[key],b[key])
        _,detail=prefix_outputs(self.model,self.x,self.u,trace=True)
        used=torch.cat(detail['previous_ids'],1)
        self.assertEqual(used[0,0].item(),4)
        torch.testing.assert_close(used[:,1:],detail['sampled_ids'][:,:-1])
        self.assertTrue(bool((used[:,1:]!=self.y['kind'][:,:-1]).any()))

    def test_discrete_choice_is_stopped_but_hidden_bptt_remains(self):
        _,trace=prefix_outputs(self.model,self.x,self.u,trace=True)
        trace['step_logits'][1].square().sum().backward()
        grad=trace['step_logits'][0].grad
        self.assertTrue(grad is None or float(grad.abs().max())==0.)
        self.assertIsNotNone(trace['hidden_states'][0].grad)
        self.assertGreater(float(trace['hidden_states'][0].grad.abs().sum()),0.)

    def test_joint_supervised_risk_has_finite_gradients(self):
        loss,parts=self_prefix_loss(self.model,self.x,self.y,self.u);loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(parts['events'],3);self.assertEqual(parts['holds'],1)
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.model.parameters()))
