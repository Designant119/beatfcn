from dataclasses import replace
import itertools
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from beatfcn.model.cardinality_dp import cardinality_counts
from beatfcn.model.canonical_event_package import CanonicalPackageCanvas,canonical_package_loss,generate_canonical,package_support
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.train.canonical_probability import encode_canonical
from beatfcn.train.local_typed_canvas import request_layout
from tests import test_local_event_package as old
from tests.test_local_event_package import RejectTargetReads


class CardinalityTests(unittest.TestCase):
    def test_polynomial_and_conditional_allocation_equal_exhaustive_enumeration(self):
        p=np.array([[.6,.3,.1],[.2,.7,.1],[.4,.2,.4]])
        counts,info=cardinality_counts(p)
        entries=[(c,float(np.prod([p[i,k] for i,k in enumerate(c)]))) for c in itertools.product(range(3),repeat=3)]
        exact=np.zeros(7)
        for c,weight in entries:exact[sum(c)]+=weight
        np.testing.assert_allclose(info['polynomial'],exact,rtol=1e-14,atol=1e-14)
        self.assertEqual(info['total'],int(exact.argmax()))
        self.assertAlmostEqual(np.exp(info['allocation_log_probability']),max(w for c,w in entries if sum(c)==info['total']))
        self.assertEqual(int(counts.sum()),info['total'])

    def test_cardinality_is_not_individual_mode_or_a_resource_truncated_law(self):
        p=np.tile([.95,.05],(100,1))
        c,info=cardinality_counts(p)
        self.assertEqual(info['total'],5);self.assertEqual(int(c.sum()),5)
        self.assertEqual(info['support_max'],100)
        with self.assertRaisesRegex(ValueError,'resource overflow'):cardinality_counts(p,max_events=4)
        c,info=cardinality_counts(np.zeros((0,17)))
        self.assertEqual(len(c),0);self.assertEqual(info['total'],0)


class CanonicalPackageTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(2001)
        self.request,self.inputs,_=old.EventPackageTests.synthetic_example()
        self.model=CanonicalPackageCanvas(old.EventPackageTests.small_model().c,density_reference=4).eval()

    def encoded(self,events,request=None):
        return {k:torch.from_numpy(v)[None] for k,v in encode_canonical(events,request or self.request).items()}

    def test_joint_objective_has_one_denominator_and_reaches_every_required_head(self):
        target=self.encoded([RhythmEvent(4000,2_012_000,'hold'),RhythmEvent(5000,5000,'flick')])
        loss,parts=canonical_package_loss(self.model,self.inputs,target,self.request)
        self.assertEqual(int(parts['valid_groups']),8)
        torch.testing.assert_close(loss,parts['count']+parts['onset']+parts['tail'])
        loss.backward()
        params=dict(self.model.named_parameters())
        for name in ('frame_read.weight','count_gate.weight','member_gru.weight_ih','onset_read.weight','tail_block.weight'):
            self.assertTrue(torch.isfinite(params[name].grad).all(),name)
            self.assertGreater(float(params[name].grad.abs().sum()),0,name)

    def test_native_copy_tail_is_one_normalized_union_and_preserves_duplicates(self):
        request=replace(self.request,end_us=6_500,duration_us=6_500,copy_times=(1234,2345,3000,3456))
        condition=torch.randn(32,requires_grad=True)
        targets=[2000,2345,3000,3456,4000,5000,6000]
        probabilities=[]
        for target in targets:
            end,embedding,nll=self.model.tail_law(condition,request,0,2000,target=target)
            self.assertEqual(end,target);probabilities.append((-nll).exp())
        torch.testing.assert_close(sum(probabilities),torch.tensor(1.),atol=1e-6,rtol=1e-6)
        nll=-probabilities[1].log();nll.backward()
        self.assertIsNotNone(self.model.copy_query.weight.grad)
        with self.assertRaisesRegex(ValueError,'authorization'):
            self.model.tail_law(condition,request,0,2000,target=2444)
        events=[RhythmEvent(1234,2345,'hold')]*2+[RhythmEvent(1234,3456,'hold')]
        inputs={k:v.clone() for k,v in self.inputs.items()}
        inputs.update({k:torch.from_numpy(np.asarray(v))[None] for k,v in request_layout(request).items()})
        target=self.encoded(events,request)
        x=self.model(inputs)['x'][0]
        members,_,_=self.model.ordered_members(x,0,1,3,inputs,request,
            teacher={k:target[k][0,0,1,:3] for k in ('offset','tail_us')})
        self.assertEqual(members,events)

    def test_full_count_tilt_and_no_density_in_audio(self):
        altered={k:v.clone() for k,v in self.inputs.items()}
        altered['controls'][0,0,:3]=torch.tensor([8/64,10/64,1.])
        a,b=self.model(self.inputs),self.model(altered)
        torch.testing.assert_close(a['x'],b['x'],atol=0,rtol=0)
        tilt=F.softplus(self.model.density_alpha)*torch.tensor(9/4).log()
        torch.testing.assert_close(b['count'],(a['count_log_probs']+tilt*torch.arange(17)).softmax(-1))

    def test_generation_never_reads_target_and_retains_event_multiplicity(self):
        with torch.no_grad():
            self.model.count_gate.weight.zero_();self.model.count_gate.bias.fill_(-100.)
        events,info=generate_canonical(self.model,RejectTargetReads(self.inputs),self.request)
        self.assertEqual(events,[]);self.assertEqual(info['total'],0)
        self.assertEqual(info['support_max'],128)
        request=replace(self.request,end_us=500,duration_us=500)
        inputs={k:v.clone() for k,v in self.inputs.items()}
        inputs.update({k:torch.from_numpy(np.asarray(v))[None] for k,v in request_layout(request).items()})
        self.assertFalse(bool(package_support(inputs,request)[0,0,1]))


if __name__=='__main__':unittest.main()
