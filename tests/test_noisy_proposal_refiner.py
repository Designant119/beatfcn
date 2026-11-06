from dataclasses import replace
import itertools
import unittest

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.noisy_proposal_refiner import monotone_proposal_matching,NoisyProposalRefiner,refiner_loss,REJECT
from beatfcn.train.noisy_proposals import proposal_inputs,proposal_targets
from tests import test_local_event_package as old


class NoisyProposalTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(7)

    def test_ordered_matching_equals_exhaustive_cardinality_distance_and_keeps_duplicates(self):
        proposals=[0,0,10,90];targets=[0,5,10]
        pairs=monotone_proposal_matching(proposals,targets,tolerance_us=6)
        feasible=[]
        for count in range(4):
            for ii in itertools.combinations(range(len(proposals)),count):
                for jj in itertools.combinations(range(len(targets)),count):
                    if all(abs(proposals[i]-targets[j])<=6 for i,j in zip(ii,jj)):
                        feasible.append((count,-sum(abs(proposals[i]-targets[j]) for i,j in zip(ii,jj))))
        self.assertEqual((len(pairs),-sum(abs(proposals[i]-targets[j]) for i,j in pairs)),max(feasible))
        self.assertEqual(monotone_proposal_matching([1,1],[1,1]),[(0,0),(1,1)])

    def test_head_soft_distribution_copy_identity_and_reject_are_separate_targets(self):
        request,_,_=old.EventPackageTests.synthetic_example();request=replace(request,copy_times=(1234,2345))
        target,report=proposal_targets([1000,1234,70000],[RhythmEvent(1234,2345,'hold'),RhythmEvent(1000,1000,'tap')],request)
        self.assertEqual(report['matched'],2);self.assertEqual(int((target['kind']==REJECT).sum()),1)
        self.assertAlmostEqual(float(target['head_distribution'][0,0].sum()),1.)
        self.assertEqual(float(target['head_distribution'][0,1,101]),1.)
        self.assertEqual(int(target['tail_copy'][0,1]),2)
        self.assertEqual(int(target['head_us'][0,1]),1234)

    def test_true_audio_to_proposal_attribute_graph_uses_no_target_in_forward(self):
        request,inputs,_=old.EventPackageTests.synthetic_example()
        inputs={k:v for k,v in inputs.items() if not k.startswith(('local','song'))}
        inputs.update(mel_flux=torch.randn(1,8,32),whole_song=torch.randn(1,2,140),whole_song_times=torch.tensor([[1.,3.]]),whole_song_valid=torch.ones(1,2,dtype=torch.bool))
        model=NoisyProposalRefiner(old.EventPackageTests.small_model().c,density_reference=4).eval()
        times=[3000,5500,70000];proposal=proposal_inputs(times,request)
        target,_=proposal_targets(times,[RhythmEvent(4000,2012000,'hold'),RhythmEvent(5000,5000,'flick')],request)
        out=model(inputs,proposal);self.assertEqual(out['kind'].shape,(1,3,5))
        loss,parts=refiner_loss(model,inputs,proposal,target);self.assertTrue(torch.isfinite(loss));loss.backward()
        for p in (model.kind.weight,model.head.weight,model.duration.weight,model.body.fine_projection.weight):
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(parts['head_target_entropy']),0)

    def test_native_copy_hold_legality_is_shared_with_soft_head_targets(self):
        request,_,_=old.EventPackageTests.synthetic_example();request=replace(request,end_us=1500,duration_us=1500)
        proposal=proposal_inputs([1000],request)
        heads=proposal['head_times'][0,0];legal=proposal['head_has_endpoint'][0,0]
        self.assertTrue(bool(legal[heads==0].all()));self.assertFalse(bool(legal[heads==1000].any()))
        target,_=proposal_targets([1000],[RhythmEvent(0,1000,'hold')],request)
        self.assertEqual(float(target['head_distribution'][0,0,heads==1000].sum()),0)
        self.assertAlmostEqual(float(target['head_distribution'].sum()),1.)


if __name__=='__main__':unittest.main()
