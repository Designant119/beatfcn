from dataclasses import replace
import unittest

import numpy as np
import torch

from beatfcn.model.local_lattice import LatticeCanvas,lattice_loss,generate_lattice,expectation
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.train.lattice_targets import lattice_target,lattice_layout
from tests import test_local_event_package as old


class LatticeTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(9)

    def test_transform_preserves_subframe_and_multiset_identity_without_dropping(self):
        request,_,_=old.EventPackageTests.synthetic_example();request=replace(request,copy_times=(1234,2345))
        events=[RhythmEvent(1234,2345,'hold')]*2+[RhythmEvent(1000,1000,'flick'),RhythmEvent(2000,2000,'drag')]
        target=lattice_target(events,request)
        self.assertEqual(int(target['count'][0]),4)
        self.assertEqual(sorted(target['head_us'][target['active']].tolist()),[1000,1234,1234,2000])
        self.assertEqual(int((target['tail_copy']>0).sum()),2)
        with self.assertRaisesRegex(ValueError,'capacity overflow'):
            lattice_target([RhythmEvent(1000,1000,'tap')]*13,request)
        with self.assertRaisesRegex(ValueError,'Unauthorized lattice tail COPY'):
            lattice_target([RhythmEvent(1000,2234,'hold')],request)

    def example(self):
        request,inputs,_=old.EventPackageTests.synthetic_example()
        inputs.update({k:torch.from_numpy(np.asarray(v))[None] for k,v in lattice_layout(request).items()})
        events=[RhythmEvent(4000,2012000,'hold'),RhythmEvent(5000,5000,'flick')]
        target={k:torch.from_numpy(v)[None] for k,v in lattice_target(events,request).items()}
        model=LatticeCanvas(old.EventPackageTests.small_model().c,density_reference=4,type_prior=[.7,.1,.1,.1]).eval()
        return model,request,inputs,target

    def test_direct_density_and_ratio_paths_and_real_attribute_gradients(self):
        model,request,inputs,target=self.example()
        a=model(inputs);changed={k:v.clone() for k,v in inputs.items()}
        changed['controls'][0,0,:3]=torch.tensor([8/64,10/64,1.])
        b=model(changed)
        self.assertGreater(float(expectation(b,changed)[0]),float(expectation(a,inputs)[0]))
        torch.testing.assert_close(a['kind'],b['kind'],atol=0,rtol=0)
        loss,parts=lattice_loss(model,inputs,target);self.assertTrue(torch.isfinite(loss));loss.backward()
        parameters=dict(model.named_parameters())
        for name in ('presence.weight','multiplicity.weight','kind.weight','residual.weight','duration_fraction.weight','mel_projection.weight'):
            self.assertIsNotNone(parameters[name].grad,name);self.assertTrue(torch.isfinite(parameters[name].grad).all(),name)
        self.assertEqual(int(target['count'].sum()),2)

    def test_empty_primary_decoding_does_not_force_a_type_or_count(self):
        model,request,inputs,target=self.example()
        with torch.no_grad():model.presence.weight.zero_();model.presence.bias.fill_(-100.)
        self.assertEqual(generate_lattice(model,inputs,request),[])


if __name__=='__main__':unittest.main()
