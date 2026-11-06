import copy
import unittest

import torch
from torch.nn import functional as F

from beatfcn.model.unified_count_canvas import UnifiedCountCanvas,unified_count_loss
from beatfcn.model.local_mamba_canvas import MambaCanvas
from beatfcn.model.local_evolution_controls import reset_mamba_boundaries
from tests import test_local_event_package as fixtures


class UnifiedCountTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(20261032)
        self.request,self.inputs,self.target=fixtures.EventPackageTests.synthetic_example()
        config=fixtures.EventPackageTests.small_model().c
        self.model=UnifiedCountCanvas(config,density_reference=4).eval()
        reset_mamba_boundaries(self.model)

    def test_joint_law_is_normalized_and_has_natural_nll(self):
        out=self.model(self.inputs);p=out['count'].exp()
        torch.testing.assert_close(p.sum(-1),torch.ones_like(p[...,0]),atol=1e-6,rtol=0)
        e=out['existence']
        torch.testing.assert_close(out['count'][...,0],F.logsigmoid(-e),atol=1e-6,rtol=0)
        torch.testing.assert_close(p[...,1:].sum(-1),e.sigmoid(),atol=1e-6,rtol=0)
        loss,parts=unified_count_loss(out,self.target,self.inputs)
        valid=self.inputs['canvas_valid'];truth=self.target['count']
        nll=-out['count'].gather(-1,truth[...,None]).squeeze(-1)
        expected=(nll*valid[...,None]).sum()/(4*valid.sum())
        torch.testing.assert_close(parts['count'],expected,atol=1e-6,rtol=0)
        torch.testing.assert_close(loss,expected+parts['offset']+parts['tail']+.2*parts['total']+.2*parts['interval'])

    def test_density_has_one_normalized_count_route(self):
        base=self.model(self.inputs)
        inputs=copy.deepcopy(self.inputs);inputs['controls'][0,0,:3]=torch.tensor([6/64,8/64,1.])
        tilted=self.model(inputs)
        torch.testing.assert_close(base['base_count'],tilted['base_count'],atol=0,rtol=0)
        expected=base['base_count']+F.softplus(self.model.density_alpha)*torch.tensor(7/4).log()*torch.arange(17)
        expected=expected-expected.logsumexp(-1,keepdim=True)
        torch.testing.assert_close(tilted['count'],expected,atol=1e-6,rtol=0)

    def test_original_zero_quantity_rows_are_unused_but_existing_heads_learn(self):
        loss,_=unified_count_loss(self.model(self.inputs),self.target,self.inputs);loss.backward()
        zero_rows=torch.arange(4)*17
        self.assertTrue(torch.equal(self.model.quantity.weight.grad[zero_rows],torch.zeros_like(self.model.quantity.weight.grad[zero_rows])))
        self.assertGreater(float(self.model.existence.weight.grad.abs().max()),0)
        self.assertGreater(float(self.model.offset.weight.grad.abs().max()),0)

    def test_parameter_dictionary_and_common_initialization_are_preserved(self):
        config=self.model.c;torch.manual_seed(123);a=MambaCanvas(config,density_reference=4)
        torch.manual_seed(123);b=UnifiedCountCanvas(config,density_reference=4)
        self.assertEqual(set(a.state_dict()),set(b.state_dict()))
        for key,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[key]),key)

    def test_padding_count_probabilities_do_not_change_the_loss(self):
        out=self.model(self.inputs);changed={**out,'count':out['count'].clone()}
        changed['count'][~self.inputs['canvas_valid']]=torch.randn_like(changed['count'][~self.inputs['canvas_valid']])*3
        a,_=unified_count_loss(out,self.target,self.inputs);b,_=unified_count_loss(changed,self.target,self.inputs)
        torch.testing.assert_close(a,b,atol=1e-6,rtol=0)


if __name__=='__main__':unittest.main()
