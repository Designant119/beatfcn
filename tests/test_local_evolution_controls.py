import copy
import unittest

import torch

from beatfcn.model.local_evolution_controls import (
    AudioInteractionGroupedProcess, BoundaryResetMamba2, reset_mamba_boundaries,
)
from beatfcn.model.grouped_chart_process import GroupedChartProcess
from beatfcn.model.mamba2_reference import Mamba2Reference
from tests.test_local_event_package import EventPackageTests


class EvolutionControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(20261029)

    def test_single_chunk_outputs_and_gradients_are_identical(self):
        full=Mamba2Reference(16,d_state=8,headdim=8,chunk_size=16)
        reset=BoundaryResetMamba2(16,d_state=8,headdim=8,chunk_size=16)
        reset.load_state_dict(full.state_dict())
        a=torch.randn(1,13,16,requires_grad=True);b=a.detach().clone().requires_grad_()
        y,z=full(a),reset(b)
        torch.testing.assert_close(y,z,atol=1e-6,rtol=0)
        y.square().sum().backward();z.square().sum().backward()
        torch.testing.assert_close(a.grad,b.grad,atol=1e-5,rtol=0)
        for p,q in zip(full.parameters(),reset.parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=1e-5,rtol=0)

    def test_only_cross_boundary_ssd_dependence_is_removed(self):
        full=Mamba2Reference(16,d_state=8,headdim=8,chunk_size=16).eval()
        reset=BoundaryResetMamba2(16,d_state=8,headdim=8,chunk_size=16).eval()
        reset.load_state_dict(full.state_dict())
        a=torch.randn(1,40,16);b=a.clone();b[:,:12]+=2
        with torch.no_grad():
            first,second=full(a),reset(a)
            torch.testing.assert_close(first[:,:16],second[:,:16],atol=1e-6,rtol=0)
            self.assertGreater(float((first[:,16:]-second[:,16:]).abs().max()),1e-6)
            torch.testing.assert_close(reset(a)[:,20:32],reset(b)[:,20:32],atol=1e-6,rtol=0)
            self.assertGreater(float((full(a)[:,20:32]-full(b)[:,20:32]).abs().max()),1e-6)

    def test_ar_interaction_changes_within_cell_odds_without_new_parameters(self):
        request,inputs,_=EventPackageTests.synthetic_example()
        c=EventPackageTests.small_model().c
        original=GroupedChartProcess(c,density_reference=4,history_dropout=0).eval()
        fixed=AudioInteractionGroupedProcess(c,density_reference=4,history_dropout=0).eval()
        fixed.load_state_dict(original.state_dict())
        self.assertEqual(list(original.state_dict()),list(fixed.state_dict()))
        for model in (original,fixed):
            context=model.context(inputs,request)
            frame=next(int(f) for f in context['frame'].unique() if int((context['frame']==f).sum())>1)
            choices=(context['frame']==frame).nonzero().flatten()[[0,-1]]
            query=model.chart_query(torch.cat((model.chart_start,context['pooled']),-1)).detach()
            odds=((context['keys'][choices[0]]-context['keys'][choices[1]])*query).sum()
            grad=torch.autograd.grad(odds,context['x'])[0][frame].abs().max()
            if model is original:self.assertLess(float(grad),1e-7)
            else:self.assertGreater(float(grad),1e-6)

    def test_core_replacement_preserves_exact_parameter_values_and_rng(self):
        from beatfcn.model.local_mamba_canvas import MambaCanvas
        from beatfcn.model.full_event_model import FullEventConfig
        model=MambaCanvas(FullEventConfig(d=32,heads=4,kv_heads=2,ffn=64,
            encoder_layers=1,decoder_layers=0,ssm_state=8,ssm_headdim=8),density_reference=4)
        before=copy.deepcopy(model.state_dict());rng=torch.get_rng_state()
        self.assertEqual(reset_mamba_boundaries(model),7)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for name,value in model.state_dict().items():self.assertTrue(torch.equal(value,before[name]),name)


if __name__=='__main__':unittest.main()
