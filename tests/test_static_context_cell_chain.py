import unittest

import torch

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_typed_canvas import CanvasBlock
from beatfcn.model.static_context_cell_chain import StaticContextBlock


class StaticContextTests(unittest.TestCase):
    def test_masked_padding_preserves_outputs_inputs_and_all_parameter_gradients(self):
        torch.manual_seed(73);config=FullEventConfig(d=32,heads=4,kv_heads=2,ffn=64,dropout=0.)
        dynamic=CanvasBlock(config);fixed=StaticContextBlock(config);fixed.load_state_dict(dynamic.state_dict())
        for q,k in ((5,13),(11,29)):
            x=torch.randn(1,q,32,requires_grad=True);y=x.detach().clone().requires_grad_()
            values=torch.randn(1,k,32,requires_grad=True);other=values.detach().clone().requires_grad_()
            times=torch.arange(q)[None].float()*.04
            mem=dict(values=values,times=torch.arange(k)[None].float()*.5,valid=torch.ones(1,k,dtype=torch.bool),timed=torch.ones(1,k,dtype=torch.bool))
            mem['valid'][0,-2:]=False
            a=dynamic(x,times,mem);b=fixed(y,times,{**mem,'values':other})
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
            ga=torch.autograd.grad(a.square().mean(),(x,values,*dynamic.parameters()))
            gb=torch.autograd.grad(b.square().mean(),(y,other,*fixed.parameters()))
            for first,second in zip(ga,gb):torch.testing.assert_close(first,second,atol=1e-6,rtol=1e-4)

    def test_no_silent_truncation(self):
        block=StaticContextBlock(FullEventConfig(d=32,heads=4,kv_heads=2,ffn=64,dropout=0.))
        with self.assertRaises(ValueError):block(torch.zeros(1,201,32),torch.zeros(1,201),dict(values=torch.zeros(1,1,32)))


if __name__=='__main__':unittest.main()
