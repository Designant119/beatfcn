import unittest

import torch

from beatfcn.model.full_event_model import rotate, _mps_periods


@unittest.skipUnless(torch.backends.mps.is_available(), 'Requires a real Apple GPU')
class MpsPositionTests(unittest.TestCase):
    def test_real_mps_rotation_matches_cpu_values_gradients_and_reuses_constants(self):
        torch.manual_seed(42)
        x=torch.randn(2,2,7,16,requires_grad=True)
        pos=torch.arange(7).repeat(2,1).float()*.04
        mask=torch.tensor([[True]*6+[False],[True]*7])
        expected=rotate(x,pos,lo=.02,hi=2400,timed=mask)
        expected.square().sum().backward()
        actual_x=x.detach().to('mps').requires_grad_(True)
        actual=rotate(actual_x,pos.to('mps'),lo=.02,hi=2400,timed=mask.to('mps'))
        actual.square().sum().backward()
        torch.mps.synchronize()
        torch.testing.assert_close(actual.cpu(),expected,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(actual_x.grad.cpu(),x.grad,atol=1e-5,rtol=1e-5)
        p=_mps_periods(.02,2400,8)
        self.assertEqual(p.data_ptr(),_mps_periods(.02,2400,8).data_ptr())


if __name__=='__main__':unittest.main()
