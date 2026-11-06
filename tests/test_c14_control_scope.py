import unittest
import torch
from beatfcn.train.c14_control_scope import unspecified_controls


class ControlScopeTests(unittest.TestCase):
    def test_unknown_preserves_applicability_and_public_window(self):
        x={'controls':torch.tensor([[[.2,.5,1.,1.,.0,.006],[0,0,0,0,0,0]]]),'mel':torch.ones(1)}
        changed=unspecified_controls(x)
        torch.testing.assert_close(changed['controls'][...,:3],torch.zeros_like(x['controls'][...,:3]))
        self.assertTrue(torch.equal(changed['controls'][...,3:],x['controls'][...,3:]))
        self.assertIs(changed['mel'],x['mel']);self.assertEqual(float(x['controls'][0,0,2]),1.)


if __name__=='__main__':unittest.main()
