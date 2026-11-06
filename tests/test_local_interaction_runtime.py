import unittest

import numpy as np
import torch

from beatfcn.train.condition_interaction import condition_interaction_loss
from beatfcn.train.local_condition_pairs import common_fork, paired_exposure
from beatfcn.train.local_interaction_runtime import backward_interaction, padded_common_prefix


class LocalInteractionTests(unittest.TestCase):
    def test_streamed_gradients_match_joint_loss_and_preserve_existing_ce_gradient(self):
        inputs=torch.tensor([[1.,2.,3.],[-2.,1.,4.]],dtype=torch.float64)
        a=torch.nn.Linear(3,2,dtype=torch.float64);b=torch.nn.Linear(3,2,dtype=torch.float64);b.load_state_dict(a.state_dict())
        ce=lambda m: m(inputs).square().mean()
        (ce(a)+condition_interaction_loss(a(inputs))[0]).backward()
        ce(b).backward()
        loss,gap,errors=backward_interaction(lambda i:b(inputs[i]),verify=True)
        self.assertEqual(errors,[0.,0.])
        for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x.grad,y.grad,atol=1e-14,rtol=1e-13)

    def test_prefix_removes_future_instead_of_only_changing_mask(self):
        seq={k:torch.full((1,7),9,dtype=torch.long) for k in ('tokens','fields','times','targets')}
        seq['valid']=torch.ones(1,7,dtype=torch.bool);seq['allowed']=torch.ones(1,7,12,dtype=torch.bool)
        result=padded_common_prefix(seq,2,length=8)
        for key in ('tokens','fields','times','targets'):
            self.assertEqual(result[key][0,:3].tolist(),[9]*3)
            self.assertEqual(result[key][0,3:].tolist(),[0]*5)
        self.assertEqual(result['valid'].sum().item(),3)
        self.assertEqual(result['allowed'][0,3:,1:].sum().item(),0)
        with self.assertRaises(ValueError):padded_common_prefix(seq,8,length=8)

    def test_shared_prefix_must_have_identical_visible_signals_and_legal_targets(self):
        a={k:np.array([0,1,2]) for k in ('tokens','fields','times','targets')}
        a['valid']=np.ones(3,bool);a['allowed']=np.ones((3,4),bool)
        b={k:v.copy() for k,v in a.items()};b['targets'][1]=3
        self.assertEqual(common_fork(a,b),(1,[1,3]))
        b['times'][0]=2
        with self.assertRaises(ValueError):common_fork(a,b)
        b['times'][0]=0;b['allowed'][:,3]=False
        with self.assertRaises(ValueError):common_fork(a,b)

    def test_auxiliary_schedule_preserves_primary_mode_and_field_song_coverage(self):
        catalog=[dict(mode=m,field=f,component=s) for m in ('phira','mania_4k')
                 for f in ('density_notes_per_second','hold_head_ratio') for s in ('a','b','c')]
        steps=paired_exposure(catalog,40,seed=4)
        self.assertEqual(steps,paired_exposure(catalog,40,seed=4))
        rows=[catalog[s['pair']] for s in steps]
        self.assertEqual(sum(r['mode']=='phira' for r in rows),30)
        self.assertEqual({r['component'] for r in rows},{'a','b','c'})
        self.assertEqual({r['field'] for r in rows},{'density_notes_per_second','hold_head_ratio'})
