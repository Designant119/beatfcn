from collections import Counter
from itertools import product
import math
import os
import unittest

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.typed_inventory import TypedInventory, count_logz, count_nll, count_log_marginals, count_modes
from beatfcn.train.exact_cell_support import ExactCellSupport
from beatfcn.train.typed_inventory import carry_reference_row, run_typed_members


class TypedInventoryTests(unittest.TestCase):
    def test_exact_count_mass_mode_marginals_and_gradient(self):
        torch.manual_seed(338)
        scores = torch.randn(1, 3, 4, 4, dtype=torch.float64, requires_grad=True)
        allowed = torch.tensor([[[True]*4, [True,False,True,False], [False]*4]])
        overflow = torch.tensor([[-1., -.7, 0.]], dtype=torch.float64, requires_grad=True)
        z = count_logz(scores, allowed)
        marginals = count_log_marginals(scores, overflow, allowed).exp()
        torch.testing.assert_close(marginals.sum(-1), torch.ones(1,3,4,dtype=torch.float64))
        modes, failures = count_modes(scores, overflow, allowed)
        for c in range(3):
            vectors=[n for n in product(range(4),repeat=4) if sum(n)<=3 and
                     all(allowed[0,c,k] or n[k]==0 for k in range(4))]
            values=torch.stack([sum(scores[0,c,k,n[k]] for k in range(4)) for n in vectors])
            torch.testing.assert_close(z[0,c],values.logsumexp(0))
            for k in range(4):
                for n in range(4):
                    ids=[i for i,v in enumerate(vectors) if v[k]==n]
                    mass=values[ids].logsumexp(0).sub(z[0,c]).exp() if ids else torch.tensor(0.,dtype=torch.float64)
                    mass=mass*torch.sigmoid(-overflow[0,c]) if c<2 else mass
                    torch.testing.assert_close(mass,marginals[0,c,k,n])
            best=min(zip(values.detach().tolist(),vectors),key=lambda v:(-v[0],v[1]))
            self.assertEqual(tuple(modes[c]),best[1])
            self.assertEqual(c in failures,c<2 and float(overflow[0,c])>best[0]-float(z[0,c]))
        counts=torch.tensor([[[1,0,1,0],[1,0,0,0],[0,0,0,0]]])
        loss=count_nll(scores,overflow,allowed,counts).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertTrue(torch.isfinite(overflow.grad).all())
        self.assertEqual(float(overflow.grad[0,2]),0.)
        self.assertEqual(float(scores.grad[0,2].abs().max()),0.)
        with self.assertRaises(ValueError):count_nll(scores,overflow,allowed,counts+1)

    def test_no_quota_and_lexicographic_mode(self):
        s=torch.full((1,1,4,5),-torch.inf);s[...,0]=math.log(.7);s[...,1]=math.log(.3)
        a=torch.ones(1,1,4,dtype=torch.bool)
        mode,failed=count_modes(s,torch.tensor([[-8.]]),a)
        self.assertEqual(mode.tolist(),[[0,0,0,0]]);self.assertFalse(failed)
        s=torch.zeros(1,1,4,3)
        mode,failed=count_modes(s,torch.tensor([[-20.]]),a)
        self.assertEqual(mode.tolist(),[[0,0,0,0]])

    def test_cpu_mps_count_probability_and_gradient(self):
        if not torch.backends.mps.is_available():self.skipTest('MPS unavailable')
        torch.manual_seed(447);torch.set_num_threads(4)
        base=torch.randn(1,7,4,65)
        allowed=torch.tensor([[[True]*4,[False]*4,[True,False,True,False],[False,True,False,True],[True]*4,[True]*4,[True]*4]])
        counts=torch.zeros(1,7,4,dtype=torch.long);counts[0,0,1]=2;counts[0,2,2]=1
        results=[]
        for device in ('cpu','mps'):
            scores=base.to(device).detach().requires_grad_();o=torch.full((1,7),-8.,device=device,requires_grad=True)
            loss=count_nll(scores,o,allowed.to(device),counts.to(device)).sum();loss.backward()
            results.append((float(loss.detach().cpu()),scores.grad.cpu(),count_log_marginals(scores.detach(),o.detach(),allowed.to(device)).exp().cpu()))
        self.assertLess(abs(results[0][0]-results[1][0]),1e-4)
        self.assertLess(float((results[0][1]-results[1][1]).abs().max()),1e-4)
        self.assertLess(float((results[0][2]-results[1][2]).abs().max()),1e-4)

    def test_carry_multiset_alias_and_masked_reference(self):
        hold=RhythmEvent(1000,90000,'hold')
        request=EventRequest('phira',40000,80000,100000,known_carry=(hold,hold))
        row=dict(request=request,visible=[hold],view='draft',index=0,source_role='human_input',inputs={})
        for masked,extra in ((False,1),(True,2)):
            result=carry_reference_row(row,'a'*64,mask_reference=masked)
            self.assertIs(result['request'],request)
            self.assertEqual(len(result['request'].known_carry),2)
            self.assertEqual(len(result['carry_reference_additions']),extra)
            self.assertEqual(int(result['inputs']['reference'][:,10].sum()),2)
        self.assertEqual(len(row['visible']),1)

    def test_type_members_keep_duplicate_short_and_fractional_cross_boundary_hold(self):
        device=os.environ.get('REPLAY_TEST_DEVICE','cpu')
        torch.set_num_threads(4);torch.manual_seed(919)
        c=FullEventConfig(d=32,heads=4,kv_heads=2,ffn=64,encoder_layers=1,decoder_layers=0,
            taps=2,bands=1,band_slots=1,tap_dim=1024,ssm_state=8,ssm_headdim=16,mel_bins=128)
        model=TypedInventory(c,density_reference=6.).to(device)
        request=EventRequest('phira',0,4000,12000,copy_times=(1501,1502,7001))
        events=[RhythmEvent(1501,7001,'hold'),RhythmEvent(1501,7001,'hold'),
                RhythmEvent(1502,1502,'tap'),RhythmEvent(1502,1502,'flick'),RhythmEvent(1502,1502,'drag')]
        heads,allowed=ExactCellSupport().addresses(request);_,members=ExactCellSupport().encode(events,request)
        by=[[[] for _ in range(4)]];counts=np.zeros((1,4),np.int64)
        for m in members:by[0][m.kind].append(m);counts[0,m.kind]+=1
        target=dict(counts=counts,members=by,heads=heads,allowed=allowed)
        out=dict(x=torch.randn(1,1,32,device=device),typed_scores=torch.zeros(1,1,4,65,device=device),
            song_values=torch.randn(1,2,32,device=device),song_times=torch.tensor([[0.,.5]],device=device),song_valid=torch.ones(1,2,dtype=torch.bool,device=device))
        inputs=dict(mode=torch.tensor([1],device=device),mel=torch.randn(1,4,128,device=device),mel_times=torch.tensor([[-.01,0.,.01,.02]],device=device),mel_valid=torch.ones(1,4,dtype=torch.bool,device=device))
        actual,parts,steps,holds=run_typed_members(model,out,inputs,request,target,counts,teacher=True,trace=True)
        self.assertEqual(Counter(actual),Counter(events));self.assertEqual(holds,2)
        self.assertEqual(len(steps),5)
        (parts['head']+parts['tail']).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertGreater(float(model.member_initial.weight.grad.abs().sum().cpu()),0.)


if __name__=='__main__':unittest.main()
