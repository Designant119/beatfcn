from collections import Counter
from itertools import product
import os
import math
import unittest

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.train.exact_cell_support import ExactCellSupport
from beatfcn.train.typed_inventory import run_typed_members
from beatfcn.model.shared_total_inventory import SharedTotalInventory, count_nll, count_modes, count_log_marginals, total_log_probs
from beatfcn.train.shared_total_inventory import apply_update


class SharedTotalTests(unittest.TestCase):
    def test_joint_mass_marginals_conditional_mode_and_gradients(self):
        torch.manual_seed(338)
        scores = torch.randn(1, 3, 4, 4, dtype=torch.float64, requires_grad=True)
        totals = torch.randn(1, 3, 5, dtype=torch.float64, requires_grad=True)
        allowed = torch.tensor([[[True]*4, [True, False, True, False], [False]*4]])
        q = total_log_probs(totals, allowed)
        marginal = count_log_marginals(scores, totals, allowed).exp()
        torch.testing.assert_close(marginal.sum(-1), torch.ones(1, 3, 4, dtype=torch.float64))
        mode, failures = count_modes(scores, totals, allowed)
        for c in range(3):
            vectors = [v for v in product(range(4), repeat=4) if sum(v) <= 3 and
                       all(allowed[0, c, k] or n == 0 for k, n in enumerate(v))]
            masses = []
            for v in vectors:
                labels = torch.tensor([[v]], dtype=torch.long)
                masses.append(count_nll(scores[:, c:c+1], totals[:, c:c+1],
                                        allowed[:, c:c+1], labels).neg().exp()[0, 0])
            weights = torch.stack(masses)
            torch.testing.assert_close(weights.sum()+q[0, c, -1].exp(), torch.tensor(1., dtype=torch.float64))
            for k in range(4):
                for n in range(4):
                    ids = [i for i, v in enumerate(vectors) if v[k] == n]
                    expected = weights[ids].sum()
                    torch.testing.assert_close(expected, marginal[0, c, k, n])
            chosen_total = int(q[0, c].argmax())
            self.assertEqual(c in failures, chosen_total == 4)
            if chosen_total < 4:
                choices = [(sum(float(scores[0, c, k, n].detach()) for k, n in enumerate(v)), v)
                           for v in vectors if sum(v) == chosen_total]
                expected = min(choices, key=lambda v: (-v[0], v[1]))[1]
                self.assertEqual(tuple(mode[c]), expected)
        labels = torch.tensor([[[1, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]]])
        loss = count_nll(scores, totals, allowed, labels).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertTrue(torch.isfinite(totals.grad).all())
        self.assertEqual(float(scores.grad[0, 2].abs().max()), 0.)
        self.assertEqual(float(totals.grad[0, 2].abs().max()), 0.)
        with self.assertRaises(ValueError):
            count_nll(scores, totals, allowed, labels+1)

    def test_n0_and_forced_type_mark_gradients_cancel(self):
        for positive in (False, True):
            torch.manual_seed(51)
            scores = torch.randn(1, 1, 4, 5, dtype=torch.float64, requires_grad=True)
            totals = torch.randn(1, 1, 6, dtype=torch.float64, requires_grad=True)
            allowed = torch.tensor([[[True, False, False, False] if positive else [True]*4]])
            labels = torch.tensor([[[3, 0, 0, 0] if positive else [0]*4]])
            loss = count_nll(scores, totals, allowed, labels).sum()
            loss.backward()
            self.assertLess(float(scores.grad.abs().max()), 1e-12)
            self.assertGreater(float(totals.grad.abs().max()), 0.)

    def test_gradient_matches_enumeration(self):
        torch.manual_seed(52)
        scores = torch.randn(1, 1, 4, 3, dtype=torch.float64, requires_grad=True)
        totals = torch.randn(1, 1, 4, dtype=torch.float64, requires_grad=True)
        allowed = torch.ones(1, 1, 4, dtype=torch.bool)
        labels = torch.tensor([[[1, 0, 1, 0]]])
        actual = count_nll(scores, totals, allowed, labels).sum()
        vectors = [v for v in product(range(3), repeat=4) if sum(v) == 2]
        z = torch.stack([sum(scores[0, 0, k, n] for k, n in enumerate(v)) for v in vectors]).logsumexp(0)
        expected = -totals.log_softmax(-1)[0, 0, 2]+z-sum(scores[0, 0, k, n] for k, n in enumerate(labels[0, 0]))
        for a, b in zip(torch.autograd.grad(actual, (scores, totals), retain_graph=True),
                        torch.autograd.grad(expected, (scores, totals))):
            torch.testing.assert_close(a, b)

    def test_overflow_uneditable_and_ties(self):
        scores = torch.zeros(1, 2, 4, 3)
        totals = torch.tensor([[[0., 10., 0., -20.], [0., 0., 0., 20.]]])
        allowed = torch.ones(1, 2, 4, dtype=torch.bool)
        counts, failures = count_modes(scores, totals, allowed)
        self.assertEqual(counts[0].tolist(), [0, 0, 0, 1])
        self.assertEqual(failures, [1])
        counts, failures = count_modes(scores, totals, ~allowed)
        self.assertFalse(failures)
        self.assertEqual(int(counts.sum()), 0)
        self.assertTrue(torch.isfinite(count_log_marginals(scores, totals, ~allowed)[..., 0]).all())

    def test_cpu_mps_real_count_shape_and_marginal_gradient(self):
        if not torch.backends.mps.is_available():
            self.skipTest('MPS unavailable')
        torch.manual_seed(54)
        torch.set_num_threads(4)
        scores = torch.randn(1, 7, 4, 65)
        totals = torch.randn(1, 7, 66)
        allowed = torch.tensor([[[True]*4, [False]*4, [True,False,True,False],
                                 [False,True,False,True], [True]*4, [True]*4, [True]*4]])
        labels = torch.zeros(1, 7, 4, dtype=torch.long)
        labels[0, 0, 1] = 2
        labels[0, 2, 2] = 1
        result = []
        for device in ('cpu', 'mps'):
            s = scores.to(device).detach().requires_grad_()
            t = totals.to(device).detach().requires_grad_()
            a = allowed.to(device)
            loss = count_nll(s, t, a, labels.to(device)).sum()
            loss.backward()
            marginal = count_log_marginals(s.detach(), t.detach(), a).exp()
            self.assertTrue(torch.isfinite(s.grad).all())
            self.assertTrue(torch.isfinite(t.grad).all())
            torch.testing.assert_close(marginal.sum(-1), torch.ones_like(marginal[..., 0]), atol=1e-4, rtol=1e-4)
            result.append((loss.detach().cpu(), s.grad.cpu(), t.grad.cpu(), marginal.cpu()))
        for cpu, mps in zip(*result):
            self.assertLess(float((cpu-mps).abs().max()), 1e-4)

    def test_applied_counter_boundary(self):
        class Optimizer:
            def __init__(self, fail=False): self.fail = fail
            def step(self):
                if self.fail: raise RuntimeError('step did not return')
        report = dict(updates=0, optimizer_updates=0)
        with self.assertRaises(RuntimeError): apply_update(Optimizer(True), report)
        self.assertEqual(report['updates'], 0)
        self.assertEqual(report['accumulation_phase'], 'optimizer_step_started')
        apply_update(Optimizer(), report)
        self.assertEqual((report['updates'], report['optimizer_updates']), (1, 1))
        # A subsequent finite-state failure must leave the already-applied count intact.
        try: raise FloatingPointError('post-step finite-state failure')
        except FloatingPointError: pass
        self.assertEqual(report['optimizer_updates'], 1)


    def test_type_members_keep_duplicate_short_and_fractional_cross_boundary_hold(self):
        device=os.environ.get('REPLAY_TEST_DEVICE','cpu')
        torch.set_num_threads(4);torch.manual_seed(919)
        c=FullEventConfig(d=32,heads=4,kv_heads=2,ffn=64,encoder_layers=1,decoder_layers=0,
            taps=2,bands=1,band_slots=1,tap_dim=1024,ssm_state=8,ssm_headdim=16,mel_bins=128)
        model=SharedTotalInventory(c,density_reference=6.).to(device)
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


if __name__ == '__main__':
    unittest.main()
