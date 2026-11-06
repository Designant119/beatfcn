from collections import Counter
from itertools import permutations
import unittest

import numpy as np
import torch

from beatfcn.model.full_event_codec import EventRequest
from beatfcn.model.joint_native_set import relative_tail_log_partition, relative_tail_nll
from beatfcn.train.joint_native_set import SUPPORT, decode_set, injective_assignment


class JointNativeSetTests(unittest.TestCase):
    def test_MPS_sparse_tail_insertion_preserves_value_and_gradient(self):
        # Reproduce the exact failed operation's replacement with an independent
        # CPU index_copy reference. No enable-fallback environment is used.
        if not torch.backends.mps.is_available():
            self.skipTest('MPS unavailable')
        a = torch.tensor([.8, 1.2, 3.1], requires_grad=True)
        indices = torch.tensor([0, 3, 6])
        ref = torch.zeros(7).index_copy(0, indices, a)
        (ref.square().sum()).backward()
        b = a.detach().to('mps').requires_grad_()
        incidence = (torch.arange(7, device='mps')[:, None] == indices.to('mps')[None]).float()
        actual = incidence @ b
        actual.square().sum().backward()
        torch.mps.synchronize()
        torch.testing.assert_close(actual.cpu(), ref)
        torch.testing.assert_close(b.grad.cpu(), a.grad)

    def test_matching_agrees_with_exhaustive_multiset_assignment(self):
        rng = np.random.default_rng(19)
        for n in (0, 1, 2, 3):
            a = rng.normal(size=(5, n))
            if n > 1:
                a[:, 1] = a[:, 0]  # Duplicate target is still a separate column.
            rows, cols = injective_assignment(a)
            best = min((sum(a[p[j], j] for j in range(n)) for p in permutations(range(5), n)), default=0.)
            self.assertAlmostEqual(float(a[rows, cols].sum()), best)
            self.assertEqual(len(set(rows)), n)
            self.assertEqual(set(cols), set(range(n)))

    def test_matching_target_permutation_preserves_cost(self):
        a = np.random.default_rng(28).normal(size=(12, 7))
        r, c = injective_assignment(a)
        b = a[:, [6, 1, 4, 2, 3, 5, 0]]
        rb, cb = injective_assignment(b)
        self.assertAlmostEqual(float(a[r, c].sum()), float(b[rb, cb].sum()))

    def test_relative_legal_normalizer_matches_enumeration_and_gradients(self):
        torch.manual_seed(41)
        b = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
        c = torch.randn(6, 50, dtype=torch.float64, requires_grad=True)
        m = torch.randn(6, 40, dtype=torch.float64, requires_grad=True)
        hi = torch.tensor([1, 39, 41, 2000, 2013, 4001])
        z = relative_tail_log_partition(b, c, m, hi)
        exact = []
        for i, end in enumerate(hi):
            t = torch.arange(1, int(end) + 1)
            exact.append((b[i, t // 2000] + c[i, (t % 2000) // 40] + m[i, t % 40]).logsumexp(0))
        exact = torch.stack(exact)
        torch.testing.assert_close(z, exact)
        g = torch.autograd.grad(z.sum(), (b, c, m), retain_graph=True)
        ge = torch.autograd.grad(exact.sum(), (b, c, m))
        for actual, wanted in zip(g, ge):
            torch.testing.assert_close(actual, wanted)
        loss = relative_tail_nll(b, c, m, hi, hi)
        self.assertTrue(bool(torch.isfinite(loss).all()))

    def test_decode_keeps_fractional_COPY_duplicates_and_cross_core_hold(self):
        request = EventRequest('phira', 0, 80000, 200000, copy_times=(1501,))
        heads, valid = SUPPORT.addresses(request)
        address = int(np.where((heads.reshape(-1) == 1501) & valid.reshape(-1))[0][0])

        class Stub:
            def eval(self):
                return self

            def tail(self, query, times, inputs):
                count = len(query)
                b, c, m = (torch.full((count, n), -100.) for n in (601, 50, 40))
                b[:, 0], c[:, 2], m[:, 9] = 100., 100., 100.  # d=89ms; end=90ms.
                return b, c, m

        output = dict(kind=torch.tensor([[[0., 0., 100., 0., 0.], [0., 100., 0., 0., 0.],
                                          [0., 100., 0., 0., 0.], [100., 0., 0., 0., 0.]]]),
            head=torch.full((1, 4, heads.size), -100.), query=torch.zeros(1, 4, 8))
        output['head'][:, :, address] = 100.
        result = decode_set(Stub(), {}, request, output=output)
        self.assertTrue(result['complete'], result['failure'])
        values = Counter((e['onset_us'], e['end_us'], e['type']) for e in result['events'])
        self.assertEqual(values, Counter({(1501, 90000, 'hold'): 1, (1501, 1501, 'tap'): 2}))

    def test_decode_reports_capacity_instead_of_truncation(self):
        request = EventRequest('phira', 0, 40000, 200000, max_events=1)
        heads, _ = SUPPORT.addresses(request)
        class Stub:
            def eval(self):
                return self
        output = dict(kind=torch.tensor([[[0., 10., 0., 0., 0.], [0., 10., 0., 0., 0.]]]),
            head=torch.zeros(1, 2, heads.size), query=torch.zeros(1, 2, 8))
        result = decode_set(Stub(), {}, request, output=output)
        self.assertFalse(result['complete'])
        self.assertIn('resource overflow', result['failure'])


if __name__ == '__main__':
    unittest.main()
