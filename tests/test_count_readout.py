from itertools import product
import unittest

import numpy as np

from beatfcn.model.count_readout import cardinality_map, expected_tail_rank


class CountReadoutTests(unittest.TestCase):
    def test_cardinality_and_conditional_path_against_exhaustive(self):
        rng=np.random.default_rng(27)
        for n in (1,2,4):
            p=rng.random((n,4));p/=p.sum(-1,keepdims=True)
            counts,info=cardinality_map(p,event_limit=20)
            configurations=list(product(range(4),repeat=n));mass=np.zeros(3*n+1)
            for c in configurations:mass[sum(c)]+=np.prod([p[i,x] for i,x in enumerate(c)])
            self.assertEqual(info['total_mode'],int(mass.argmax()))
            best=max(np.prod([p[i,x] for i,x in enumerate(c)]) for c in configurations if sum(c)==sum(counts))
            self.assertAlmostEqual(np.exp(info['conditional_log_probability']),best,places=12)
            self.assertAlmostEqual(info['mode_probability'],float(mass.max()),places=12)

    def test_sparse_marginals_and_explicit_overflow(self):
        p=np.tile([.94,.06],(800,1));counts,info=cardinality_map(p,event_limit=4096)
        self.assertEqual(int(counts.sum()),48)
        self.assertEqual(int(p.argmax(-1).sum()),0)
        self.assertAlmostEqual(info['expected'],48)
        with self.assertRaisesRegex(ValueError,'resource'):
            cardinality_map(np.array([[0.,0.,1.]]),event_limit=1)
        self.assertEqual(cardinality_map(np.array([[1.,0.],[1.,0.]]),event_limit=10)[0].tolist(),[0,0])

    def test_expected_tail_prefix_and_no_target_count(self):
        p=np.array([[.4,.3,.3],[.8,.1,.1],[1.,0.,0.]])
        c,d=expected_tail_rank(p,event_limit=20)
        self.assertEqual(int(c.sum()),round(d['expected']))
        self.assertTrue(((c>=0)&(c<=2)).all())


if __name__=='__main__':unittest.main()
