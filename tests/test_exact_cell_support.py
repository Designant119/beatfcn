from collections import Counter
from dataclasses import replace
from itertools import permutations
import unittest

import numpy as np

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest,encode_events
from beatfcn.train.consistent_assignment import balanced_assignment_cost
from beatfcn.train.exact_cell_support import ExactCellSupport,ExactEndpointTree
from beatfcn.train.joint_native_set import injective_assignment


class ExactCellSupportTests(unittest.TestCase):
    def test_fractional_tail_short_hold_duplicates_and_cross_core(self):
        r=EventRequest('phira',0,80000,200000,copy_times=(1501,1502,90001,90000))
        e=[RhythmEvent(1501,1502,'hold'),RhythmEvent(1501,90001,'hold'),
           RhythmEvent(1501,90001,'hold'),RhythmEvent(1501,90000,'hold'),RhythmEvent(2000,2000,'tap')]
        codec=ExactCellSupport();counts,m=codec.encode(e,r)
        self.assertEqual(Counter(codec.decode(counts,m,r)),Counter(e))
        self.assertEqual(len(m),5)
        tree=ExactEndpointTree(r,1501)
        for end in (1502,90000,90001):
            b,c,j=tree.path(end);self.assertEqual(tree.leaves(b,c)[j],end)
        # The old token grammar explicitly rejects this extended endpoint domain.
        with self.assertRaises(ValueError):encode_events(e,r)

    def test_endpoint_tree_is_unique_complete_and_time_translation_equivariant(self):
        r=EventRequest('phira',1501,41002,120012,copy_times=(1501,1502,40001,90000,120011))
        tree=ExactEndpointTree(r,1501)
        leaves=[u for b in tree.blocks() for c in tree.cells(b) for u in tree.leaves(b,c)]
        expected=list(range(2000,120001,1000))+[1502,40001,120011]
        self.assertEqual(sorted(leaves),sorted(expected));self.assertEqual(len(leaves),len(set(leaves)))
        shifted=replace(r,start_us=r.start_us+300000,end_us=r.end_us+300000,duration_us=r.duration_us+300000,
                        copy_times=tuple(u+300000 for u in r.copy_times))
        t=ExactEndpointTree(shifted,301501)
        for u in leaves:self.assertEqual(tree.path(u),t.path(u+300000))

    def test_native_domain_exact_and_unauthorized_fraction_rejected(self):
        r=EventRequest('phira',0,80000,200000,copy_times=(1501,))
        e=[RhythmEvent(1501,90000,'hold'),RhythmEvent(2000,2000,'tap')]
        codec=ExactCellSupport();a,b=codec.encode(e,r);self.assertEqual(Counter(codec.decode(a,b,r)),Counter(e))
        self.assertTrue(encode_events(e,r))
        with self.assertRaises(ValueError):codec.encode([RhythmEvent(1501,1502,'hold')],r)
        with self.assertRaises(ValueError):ExactEndpointTree(r,1501).path(1502)

    def test_balanced_matching_cost_minimizes_actual_loss_with_absent(self):
        positive=-np.log([[.6],[.5]]);absent=-np.log([.4,.001]);kinds=[1]
        oldr,_=injective_assignment(positive);newr,_=injective_assignment(balanced_assignment_cost(positive,absent,kinds))
        self.assertEqual(oldr.tolist(),[0]);self.assertEqual(newr.tolist(),[1])

    def test_weighted_assignment_exhaustive_classes_duplicates_empty_and_full(self):
        rng=np.random.default_rng(81)
        for n in range(5):
            q=4;p=rng.uniform(.1,5,(q,n));a=rng.uniform(.1,5,q);k=np.array([1,1,3,4][:n])
            if n>1:p[:,1]=p[:,0]
            cost=balanced_assignment_cost(p,a,k);rows,cols=injective_assignment(cost)
            def loss(order):
                if not n:return float(a.mean())
                types=set(k);pos=sum(np.mean([p[order[j],j] for j in range(n) if k[j]==t]) for t in types)/len(types)
                unused=[i for i in range(q) if i not in order];neg=float(a[unused].mean()) if unused else 0.
                return .5*pos+.5*neg
            order=[None]*n
            for i,j in zip(rows,cols):order[j]=i
            actual=loss(order);best=min(loss(o) for o in permutations(range(q),n))
            self.assertAlmostEqual(actual,best)


if __name__=='__main__':unittest.main()
