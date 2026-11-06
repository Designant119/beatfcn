import unittest

import numpy as np

from beatfcn.train.event_density_drag_filter import dense_event_drag_mask


class StrictDensityRuleTest(unittest.TestCase):
    def test_twelve_kept_thirteen_bans_only_drag(self):
        for n in (12,13):
            kinds=np.array([0]*(n-1)+[3]);removed,windows=dense_event_drag_mask(np.arange(n),kinds)
            self.assertEqual(int(removed.sum()),int(n==13))
            if n==13:self.assertEqual(np.flatnonzero(removed).tolist(),[12])

    def test_window_is_half_open(self):
        times=np.array(list(range(12))+[1_000_000]);kinds=np.array([0]*12+[3])
        self.assertFalse(dense_event_drag_mask(times,kinds)[0].any())
        times[-1]=999_999
        self.assertTrue(dense_event_drag_mask(times,kinds)[0][-1])

    def test_mixed_types_count_toward_trigger(self):
        kinds=np.array([0]*4+[1]*4+[2]*4+[3]);removed,_=dense_event_drag_mask(np.arange(13)*10_000,kinds)
        self.assertEqual(np.flatnonzero(removed).tolist(),[12])
        self.assertFalse(dense_event_drag_mask(np.arange(20),np.zeros(20,int))[0].any())

    def test_duplicates_and_arbitrary_input_order(self):
        kinds=np.array([3]*3+[0]*10);times=np.full(13,500_000)
        removed,windows=dense_event_drag_mask(times,kinds)
        self.assertEqual(removed.sum(),3);self.assertEqual(windows,[(500_000,1_500_000,13)])
        ids=np.arange(13)[::-1]
        np.testing.assert_array_equal(dense_event_drag_mask(times[ids],kinds[ids])[0],removed[ids])

    def test_union_matches_bruteforce_full_chart_before_slice(self):
        rng=np.random.default_rng(54)
        for _ in range(50):
            times=rng.integers(0,20,40)*100_000;kinds=rng.integers(0,4,40)
            expected=np.zeros(40,bool)
            for start in set(map(int,times)):
                inside=(times>=start)&(times<start+1_000_000)
                if inside.sum()>12:expected|=inside&(kinds==3)
            np.testing.assert_array_equal(dense_event_drag_mask(times,kinds)[0],expected)


if __name__=='__main__':unittest.main()
