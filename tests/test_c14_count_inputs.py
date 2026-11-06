import unittest
import numpy as np
import torch
from scripts.run_c14_count_inputs import changed_input,count_summary
from beatfcn.train.c14_probe_support import prediction_identity,probability_difference


class CountInputTests(unittest.TestCase):
    def test_only_the_named_stream_is_replaced(self):
        x={k:torch.ones(2,3) for k in ('local','mel','song','controls','reference','local_times')}
        for arm,key in [('mel_zero','mel'),('song_zero','song'),('controls_unspecified','controls')]:
            y=changed_input(x,arm);self.assertEqual(int(y[key].count_nonzero()),0)
            for other in set(x)-{key}:self.assertIs(y[other],x[other])
        self.assertEqual(int(x['mel'].count_nonzero()),6)

    def test_summary_separates_false_location_mass_and_overflow(self):
        p=np.zeros((2,66));p[0,0]=.6;p[0,1]=.4;p[1,0]=.2;p[1,2]=.7;p[1,65]=.1
        r=count_summary(p,[1,0]);self.assertEqual(r['mode_events'],2)
        self.assertAlmostEqual(r['positive_expected'],.4);self.assertAlmostEqual(r['zero_expected'],1.4)
        self.assertAlmostEqual(r['overflow_mass'],.1)

    def test_probability_length_is_not_silently_truncated(self):
        with self.assertRaises(ValueError):probability_difference([1.],[1.,0.])
        a=dict(complete=True,count_decision=[0],events=[],steps=[])
        self.assertEqual(prediction_identity(a,a)['members'],0)
        b={**a,'steps':[{}]}
        with self.assertRaisesRegex(ValueError,'length'):prediction_identity(a,b)


if __name__=='__main__':unittest.main()
