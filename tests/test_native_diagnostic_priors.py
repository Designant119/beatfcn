import unittest
from beatfcn.train.native_diagnostic_priors import duration_bucket,fit_priors,type_probability,duration_probability,legal_duration_counts


class PriorTests(unittest.TestCase):
    def setUp(self):
        self.hold=dict(onset_us=1001,end_us=3501,type='hold')
        self.prior=fit_priors([[self.hold,self.hold]],[self.hold,self.hold],[6,2,1,1])

    def test_native_and_fractional_support_normalizes_without_rounding(self):
        head=1001;end=8001;copies=[1002,1501,3501,8001,3501]
        legal=sorted(set(range(2000,8001,1000))|set(copies))
        probabilities=[duration_probability(self.prior,head,t,end,copies) for t in legal]
        self.assertAlmostEqual(sum(probabilities),1.)
        self.assertTrue(all(p>0 for p in probabilities))
        self.assertEqual(duration_probability(self.prior,head,1502,end,copies),0.)
        self.assertEqual(sum(legal_duration_counts(head,end,copies)),len(legal))

    def test_canonical_minimum_can_be_exact_fractional_previous_tail(self):
        legal=[3501,4000,5000,6000]
        values=[duration_probability(self.prior,1001,t,6000,[3501],3501) for t in legal]
        self.assertAlmostEqual(sum(values),1.)
        self.assertEqual(duration_probability(self.prior,1001,3000,6000,[3501],3501),0.)

    def test_single_us_and_song_endpoint_in_last_bucket(self):
        self.assertEqual(duration_bucket(1),0);self.assertEqual(duration_bucket(1000),0)
        self.assertEqual(duration_bucket(1001),1)
        self.assertGreater(duration_probability(self.prior,0,1200000000,1200000000),0)
        self.assertAlmostEqual(duration_probability(self.prior,1000,1001,1001,[1001]),1.)

    def test_history_backoff_and_legal_masks_keep_negative_type_support(self):
        p=type_probability(self.prior,63,3,[True]*4)
        self.assertEqual(p,self.prior['global_prior'])
        p=type_probability(self.prior,0,-1,[True,True,False,True])
        self.assertEqual(p[2],0.);self.assertAlmostEqual(sum(p),1.)
        self.assertGreater(p[1],self.prior['global_prior'][1])

    def test_duplicate_events_are_independent_observations(self):
        self.assertEqual(self.prior['hold_observations'],2)
        self.assertEqual(sum(sum(v) for v in self.prior['history_counts'].values()),2)


if __name__=='__main__':unittest.main()
