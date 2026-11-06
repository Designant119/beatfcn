import unittest

import numpy as np

from beatfcn.model.count_sampling import sample_tail


class CountSamplingTests(unittest.TestCase):
    def test_exact_legal_tail_distribution_and_copy_remainder(self):
        rng=np.random.default_rng(27)
        block=np.zeros(601);cell=np.zeros(50);ms=np.zeros(40)
        block[1]=.2;ms[:3]=[.1,.6,-.2]
        samples=np.array([sample_tail(rng,block,cell,ms,1_999_001,2_002_000) for _ in range(5000)])
        self.assertTrue(np.isin(samples,[2_000_000,2_001_000,2_002_000]).all())
        logits=ms[:3];expected=np.exp(logits-logits.max());expected/=expected.sum()
        observed=np.array([(samples==t).mean() for t in [2_000_000,2_001_000,2_002_000]])
        np.testing.assert_allclose(observed,expected,atol=.025,rtol=0)
        with self.assertRaisesRegex(ValueError,'No legal'):
            sample_tail(rng,block,cell,ms,2_002_000,2_002_000)


if __name__=='__main__':unittest.main()
