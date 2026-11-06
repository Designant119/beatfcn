import unittest

import numpy as np

from scripts.run_c07_count_ancestral import inverse_cdf, normalized


class CountAncestralTests(unittest.TestCase):
    def test_inverse_cdf_respects_zero_mass_and_exact_boundaries(self):
        p=np.array([[.25,0.,.75]]*4)
        np.testing.assert_array_equal(inverse_cdf(p,np.array([0.,.249,.25,.999999])),[0,0,2,2])


    def test_uniform_quadrature_recovers_categorical_expectation(self):
        p=np.broadcast_to(np.array([.1,.2,.7]),(1000,3))
        values=inverse_cdf(p,(np.arange(1000)+.5)/1000)
        np.testing.assert_array_equal(np.bincount(values,minlength=3),[100,200,700])
        self.assertEqual(values.mean(),1.6)


    def test_log_probability_and_logit_inputs_have_same_normalized_law(self):
        x=np.array([[1000.,1001.,998.],[-1001.,-1000.,-999.]])
        p=normalized(x)
        np.testing.assert_allclose(p.sum(-1),1.,atol=1e-14)
        np.testing.assert_allclose(normalized(np.log(p)),p,atol=1e-14)


    def test_ten_cell_probability_shift_preserves_each_type_histogram(self):
        rng=np.random.default_rng(2026102901);p=normalized(rng.normal(size=(200,4,17)))
        shifted=np.roll(p,10,axis=0)
        np.testing.assert_allclose(np.sort(p,axis=0),np.sort(shifted,axis=0),atol=0)
        np.testing.assert_allclose((p*np.arange(17)).sum(0),(shifted*np.arange(17)).sum(0),atol=1e-10)
