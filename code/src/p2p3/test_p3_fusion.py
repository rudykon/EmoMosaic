"""Independent numerical contract checks for the migrated inference path."""
import unittest
import numpy as np
from p3_fusion import shapley


class CooperativeGameTests(unittest.TestCase):
    def test_additive_and_pair_interaction(self):
        # Text/audio interaction must be shared equally; dummy vision is zero.
        v=np.asarray([[2+3*bool(s&1)-2*bool(s&2)+6*bool(s&1)*bool(s&2) for s in range(8)]])
        np.testing.assert_allclose(shapley(v),[[6,1,0]],atol=1e-12)
        np.testing.assert_allclose(shapley(v).sum(1),v[:,7]-v[:,0],atol=1e-12)

    def test_all_class_probabilities_preserve_zero_sum_attribution(self):
        rng=np.random.default_rng(17);v=rng.random((9,8,3));v/=v.sum(-1,keepdims=True)
        phi=shapley(v)
        np.testing.assert_allclose(phi.sum(1),v[:,7]-v[:,0],atol=1e-12)
        np.testing.assert_allclose(phi.sum(-1),0,atol=1e-12)


if __name__=='__main__':unittest.main()
