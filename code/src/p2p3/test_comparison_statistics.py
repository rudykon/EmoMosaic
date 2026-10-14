"""Check clustered statistics against direct sklearn resampling calculations."""
import itertools
import unittest

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from analyze_boosted_comparison import (
    cm_metrics, grouped_confusion, holm, paired_comparison,
)


class ComparisonStatisticsTest(unittest.TestCase):
    def test_video_bootstrap_retains_multiple_clips_and_conditions(self):
        truth = np.array([0, 2, 1, 0, 1, 2])
        groups = np.array([0, 0, 1, 2, 2, 2])
        a = np.array([[0, 1, 1, 2, 1, 2], [1, 2, 1, 0, 2, 2]])
        b = np.array([[0, 2, 0, 0, 2, 2], [0, 2, 1, 1, 1, 0]])
        sampled_videos = np.array([[0, 0, 2], [2, 1, 2], [1, 0, 1], [2, 2, 2]])
        weights = np.array([np.bincount(v, minlength=3) for v in sampled_videos])
        ca = grouped_confusion(truth, a, groups, 3)
        cb = grouped_confusion(truth, b, groups, 3)
        result = paired_comparison(ca, cb, weights)
        expected = []
        for draw in sampled_videos:
            idx = np.concatenate([np.flatnonzero(groups == v) for v in draw])
            yy = np.tile(truth[idx], len(a))
            expected.append(f1_score(yy, a[:, idx].ravel(), labels=[0, 1, 2], average='macro', zero_division=0)
                            - f1_score(yy, b[:, idx].ravel(), labels=[0, 1, 2], average='macro', zero_division=0))
        np.testing.assert_allclose(result['macro_f1']['delta_ci95'], np.quantile(expected, [.025, .975]))
        np.testing.assert_allclose(cm_metrics(ca.sum(0))['accuracy'], accuracy_score(np.tile(truth, 2), a.ravel()))

    def test_cluster_permutation_matches_direct_swapping(self):
        truth = np.array([0, 2, 1, 0, 1, 2])
        groups = np.array([0, 0, 1, 2, 2, 2])
        a = np.array([0, 1, 1, 2, 1, 2])
        b = np.array([0, 2, 0, 0, 2, 2])
        signs = np.array(list(itertools.product([-1, 1], repeat=3)))
        ca = grouped_confusion(truth, a, groups, 3)
        cb = grouped_confusion(truth, b, groups, 3)
        observed = f1_score(truth, a, average='macro')-f1_score(truth, b, average='macro')
        direct = []
        for sign in signs:
            swap = sign[groups] == -1
            pa = np.where(swap, b, a)
            pb = np.where(swap, a, b)
            direct.append(f1_score(truth, pa, average='macro')-f1_score(truth, pb, average='macro'))
        expected = (sum(abs(x) >= abs(observed)-1e-14 for x in direct)+1)/(len(signs)+1)
        result = paired_comparison(ca, cb, np.ones((2, 3)), signs)
        self.assertEqual(result['macro_f1']['cluster_permutation_p_two_sided'], expected)

    def test_holm_known_example(self):
        np.testing.assert_allclose(holm([.01, .04, .03, .2]), [.04, .09, .09, .2])


if __name__ == '__main__':
    unittest.main()
