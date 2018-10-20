import decision_tree as tree
import numpy as np


def test_entropy_same_class():
    y = np.array([1, 1, 1, 1])
    np.testing.assert_almost_equal(tree.entropy(y), 0.0, 3)


def test_entropy():
    y = np.array([1, 0, 1, 1, 0, 0, 0])
    np.testing.assert_almost_equal(tree.entropy(y), 0.985, 3)


def test_entropy_50_percent():
    y = np.array([1, 0, 1, 1, 0, 0])
    np.testing.assert_almost_equal(tree.entropy(y), 1.0, 3)


def test_gini_50_percent():
    y = np.array([1, 0, 1, 1, 0, 0])
    np.testing.assert_almost_equal(tree.gini(y), 0.5, 3)


def test_gini_same_class():
    y = np.array([1, 1, 1, 1])
    np.testing.assert_almost_equal(tree.entropy(y), 0.0, 3)


def test_variance_same_values():
    y = np.array([1, 1, 1, 1])
    np.testing.assert_almost_equal(tree.variance(y), 0.0, 3)


def test_variance_uniform():
    y = np.array([1, 2, 3, 4])
    np.testing.assert_almost_equal(tree.variance(y), 1.25, 3)


def test_mad_median_same_values():
    y = np.array([12, 12, 12])
    np.testing.assert_almost_equal(tree.mad_median(y), 0.0, 3)


def test_mad_median_50_persent():
    y = np.array([12, 0, 12, 0, 12, 0])
    np.testing.assert_almost_equal(tree.mad_median(y), 6.0, 3)

