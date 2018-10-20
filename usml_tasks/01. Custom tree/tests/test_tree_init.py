import decision_tree as tree
import numpy as np
import pytest


def test_default():
    t = tree.DecisionTree()
    assert t.max_depth == np.inf
    assert t.min_samples_split == 2
    assert t.criterion == 'gini'


def test_wrong_type_of_max_depth():
    with pytest.raises(ValueError):
        tree.DecisionTree(max_depth=1.5)


def test_wrong_value_of_max_depth():
    with pytest.raises(ValueError):
        tree.DecisionTree(max_depth=0)


def test_wrong_criterion():
    with pytest.raises(ValueError):
        tree.DecisionTree(criterion='non-existed')
