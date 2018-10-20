import decision_tree as tree
import numpy as np

def _trivial_split():
    X_train = np.array([
        [1],
        [1],
        [0],
        [0],
        [1],
        [1],
        [1]
    ])

    y_train = np.array([1, 1, 1, 1, 1, 1, 1])

    X_test = np.array([
        [0],
        [1],
        [1]
    ])

    y_test = np.array([1, 1, 1])

    return X_train, X_test, y_train, y_test


def _will_they_split():
    X_train = np.array([
        [0, 1, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 1, 0, 0, 1],
        [0, 1, 1, 1, 0, 0, 0, 1]
    ])

    y_train = np.array([0, 1, 0, 1, 1, 0, 0])

    X_test = np.array([
        [0, 1, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 1, 0, 1],
        [1, 1, 1, 0, 0, 0, 0, 1]
    ])

    y_test = np.array([0, 1, 0])

    return X_train, X_test, y_train, y_test


def test_same_class_tree_default_params():
    t = tree.DecisionTree()
    X_train, _, y_train, _ = _trivial_split()
    t.fit(X_train, y_train)
    raw_decision_tree = t.root

    assert raw_decision_tree.is_leaf
    assert raw_decision_tree.prediction == 1


def test_will_they_fit_no_error():
    t = tree.DecisionTree()
    X_train, _, y_train, _ = _will_they_split()
    t.fit(X_train, y_train)

    assert not t.root.is_leaf
