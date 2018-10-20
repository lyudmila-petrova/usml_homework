from sklearn.model_selection import train_test_split

import decision_tree as tree
import numpy as np
from sklearn.datasets import load_digits


def test_predict_proba_on_digits_dataset():
    digits_dataset = load_digits()
    RANDOM_STATE = 17
    X_train, X_test, y_train, y_test = train_test_split(
        digits_dataset['data'], digits_dataset['target'], test_size=0.2, random_state=RANDOM_STATE)

    t = tree.DecisionTree(criterion='gini', max_depth=3)
    t.fit(X_train, y_train)

    proba = t.predict_proba(X_train[0:1])

    np.testing.assert_almost_equal(proba[0].sum(), 1.0, 3)
