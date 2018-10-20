import numpy as np
from sklearn.base import BaseEstimator

__version__ = '0.1'


def _proba(y):
    """ Array of relative frequencies of the unique values. """
    N = len(y)
    _, counts = np.unique(y, return_counts=True)
    return counts / N


def entropy(y):
    """ A way to measure impurity """
    p = _proba(y)
    return (-p * np.log2(p)).sum()


def gini(y):
    """ A criterion to minimize the probability of misclassification """
    p = _proba(y)
    return 1.0 - sum(list(map(lambda x: x * x, p)))


def variance(y):
    return np.var(y)


def mad_median(y):
    median = np.median(y)
    X = len(y)
    return sum(list(map(lambda x: abs(x - median), y))) / X


class DecisionTreeNode:
    def __init__(self, depth=-1):
        self.depth = depth
        self.left = None
        self.right = None
        self.threshold = None
        self.split_feature_index = None
        self.is_leaf = False
        self.prediction = None

        self.stats = []
        self.score = np.inf

    def make_leaf(self, prediction):
        self.is_leaf = True
        self.prediction = prediction

    def __str__(self):
        result = "\n" + " " * 3 * self.depth
        if self.is_leaf:
            result += f"Prediction: {self.prediction}"
        else:
            result += f"(Feature: {self.split_feature_index}; Threshold: {self.threshold};)"

        result += " Score: " + str(self.score)
        result += " Stats: " + str(self.stats)

        if not self.is_leaf:
            result += str(self.left)
            result += str(self.right)
        return result


class DecisionTree(BaseEstimator):
    """
    Decision Tree
    Parameters
    ----------
    max_depth : int, optional
        Maximum depth of trees
    criterion : string, optional
        Split criterion. Valid values are 'entropy', 'gini', 'variance', 'mad_median'.
    min_samples_split : int, optional
    """

    CRITERION_MAP = {
            'entropy': entropy,
            'gini': gini,
            'variance': variance,
            'mad_median': mad_median
        }

    def __init__(self, max_depth=np.inf, min_samples_split=2,
                 criterion='gini', debug=False):
        if max_depth != np.inf and not isinstance(max_depth, int):
            raise ValueError(f"max_depth must be integer, but (type{max_depth}) given")

        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1")

        if not isinstance(min_samples_split, int):
            raise ValueError(f"min_samples_split must be integer, but (type{min_samples_split}) given")

        if criterion not in DecisionTree.CRITERION_MAP:
            raise ValueError(f"Unknown criterion given.", "Expected one of:", *DecisionTree.CRITERION_MAP.keys(),
                             f"But {criterion} given.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.debug = debug

        self.root = None

        self.is_classification = criterion in ['entropy', 'gini']
        self.classes = []

    def fit(self, X, y):

        if len(y.shape) != 1:
            raise ValueError(f"y must have only 1 dimension")

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"y and X must have same length")

        if self.is_classification:
            self.classes = sorted(list(set(y)))

        self.root = self._build(X, y)

    def predict(self, X):
        return np.array([self._estimate(x) for x in X])

    def predict_proba(self, X):
        """Predict class probabilities of the input samples X."""
        if not self.is_classification:
            raise Exception("predict_proba is only for classification tree")

        return np.array([self._proba(x) for x in X])

    def _build(self, X, y, depth: int = 0) -> DecisionTreeNode:
        if self._stop_criteria(X, y, depth):
            return self._create_leaf(X, y, depth)

        node = DecisionTreeNode(depth)

        if self.debug:
            classes, counts = np.unique(y, return_counts=True)
            stats = list(zip(classes, counts))
            sorted_stats = sorted(stats, key=lambda x: x[1], reverse=True)
            node.stats = sorted_stats

        node.score = self._score(X, y)

        split_feature_index, threshold = self._find_best_split(X, y)
        node.split_feature_index = split_feature_index
        node.threshold = threshold

        left_mask = X[:, split_feature_index] < threshold

        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _create_leaf(self, X, y, depth):
        node = DecisionTreeNode(depth)

        if self.is_classification:
            classes, counts = np.unique(y, return_counts=True)
            stats = list(zip(classes, counts))
            sorted_stats = sorted(stats, key=lambda x: x[1], reverse=True)
            prediction = sorted_stats[0][0]
            node.stats = sorted_stats
        else:
            prediction = sum(y) / len(y)

        node.make_leaf(prediction=prediction)
        node.score = self._score(X, y)
        return node

    def _stop_criteria(self, X, y, depth):
        if depth > self.max_depth:
            return True

        if len(np.unique(y)) == 1:
            return True

        if len(y) < self.min_samples_split:
            return True

        return False

    def _estimate(self, x):
        current_node = self.root
        while not current_node.is_leaf:
            if x[current_node.split_feature_index] < current_node.threshold:
                current_node = current_node.left
            else:
                current_node = current_node.right
        return current_node.prediction

    def _proba(self, x):
        current_node = self.root
        while not current_node.is_leaf:
            if x[current_node.split_feature_index] < current_node.threshold:
                current_node = current_node.left
            else:
                current_node = current_node.right

        stats = current_node.stats

        node_samples_count = sum(map(lambda x: x[1], stats))
        raw_result = list(map(lambda x: (x, 0.0), self.classes))

        for class_name, count in stats:
            index = raw_result.index((class_name, 0))
            raw_result[index] = (class_name, count)

        result = np.array(list(map(lambda x: x[1], raw_result))) / node_samples_count
        return result

    def _find_best_split(self, X, y):
        best_feature_index = None
        best_feature_threshold = None
        best_score = -np.inf
        best_uniformity = 0

        for feature_index in range(X.shape[1]):
            feature_column = X[:, feature_index]

            for threshold in list(set(feature_column)):
                score, uniformity = self._split_score(X, y, feature_index, threshold)
                if score > best_score and uniformity > best_uniformity:
                    best_score = score
                    best_feature_index = feature_index
                    best_feature_threshold = threshold
                    best_uniformity = uniformity

        return best_feature_index, best_feature_threshold

    def _score(self, X, y):
        fn = DecisionTree.CRITERION_MAP[self.criterion]

        score = fn(y)
        return score

    def _split_score(self, dataset, y, feature_index, threshold):
        mask = dataset[:, feature_index] < threshold

        left = y[mask]
        right = y[~mask]

        default_score = -np.inf
        default_uniformity = 0

        if len(left) == 0 or len(right) == 0:
            return default_score, default_uniformity

        fn = DecisionTree.CRITERION_MAP[self.criterion]

        score = fn(y) - len(left) * fn(left) / len(y) - len(right) * fn(right) / len(y)

        uniformity = len(left) * len(right)

        return score, uniformity
