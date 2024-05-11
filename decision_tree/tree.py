import numpy as np


class Node:
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
    ) -> None:
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def isLeaf(self) -> bool:
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(
        self,
        max_depth: int = 10,
        criterion: str = "gini",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int = None,
        min_impurity_decrease: float = 0.0,
        max_leaf_nodes: int = None,
        random_state: int = None,
    ) -> None:
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.root = None
        self.classes_ = None
        self._fitted = False

        np.random.seed(random_state)

    def fit(self, X, y, sample_weights=None):
        if sample_weights is None:
            sample_weights = np.ones(len(y))
        sample_weights = np.array(sample_weights)

        self.classes_ = np.unique(y)
        self.root = self._build_tree(X, y, 0, sample_weights)
        self._fitted = True
        return self

    def _stopping_criteria(self, y, depth):
        if len(np.unique(y)) == 1:
            return True
        if depth == self.max_depth:
            return True
        if len(y) < self.min_samples_split:
            return True
        return False

    def _build_tree(self, X, y, depth, sample_weights):
        if self._stopping_criteria(y, depth):
            leaf_value = self._calculate_leaf_value(y, sample_weights)
            return Node(value=leaf_value)

        feature_index, threshold = self._best_split(X, y, sample_weights)
        if feature_index is None:
            leaf_value = self._calculate_leaf_value(y, sample_weights)
            return Node(value=leaf_value)

        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold

        X_left, X_right = X[left_indices], X[right_indices]
        y_left, y_right = y[left_indices], y[right_indices]

        weights_left = sample_weights[left_indices]
        weights_right = sample_weights[right_indices]

        left_subtree = self._build_tree(X_left, y_left, depth + 1, weights_left)
        right_subtree = self._build_tree(X_right, y_right, depth + 1, weights_right)

        return Node(
            feature_index=feature_index,
            threshold=threshold,
            left=left_subtree,
            right=right_subtree,
        )

    def _calculate_leaf_value(self, y, sample_weights) -> dict:
        classes, counts = np.unique(y, return_counts=True)
        weighted_counts = np.zeros_like(counts, dtype=np.float64)

        for i, cl in enumerate(classes):
            weighted_counts[i] = sample_weights[y == cl].sum()

        return dict(zip(classes, weighted_counts))

    def _best_split(self, X, y, sample_weights):
        best_gain = 0
        best_feature, best_threshold = None, None
        current_impurity = self._calculate_impurity(y, sample_weights)

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold

                if (
                    np.sum(left_indices) < self.min_samples_split
                    or np.sum(right_indices) < self.min_samples_split
                ):
                    continue

                gain = self._calculate_gain(
                    y,
                    sample_weights,
                    left_indices,
                    right_indices,
                    current_impurity,
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_impurity(self, y, sample_weights) -> float:
        if self.criterion == "gini":
            return self._calculate_gini(y, sample_weights)
        elif self.criterion == "entropy":
            return self._calculate_entropy(y, sample_weights)

    def _calculate_gini(self, y, sample_weights) -> float:
        total_weight = np.sum(sample_weights)
        _, counts = np.unique(y, return_counts=True)
        weighted_counts = np.zeros_like(counts, dtype=np.float64)

        for i, cl in enumerate(np.unique(y)):
            weighted_counts[i] = np.sum(sample_weights[y == cl])

        prob = weighted_counts / total_weight
        gini = 1 - np.sum(prob**2)

        return gini

    def _calculate_entropy(self, y, sample_weights) -> float:
        total_weight = np.sum(sample_weights)
        _, counts = np.unique(y, return_counts=True)
        weighted_counts = np.zeros_like(counts, dtype=np.float64)

        for i, cl in enumerate(np.unique(y)):
            weighted_counts[i] = np.sum(sample_weights[y == cl])

        prob = weighted_counts / total_weight
        entropy = -np.sum(prob * np.log2(prob + np.finfo(float).eps))

        return entropy

    def _calculate_gain(
        self,
        y,
        sample_weights,
        left_indices,
        right_indices,
        current_impurity,
    ):
        left_weight = np.sum(sample_weights[left_indices])
        right_weight = np.sum(sample_weights[right_indices])
        total_weight = left_weight + right_weight

        weights_left = sample_weights[left_indices]
        weights_right = sample_weights[right_indices]

        left_impurity = self._calculate_impurity(y[left_indices], weights_left)
        right_impurity = self._calculate_impurity(y[right_indices], weights_right)

        weighted_impurity = (left_weight / total_weight) * left_impurity
        weighted_impurity += (right_weight / total_weight) * right_impurity
        gain = current_impurity - weighted_impurity

        return gain

    def predict(self, X):
        preds = [self._predict_single(self.root, x) for x in X]
        return np.array(preds)

    def _predict_single(self, node, x):
        while node.value is None:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return max(node.value, key=node.value.get)
