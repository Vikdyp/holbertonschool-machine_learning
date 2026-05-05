#!/usr/bin/env python3
"""Build basic decision tree classes."""

import numpy as np


class Node:
    """Decision tree internal node."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize a Node instance."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def left_child_add_prefix(self, text):
        """Add the visual prefix for a left child subtree."""
        lines = text.rstrip("\n").split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "    |  " + line + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Add the visual prefix for a right child subtree."""
        lines = text.rstrip("\n").split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "       " + line + "\n"
        return new_text

    def __str__(self):
        """Return a printable representation of the node and its children."""
        if self.is_root:
            text = "root "
        else:
            text = "-> node "

        text += f"[feature={self.feature}, "
        text += f"threshold={self.threshold}]\n"
        text += self.left_child_add_prefix(self.left_child.__str__())
        text += self.right_child_add_prefix(self.right_child.__str__())
        return text

    def get_leaves_below(self):
        """Return all leaves below the current node."""
        leaves = self.left_child.get_leaves_below()
        leaves += self.right_child.get_leaves_below()
        return leaves

    def max_depth_below(self):
        """Return the maximum depth below the current node."""
        left_depth = self.left_child.max_depth_below()
        right_depth = self.right_child.max_depth_below()
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """Count nodes below the current node."""
        left_count = self.left_child.count_nodes_below(only_leaves)
        right_count = self.right_child.count_nodes_below(only_leaves)
        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count

    def update_bounds_below(self):
        """Update feature bounds for all nodes below the current node."""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        self.left_child.lower = self.lower.copy()
        self.left_child.upper = self.upper.copy()
        self.left_child.lower[self.feature] = max(
            self.threshold,
            self.left_child.lower.get(self.feature, -np.inf)
        )

        self.right_child.lower = self.lower.copy()
        self.right_child.upper = self.upper.copy()
        self.right_child.upper[self.feature] = min(
            self.threshold,
            self.right_child.upper.get(self.feature, np.inf)
        )

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Update the indicator function for the current node."""
        def is_large_enough(x):
            lower_checks = [
                np.greater(x[:, key], self.lower[key])
                for key in self.lower.keys()
            ]
            return np.all(np.array(lower_checks), axis=0)

        def is_small_enough(x):
            upper_checks = [
                np.less_equal(x[:, key], self.upper[key])
                for key in self.upper.keys()
            ]
            return np.all(np.array(upper_checks), axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0
        )

    def pred(self, x):
        """Predict the value for a single individual."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)


class Leaf(Node):
    """Decision tree leaf."""

    def __init__(self, value, depth=None):
        """Initialize a Leaf instance."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """Return a printable representation of the leaf."""
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """Return this leaf in a list."""
        return [self]

    def max_depth_below(self):
        """Return this leaf's depth."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Count this leaf."""
        return 1

    def update_bounds_below(self):
        """Stop bounds propagation at the leaf."""
        pass

    def pred(self, x):
        """Return the prediction value stored in this leaf."""
        return self.value


class Decision_Tree:
    """Decision tree model."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialize a Decision_Tree instance."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def __str__(self):
        """Return a printable representation of the tree."""
        return self.root.__str__()

    def get_leaves(self):
        """Return the leaves of the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Update lower and upper bounds for all nodes in the tree."""
        self.root.update_bounds_below()

    def pred(self, x):
        """Predict the value for a single individual."""
        return self.root.pred(x)

    def update_predict(self):
        """Update the vectorized prediction function."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda A: np.sum(
            np.array([
                leaf.value * leaf.indicator(A)
                for leaf in leaves
            ]),
            axis=0
        )

    def depth(self):
        """Return the maximum depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count nodes in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def np_extrema(self, arr):
        """Return the minimum and maximum values of an array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Choose a random feature and threshold for a node."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_values = self.explanatory[:, feature][node.sub_population]
            feature_min, feature_max = self.np_extrema(feature_values)
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Create a leaf child from a sub-population."""
        values, counts = np.unique(
            self.target[sub_population],
            return_counts=True
        )
        value = values[np.argmax(counts)]
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create a node child from a sub-population."""
        node_child = Node()
        node_child.depth = node.depth + 1
        node_child.sub_population = sub_population
        return node_child

    def is_leaf(self, node, sub_population):
        """Return True if a child should be a leaf."""
        targets = self.target[sub_population]
        if targets.size < self.min_pop:
            return True
        if node.depth + 1 >= self.max_depth:
            return True
        if np.unique(targets).size == 1:
            return True
        return False

    def fit_node(self, node):
        """Recursively fit a node and its children."""
        node.feature, node.threshold = self.split_criterion(node)

        left_population = np.logical_and(
            node.sub_population,
            self.explanatory[:, node.feature] > node.threshold
        )
        right_population = np.logical_and(
            node.sub_population,
            self.explanatory[:, node.feature] <= node.threshold
        )

        if self.is_leaf(node, left_population):
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        if self.is_leaf(node, right_population):
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def accuracy(self, test_explanatory, test_target):
        """Return the accuracy of the tree on test data."""
        return np.sum(
            np.equal(self.predict(test_explanatory), test_target)
        ) / test_target.size

    def fit(self, explanatory, target, verbose=0):
        """Fit the decision tree on explanatory data and targets."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype="bool")

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {self.accuracy(self.explanatory,
                                                 self.target)}""")
