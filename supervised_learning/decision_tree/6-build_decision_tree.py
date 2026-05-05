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
