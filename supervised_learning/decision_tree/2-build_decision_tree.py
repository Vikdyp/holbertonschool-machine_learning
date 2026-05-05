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
