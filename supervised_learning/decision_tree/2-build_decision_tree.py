#!/usr/bin/env python3
"""
Decision tree building blocks: Node, Leaf, and Decision_Tree.

Task 2: implement __str__ methods to print the tree structure
using the exact prefix style provided in the statement.
"""

import numpy as np


class Node:
    """Internal node of a binary decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize a decision node."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Return the maximum depth found in the subtree below this node."""
        if self.left_child is None or self.right_child is None:
            return self.depth
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """
        Count the number of nodes below this node.

        Args:
            only_leaves (bool): if True, count only leaves

        Returns:
            int: number of nodes or leaves
        """
        left_count = self.left_child.count_nodes_below(only_leaves)
        right_count = self.right_child.count_nodes_below(only_leaves)
        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count

    def left_child_add_prefix(self, text):
        """Add the left-child ASCII prefix (exactly as provided)."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Add the right-child ASCII prefix (symmetric to left)."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Return an ASCII representation of the subtree rooted at this node."""
        if self.is_root:
            text = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            text = f"-> node [feature={self.feature}, threshold={self.threshold}]"

        if self.left_child is None or self.right_child is None:
            return text

        left_str = self.left_child.__str__()
        right_str = self.right_child.__str__()

        text += "\n"
        text += self.left_child_add_prefix(left_str)
        text += self.right_child_add_prefix(right_str)

        return text.rstrip("\n")


class Leaf(Node):
    """Leaf node of a decision tree, storing a predicted value."""

    def __init__(self, value, depth=None):
        """Initialize a leaf."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return this leaf's depth."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """A leaf always counts as one."""
        return 1

    def __str__(self):
        """String representation of a leaf (provided in the statement)."""
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """Decision tree container exposing helper methods."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialize the decision tree."""
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

    def depth(self):
        """Return the maximum depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count nodes or leaves in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """String representation of the whole tree (provided)."""
        return self.root.__str__()
