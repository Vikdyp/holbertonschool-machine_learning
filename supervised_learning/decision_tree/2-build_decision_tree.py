#!/usr/bin/env python3
"""
Decision tree building blocks: Node, Leaf, and Decision_Tree.

Task 2: implement __str__ methods to print the tree structure.
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
        """Prefix a subtree string as a left child."""
        lines = text.split("\n")
        new_lines = ["+--" + lines[0]]
        for line in lines[1:]:
            new_lines.append("| " + line)
        return "\n".join(new_lines)

    def right_child_add_prefix(self, text):
        """
        Prefix a subtree string as a right child.

        Important: do NOT indent subsequent lines, otherwise the right subtree
        shifts and the expected ASCII layout breaks.
        """
        lines = text.split("\n")
        new_lines = ["+--" + lines[0]]
        for line in lines[1:]:
            new_lines.append(line)
        return "\n".join(new_lines)

    def __str__(self):
        """Return an ASCII representation of the subtree rooted at this node."""
        if self.is_root:
            head = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            head = f"-> node [feature={self.feature}, threshold={self.threshold}]"

        if self.left_child is None or self.right_child is None:
            return head

        left_txt = self.left_child_add_prefix(str(self.left_child))
        right_txt = self.right_child_add_prefix(str(self.right_child))
        return "\n".join([head, left_txt, right_txt])


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
        """Return the printable representation of a leaf."""
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
        """
        Count nodes in the tree.

        Args:
            only_leaves (bool): if True, count only leaves

        Returns:
            int: number of nodes or leaves
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Return the printable representation of the whole tree."""
        return str(self.root)
