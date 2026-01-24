#!/usr/bin/env python3
"""
2-build_decision_tree.py

Decision tree structures for printing (Node / Leaf / Decision_Tree).

This file focuses on string representation of a binary decision tree,
matching the expected output format from 2-main.py.
"""

from __future__ import annotations
from typing import Optional


class Node:
    """Internal decision-tree node."""

    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child: Optional["Node"] = None,
        right_child: Optional["Node"] = None,
        is_root: bool = False,
        depth: int = 0,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def left_child_add_prefix(self, text: str) -> str:
        """
        Prefix formatting for the left subtree.

        Parameters
        ----------
        text : str
            Multiline string representing the left child.

        Returns
        -------
        str
            Prefixed text with proper tree connectors.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text: str) -> str:
        """
        Prefix formatting for the right subtree.

        Parameters
        ----------
        text : str
            Multiline string representing the right child.

        Returns
        -------
        str
            Prefixed text with proper tree connectors.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self) -> str:
        """
        String representation of the node and its subtree.

        Returns
        -------
        str
            Pretty-printed tree with connectors, matching project expected output.
        """
        label = "root" if self.is_root else "node"
        header = f"{label} [feature={self.feature}, threshold={self.threshold}]"

        if self.is_leaf:
            # Defensive: in this project, leaves are Leaf objects.
            # But keep a fallback if a node was converted to leaf elsewhere.
            value = getattr(self, "value", None)
            return f"-> leaf [value={value}]"

        left_text = self.left_child.__str__() if self.left_child is not None else ""
        right_text = self.right_child.__str__() if self.right_child is not None else ""

        out = header + "\n"
        if left_text:
            out += self.left_child_add_prefix(left_text)
        if right_text:
            out += self.right_child_add_prefix(right_text)
        return out.rstrip("\n")


class Leaf(Node):
    """Leaf node holding a predicted value."""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self) -> str:
        """
        String representation of a leaf.

        Returns
        -------
        str
            Leaf label matching project expected output.
        """
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """Decision Tree container class (printing delegates to root)."""

    def __init__(
        self,
        max_depth=10,
        min_pop=1,
        seed=0,
        split_criterion="random",
        root=None,
    ):
        self.rng = None
        self.root = root if root is not None else Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def __str__(self) -> str:
        """
        String representation of the entire tree.

        Returns
        -------
        str
            Tree string produced by the root node.
        """
        return self.root.__str__()
