#!/usr/bin/env python3
"""
Module that provides a function to compute the sum of squares
from 1 to n using a closed-form formula.
"""


def summation_i_squared(n):
    """
    Computes the sum of squares from 1 to n.

    Args:
        n (int): stopping condition

    Returns:
        int: sum of i^2 from 1 to n
        None: if n is not a valid number
    """
    if not isinstance(n, int) or n < 1:
        return None

    return n * (n + 1) * (2 * n + 1) // 6
