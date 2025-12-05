#!/usr/bin/env python3
"""
This module provides a function that performs
element-wise operations on two numpy-compatible objects.
"""


def np_elementwise(mat1, mat2):
    """
    Returns the element-wise sum, difference,
    product, and quotient of two matrices.
    """
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
