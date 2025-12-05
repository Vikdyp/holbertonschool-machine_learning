#!/usr/bin/env python3
"""Module that defines cat_matrices2D function."""


def cat_matrices2D(mat1, mat2, axis=0):
    """Function that concatenates two 2D matrices along a specific axis.

    If axis == 0 → concatenation by rows (vertical stack).
    If axis == 1 → concatenation by columns (horizontal stack).
    Returns a new matrix, or None if shapes are incompatible.
    """
    # axis 0 → vertical concatenation
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    # axis 1 → horizontal concatenation
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        result = []
        for i in range(len(mat1)):
            result.append(mat1[i][:] + mat2[i][:])
        return result

    # invalid axis, not requested but safe fallback
    return None
