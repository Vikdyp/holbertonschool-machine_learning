#!/usr/bin/env python3
"""Module that defines add_matrices2D function."""


def add_matrices2D(mat1, mat2):
    """Function that adds two 2D matrices element-wise.

    Returns a new matrix containing the sums.
    If the matrices do not have the same shape, returns None.
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[0])):
            row.append(mat1[i][j] + mat2[i][j])
        result.append(row)

    return result
