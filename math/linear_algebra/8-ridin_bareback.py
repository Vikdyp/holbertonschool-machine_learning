#!/usr/bin/env python3
"""Module that defines mat_mul function."""


def mat_mul(mat1, mat2):
    """Function that performs matrix multiplication.

    Returns a new matrix resulting from multiplying mat1 by mat2.
    If the matrices cannot be multiplied, returns None.
    """
    # Check if multiplication is possible:
    # number of columns in mat1 must equal number of rows in mat2
    if len(mat1[0]) != len(mat2):
        return None

    result = []
    for i in range(len(mat1)):  # for each row of mat1
        row = []
        for j in range(len(mat2[0])):  # for each column of mat2
            val = 0
            for k in range(len(mat2)):  # dot product
                val += mat1[i][k] * mat2[k][j]
            row.append(val)
        result.append(row)

    return result
