#!/usr/bin/env python3
"""Compute the minor of a square matrix."""


def _determinant(matrix):
    """Expand a validated square matrix along its first row."""
    size = len(matrix)
    if size == 0:
        return 1
    if size == 1:
        return matrix[0][0]
    if size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    total = 0
    for column, value in enumerate(matrix[0]):
        submatrix = [row[:column] + row[column + 1:] for row in matrix[1:]]
        total += (-1) ** column * value * _determinant(submatrix)
    return total


def minor(matrix):
    """Validate the matrix and return its minor."""
    if (not isinstance(matrix, list) or not matrix
            or any(not isinstance(row, list) for row in matrix)):
        raise TypeError('matrix must be a list of lists')
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError('matrix must be a non-empty square matrix')
    size = len(matrix)
    result = []
    for row in range(size):
        values = []
        for column in range(size):
            submatrix = [line[:column] + line[column + 1:]
                         for index, line in enumerate(matrix) if index != row]
            value = _determinant(submatrix)
            values.append(value)
        result.append(values)
    return result
