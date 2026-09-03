#!/usr/bin/env python3
"""Classify the definiteness of a real symmetric matrix."""

import numpy as np


def definiteness(matrix):
    """Classify eigenvalue signs, returning None for invalid matrices."""
    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a numpy.ndarray')
    if (matrix.ndim != 2 or matrix.shape[0] == 0
            or matrix.shape[0] != matrix.shape[1]
            or not np.isrealobj(matrix)
            or not np.array_equal(matrix, matrix.T)):
        return None
    try:
        if not np.all(np.isfinite(matrix)):
            return None
        eigenvalues = np.linalg.eigvalsh(matrix)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None
    positive = eigenvalues > 0
    negative = eigenvalues < 0
    if np.all(positive):
        return 'Positive definite'
    if np.all(negative):
        return 'Negative definite'
    if not np.any(negative):
        return 'Positive semi-definite'
    if not np.any(positive):
        return 'Negative semi-definite'
    return 'Indefinite'
