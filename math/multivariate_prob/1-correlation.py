#!/usr/bin/env python3
"""Convert a covariance matrix to a correlation matrix."""
import numpy as np


def correlation(C):
    """Normalize each covariance by the corresponding standard deviations."""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    std = np.sqrt(np.diag(C))
    return C / np.outer(std, std)
