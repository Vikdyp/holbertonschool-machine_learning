#!/usr/bin/env python3
"""Measure total within-cluster squared deviation."""
import numpy as np


def variance(X, C):
    """Sum each observation's squared distance to its nearest centroid."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape
            or not isinstance(C, np.ndarray) or C.ndim != 2
            or 0 in C.shape or C.shape[1] != X.shape[1]):
        return None
    distances = np.sum((X[:, None, :] - C) ** 2, axis=2)
    return np.sum(np.min(distances, axis=1))
