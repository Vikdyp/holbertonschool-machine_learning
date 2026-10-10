#!/usr/bin/env python3
"""Initialize K-means centroids inside the observed coordinate bounds."""
import numpy as np


def initialize(X, k):
    """Draw k uniformly distributed centroids, or return None on failure."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape
            or type(k) is not int or not 0 < k <= X.shape[0]):
        return None
    return np.random.uniform(np.min(X, axis=0), np.max(X, axis=0),
                             (k, X.shape[1]))
