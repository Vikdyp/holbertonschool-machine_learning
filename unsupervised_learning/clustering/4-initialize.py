#!/usr/bin/env python3
"""Initialize a Gaussian mixture from K-means centroids."""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Return equal priors, K-means means and identity covariances."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape
            or type(k) is not int or not 0 < k <= X.shape[0]):
        return None, None, None
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None
    pi = np.full(k, 1 / k)
    S = np.repeat(np.eye(X.shape[1])[None, :, :], k, axis=0)
    return pi, m, S
