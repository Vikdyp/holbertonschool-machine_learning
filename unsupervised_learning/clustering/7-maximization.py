#!/usr/bin/env python3
"""Update Gaussian mixture parameters from posterior responsibilities."""
import numpy as np


def maximization(X, g):
    """Return maximum-likelihood priors, weighted means and covariances."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape
            or not isinstance(g, np.ndarray) or g.ndim != 2
            or g.shape[0] == 0 or g.shape[1] != X.shape[0]
            or not np.allclose(np.sum(g, axis=0), 1)):
        return None, None, None
    counts = np.sum(g, axis=1)
    if np.any(counts == 0):
        return None, None, None
    pi = counts / X.shape[0]
    m = g @ X / counts[:, None]
    S = np.empty((g.shape[0], X.shape[1], X.shape[1]))
    for j in range(g.shape[0]):
        centered = X - m[j]
        S[j] = (centered.T * g[j]) @ centered / counts[j]
    return pi, m, S
