#!/usr/bin/env python3
"""Calculate Gaussian mixture responsibilities and log likelihood."""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Normalize prior-weighted component densities for every observation."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape
            or not isinstance(pi, np.ndarray) or pi.ndim != 1
            or pi.size == 0 or not isinstance(m, np.ndarray)
            or m.shape != (pi.size, X.shape[1])
            or not isinstance(S, np.ndarray)
            or S.shape != (pi.size, X.shape[1], X.shape[1])
            or np.any(pi < 0) or not np.isclose(np.sum(pi), 1)):
        return None, None
    weighted = np.empty((pi.size, X.shape[0]))
    for j in range(pi.size):
        density = pdf(X, m[j], S[j])
        if density is None:
            return None, None
        weighted[j] = pi[j] * density
    totals = np.sum(weighted, axis=0)
    return weighted / totals, np.sum(np.log(totals))
