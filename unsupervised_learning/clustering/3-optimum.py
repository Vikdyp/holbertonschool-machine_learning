#!/usr/bin/env python3
"""Compare K-means cluster counts with an elbow variance series."""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Return fitted models and variance reductions relative to kmin."""
    if not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if (type(kmin) is not int or type(kmax) is not int
            or not 0 < kmin < kmax <= X.shape[0]
            or type(iterations) is not int or iterations <= 0):
        return None, None
    results, d_vars = [], []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None:
            return None, None
        current = variance(X, C)
        if k == kmin:
            baseline = current
        results.append((C, clss))
        d_vars.append(baseline - current)
    return results, d_vars
