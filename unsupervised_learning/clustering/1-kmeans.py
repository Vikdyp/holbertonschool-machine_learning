#!/usr/bin/env python3
"""Fit cluster centroids using Lloyd's K-means algorithm."""
import numpy as np


def kmeans(X, k, iterations=1000):
    """Return centroids and nearest-centroid labels, or a failure tuple."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape
            or type(k) is not int or not 0 < k <= X.shape[0]
            or type(iterations) is not int or iterations <= 0):
        return None, None
    low, high = np.min(X, axis=0), np.max(X, axis=0)
    C = np.random.uniform(low, high, (k, X.shape[1]))
    for _ in range(iterations):
        distances = np.sum((X[:, None, :] - C) ** 2, axis=2)
        clss = np.argmin(distances, axis=1)
        updated = np.empty_like(C)
        for j in range(k):
            members = X[clss == j]
            if members.shape[0]:
                updated[j] = np.mean(members, axis=0)
            else:
                updated[j] = np.random.uniform(low, high, X.shape[1])
        if np.array_equal(C, updated):
            return C, clss
        C = updated
    clss = np.argmin(np.sum((X[:, None, :] - C) ** 2, axis=2), axis=1)
    return C, clss
