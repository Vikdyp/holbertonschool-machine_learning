#!/usr/bin/env python3
"""Select principal component weights for a centered dataset."""
import numpy as np


def pca(X, var=0.95):
    """Retain the task's requested cumulative singular-value fraction."""
    _, singular, vectors = np.linalg.svd(X, full_matrices=False)
    cumulative = np.cumsum(singular) / np.sum(singular)
    ndim = np.searchsorted(cumulative, var) + 1
    return vectors[:ndim].T
