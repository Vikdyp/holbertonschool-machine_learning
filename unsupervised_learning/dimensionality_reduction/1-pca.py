#!/usr/bin/env python3
"""Project observations onto a fixed number of principal components."""
import numpy as np


def pca(X, ndim):
    """Center observations and return their principal component coordinates."""
    centered = X - np.mean(X, axis=0)
    _, _, vectors = np.linalg.svd(centered,
                                  full_matrices=ndim > min(X.shape))
    return centered @ vectors[:ndim].T
