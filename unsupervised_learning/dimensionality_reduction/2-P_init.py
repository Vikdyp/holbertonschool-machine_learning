#!/usr/bin/env python3
"""Initialize distances and conditional affinity search parameters."""
import numpy as np


def P_init(X, perplexity):
    """Return squared distances, zero affinities, betas and target entropy."""
    n = X.shape[0]
    squared = np.sum(X * X, axis=1)
    D = squared[:, None] + squared[None, :] - 2 * X @ X.T
    np.fill_diagonal(D, 0)
    return D, np.zeros((n, n)), np.ones((n, 1)), np.log2(perplexity)
