#!/usr/bin/env python3
"""Search Gaussian scales and construct symmetric t-SNE affinities."""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """Match each conditional entropy, then symmetrize the probabilities."""
    D, P, betas, target = P_init(X, perplexity)
    n = X.shape[0]
    for i in range(n):
        neighbors = np.arange(n) != i
        distances = D[i, neighbors]
        beta = betas[i].copy()
        low, high = None, None
        entropy, probabilities = HP(distances, beta)
        minimum = np.log2(np.count_nonzero(distances == np.min(distances)))
        maximum = np.log2(n - 1)
        if target < minimum - tol or target > maximum + tol:
            raise ValueError("perplexity cannot be matched for this data")
        while abs(entropy - target) > tol:
            difference = entropy - target
            if difference > 0:
                low = beta.copy()
                candidate = beta * 2 if high is None else (beta + high) / 2
            else:
                high = beta.copy()
                candidate = beta / 2 if low is None else (beta + low) / 2
            if not np.all(np.isfinite(candidate)) or np.all(candidate == beta):
                raise ValueError("entropy search cannot reach the tolerance")
            beta = candidate
            entropy, probabilities = HP(distances, beta)
        P[i, neighbors] = probabilities
    return (P + P.T) / (2 * n)
