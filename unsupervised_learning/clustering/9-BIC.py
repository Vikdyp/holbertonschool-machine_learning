#!/usr/bin/env python3
"""Select a Gaussian mixture size with the Bayesian information criterion."""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Return the minimum-BIC model and all candidate likelihoods and BICs."""
    if not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape:
        return (None,) * 4
    n, d = X.shape
    if kmax is None:
        kmax = n
    if (type(kmin) is not int or type(kmax) is not int
            or not 0 < kmin <= kmax <= n
            or type(iterations) is not int or iterations <= 0
            or type(tol) is not float or tol < 0 or not np.isfinite(tol)
            or type(verbose) is not bool):
        return (None,) * 4
    likelihoods = np.empty(kmax - kmin + 1)
    criteria = np.empty_like(likelihoods)
    best_k, best_result, best_score = None, None, np.inf
    for k in range(kmin, kmax + 1):
        pi, m, S, _, likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None:
            return (None,) * 4
        parameters = k * (d + d * (d + 1) / 2) + k - 1
        score = parameters * np.log(n) - 2 * likelihood
        likelihoods[k - kmin], criteria[k - kmin] = likelihood, score
        if score < best_score:
            best_k, best_result, best_score = k, (pi, m, S), score
    return best_k, best_result, likelihoods, criteria
