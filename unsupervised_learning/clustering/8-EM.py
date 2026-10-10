#!/usr/bin/env python3
"""Fit a Gaussian mixture with expectation maximization."""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Alternate EM updates until the likelihood converges or time runs out."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape
            or type(k) is not int or not 0 < k <= X.shape[0]
            or type(iterations) is not int or iterations <= 0
            or type(tol) is not float or tol < 0 or not np.isfinite(tol)
            or type(verbose) is not bool):
        return (None,) * 5
    pi, m, S = initialize(X, k)
    if pi is None:
        return (None,) * 5
    g, likelihood = expectation(X, pi, m, S)
    if g is None:
        return (None,) * 5
    if verbose:
        print("Log Likelihood after 0 iterations: {}".format(
            round(likelihood, 5)))
    for i in range(1, iterations + 1):
        previous = likelihood
        pi, m, S = maximization(X, g)
        if pi is None:
            return (None,) * 5
        g, likelihood = expectation(X, pi, m, S)
        if g is None:
            return (None,) * 5
        finished = abs(likelihood - previous) <= tol or i == iterations
        if verbose and (i % 10 == 0 or finished):
            print("Log Likelihood after {} iterations: {}".format(
                i, round(likelihood, 5)))
        if finished:
            break
    return pi, m, S, g, likelihood
