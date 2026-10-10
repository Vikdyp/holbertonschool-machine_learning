#!/usr/bin/env python3
"""Evaluate a multivariate Gaussian density with a positive floor."""
import numpy as np


def pdf(X, m, S):
    """Return Gaussian densities, or None for an invalid distribution."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape
            or not isinstance(m, np.ndarray) or m.shape != (X.shape[1],)
            or not isinstance(S, np.ndarray)
            or S.shape != (X.shape[1], X.shape[1])):
        return None
    if not np.allclose(S, S.T):
        return None
    try:
        np.linalg.cholesky(S)
        inverse = np.linalg.inv(S)
        _, log_determinant = np.linalg.slogdet(S)
    except np.linalg.LinAlgError:
        return None
    centered = X - m
    exponent = np.sum((centered @ inverse) * centered, axis=1)
    log_normalizer = X.shape[1] * np.log(2 * np.pi) + log_determinant
    return np.maximum(np.exp(-(exponent + log_normalizer) / 2), 1e-300)
