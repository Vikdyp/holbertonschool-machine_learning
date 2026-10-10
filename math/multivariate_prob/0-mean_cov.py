#!/usr/bin/env python3
"""Estimate the mean and sample covariance of multivariate observations."""
import numpy as np


def mean_cov(X):
    """Return the row mean and unbiased covariance of a two-dimensional set."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0, keepdims=True)
    centered = X - mean
    cov = centered.T @ centered / (X.shape[0] - 1)
    return mean, cov
