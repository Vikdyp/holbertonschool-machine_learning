#!/usr/bin/env python3
"""Calculate normalization constants for a matrix."""

import numpy as np


def normalization_constants(X):
    """Return the mean and standard deviation for each feature in X."""
    return np.mean(X, axis=0), np.std(X, axis=0)
