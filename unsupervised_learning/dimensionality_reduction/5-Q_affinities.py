#!/usr/bin/env python3
"""Compute Student-t affinities for an embedding."""
import numpy as np


def Q_affinities(Y):
    """Return normalized pairwise affinities and their unscaled numerators."""
    squared = np.sum(Y * Y, axis=1)
    distances = squared[:, None] + squared[None, :] - 2 * Y @ Y.T
    num = 1 / (1 + distances)
    np.fill_diagonal(num, 0)
    return num / np.sum(num), num
