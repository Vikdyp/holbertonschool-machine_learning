#!/usr/bin/env python3
"""Calculate the t-SNE embedding gradient without the scalar factor four."""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """Return affinity-weighted coordinate gradients and current Q values."""
    Q, num = Q_affinities(Y)
    weights = (P - Q) * num
    dY = np.sum(weights, axis=1, keepdims=True) * Y - weights @ Y
    return dY, Q
