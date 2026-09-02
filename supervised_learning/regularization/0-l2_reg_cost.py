#!/usr/bin/env python3
"""Calculate a neural network cost with L2 regularization."""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Return the network cost including L2 weight regularization."""
    penalty = sum(
        np.sum(np.square(weights[f"W{layer}"]))
        for layer in range(1, L + 1)
    )
    return cost + lambtha * penalty / (2 * m)
