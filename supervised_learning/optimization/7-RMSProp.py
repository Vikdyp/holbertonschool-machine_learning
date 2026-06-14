#!/usr/bin/env python3
"""Update variables with the RMSProp optimization algorithm."""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Return an updated variable and second moment using RMSProp."""
    s = beta2 * s + (1 - beta2) * grad ** 2
    var = var - alpha * grad / (np.sqrt(s) + epsilon)
    return var, s
