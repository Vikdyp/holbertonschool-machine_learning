#!/usr/bin/env python3
"""Perform gradient descent with L2 regularization."""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Update network weights and biases using L2-regularized gradients."""
    m = Y.shape[1]
    dz = cache[f"A{L}"] - Y

    for layer in range(L, 0, -1):
        weight = weights[f"W{layer}"]
        previous_activation = cache[f"A{layer - 1}"]
        dw = (
            np.matmul(dz, previous_activation.T) + lambtha * weight
        ) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        if layer > 1:
            previous_error = np.matmul(weight.T, dz)
            dz = previous_error * (1 - np.square(previous_activation))

        weights[f"W{layer}"] = weight - alpha * dw
        weights[f"b{layer}"] = weights[f"b{layer}"] - alpha * db
