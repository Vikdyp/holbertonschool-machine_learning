#!/usr/bin/env python3
"""Perform gradient descent with dropout regularization."""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Update network weights and biases using cached dropout masks."""
    m = Y.shape[1]
    dz = cache[f"A{L}"] - Y

    for layer in range(L, 0, -1):
        weight = weights[f"W{layer}"]
        previous_activation = cache[f"A{layer - 1}"]
        dw = np.matmul(dz, previous_activation.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        if layer > 1:
            previous_error = np.matmul(weight.T, dz)
            previous_error *= cache[f"D{layer - 1}"]
            previous_error /= keep_prob
            dz = previous_error * (1 - np.square(previous_activation))

        weights[f"W{layer}"] = weight - alpha * dw
        weights[f"b{layer}"] = weights[f"b{layer}"] - alpha * db
