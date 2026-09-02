#!/usr/bin/env python3
"""Perform neural network forward propagation with dropout."""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Return layer activations and hidden-layer dropout masks."""
    cache = {"A0": X}

    for layer in range(1, L + 1):
        previous = cache[f"A{layer - 1}"]
        z = np.matmul(weights[f"W{layer}"], previous) + weights[f"b{layer}"]

        if layer == L:
            exponential = np.exp(z)
            activation = exponential / np.sum(
                exponential, axis=0, keepdims=True
            )
        else:
            activation = np.tanh(z)
            mask = np.random.binomial(1, keep_prob, activation.shape)
            cache[f"D{layer}"] = mask
            activation = activation * mask / keep_prob

        cache[f"A{layer}"] = activation

    return cache
