#!/usr/bin/env python3
"""Optimize a t-SNE embedding with momentum and early exaggeration."""
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """Reduce dimensions, fit affinities and center each embedding update."""
    reduced = pca(X, idims)
    P = P_affinities(reduced, perplexity=perplexity)
    Y = np.random.randn(X.shape[0], ndims)
    velocity = np.zeros_like(Y)
    for iteration in range(1, iterations + 1):
        active = P * 4 if iteration <= 100 else P
        gradient, _ = grads(Y, active)
        momentum = 0.5 if iteration <= 20 else 0.8
        velocity = momentum * velocity - lr * gradient
        Y = Y + velocity
        Y = Y - np.mean(Y, axis=0)
        if iteration % 100 == 0:
            _, Q = grads(Y, active)
            print("Cost at iteration {}: {}".format(
                iteration, cost(active, Q)))
    return Y
