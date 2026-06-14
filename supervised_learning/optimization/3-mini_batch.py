#!/usr/bin/env python3
"""Create mini-batches for mini-batch gradient descent."""

shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """Return shuffled mini-batches from X and Y."""
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    mini_batches = []

    for start in range(0, X.shape[0], batch_size):
        end = start + batch_size
        mini_batches.append((X_shuffled[start:end], Y_shuffled[start:end]))

    return mini_batches
