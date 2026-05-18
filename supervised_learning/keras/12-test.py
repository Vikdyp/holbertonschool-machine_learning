#!/usr/bin/env python3
"""Tests Keras models."""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Evaluate a Keras model on test data."""
    return network.evaluate(data, labels, verbose=verbose)
