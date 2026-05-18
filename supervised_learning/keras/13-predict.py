#!/usr/bin/env python3
"""Makes predictions with Keras models."""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Make predictions with a Keras model."""
    return network.predict(data, verbose=verbose)
