#!/usr/bin/env python3
"""Saves and loads complete Keras models."""

import tensorflow.keras as K


def save_model(network, filename):
    """Save an entire Keras model."""
    network.save(filename)


def load_model(filename):
    """Load an entire Keras model."""
    return K.models.load_model(filename)
