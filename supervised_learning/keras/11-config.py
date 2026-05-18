#!/usr/bin/env python3
"""Saves and loads Keras model configurations."""

import tensorflow.keras as K


def save_config(network, filename):
    """Save a Keras model's configuration in JSON format."""
    with open(filename, 'w') as file:
        file.write(network.to_json())


def load_config(filename):
    """Load a Keras model from a JSON configuration file."""
    with open(filename, 'r') as file:
        return K.models.model_from_json(file.read())
