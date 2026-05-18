#!/usr/bin/env python3
"""Converts label vectors to one-hot matrices."""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Convert labels to a one-hot matrix."""
    return K.utils.to_categorical(labels, num_classes=classes)
