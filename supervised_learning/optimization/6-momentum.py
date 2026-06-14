#!/usr/bin/env python3
"""Create a TensorFlow momentum optimizer."""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Return an SGD optimizer configured with momentum."""
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
