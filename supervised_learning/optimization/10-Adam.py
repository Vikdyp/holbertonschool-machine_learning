#!/usr/bin/env python3
"""Create a TensorFlow Adam optimizer."""

import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """Return an Adam optimizer with the requested hyperparameters."""
    return tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon,
    )
