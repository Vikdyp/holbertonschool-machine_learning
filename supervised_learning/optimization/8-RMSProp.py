#!/usr/bin/env python3
"""Create a TensorFlow RMSProp optimizer."""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """Return an RMSProp optimizer with the requested hyperparameters."""
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon,
    )
