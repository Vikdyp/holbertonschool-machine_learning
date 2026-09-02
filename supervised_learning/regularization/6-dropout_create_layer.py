#!/usr/bin/env python3
"""Create a dense layer followed by dropout."""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Return a dense layer output with optional training-time dropout."""
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode="fan_avg"
    )
    dense = tf.keras.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=initializer,
    )(prev)
    return tf.keras.layers.Dropout(1 - keep_prob)(dense, training=training)
