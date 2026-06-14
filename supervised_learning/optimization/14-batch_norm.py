#!/usr/bin/env python3
"""Create a TensorFlow dense layer with batch normalization."""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Return the activated output of a batch-normalized dense layer."""
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(n, kernel_initializer=initializer)
    Z = layer(prev)

    mean, variance = tf.nn.moments(Z, axes=[0])
    gamma = tf.Variable(tf.ones((n,)), trainable=True)
    beta = tf.Variable(tf.zeros((n,)), trainable=True)
    Z_norm = tf.nn.batch_normalization(Z, mean, variance, beta, gamma, 1e-7)

    return activation(Z_norm)
