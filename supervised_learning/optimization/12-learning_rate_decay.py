#!/usr/bin/env python3
"""Create a TensorFlow stepwise inverse time decay schedule."""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """Return a Keras inverse time decay learning rate schedule."""
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True,
    )
