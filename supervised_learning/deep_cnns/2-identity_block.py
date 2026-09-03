#!/usr/bin/env python3
"""Build an identity bottleneck block for a residual network."""

from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Return a residual block whose shortcut preserves the input tensor.

    Args:
        A_prev: Input tensor in channels-last format.
        filters: Filters for the 1x1, 3x3 and final 1x1 convolutions.
    """
    F11, F3, F12 = filters
    A = K.layers.Conv2D(
        F11, (1, 1), kernel_initializer=K.initializers.HeNormal(seed=0)
    )(A_prev)
    A = K.layers.BatchNormalization(axis=3)(A)
    A = K.layers.Activation('relu')(A)

    A = K.layers.Conv2D(
        F3, (3, 3), padding='same',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(A)
    A = K.layers.BatchNormalization(axis=3)(A)
    A = K.layers.Activation('relu')(A)

    A = K.layers.Conv2D(
        F12, (1, 1), kernel_initializer=K.initializers.HeNormal(seed=0)
    )(A)
    A = K.layers.BatchNormalization(axis=3)(A)
    A = K.layers.Add()([A, A_prev])
    return K.layers.Activation('relu')(A)
