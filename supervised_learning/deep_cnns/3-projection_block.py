#!/usr/bin/env python3
"""Build a residual bottleneck block with a projected shortcut."""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Return a residual block that can change spatial and channel dimensions.

    Args:
        A_prev: Input tensor in channels-last format.
        filters: Filters for the 1x1, 3x3 and final 1x1 convolutions.
        s: Stride for the first convolution and the shortcut projection.
    """
    F11, F3, F12 = filters
    A = K.layers.Conv2D(
        F11, (1, 1), strides=(s, s),
        kernel_initializer=K.initializers.HeNormal(seed=0)
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
    shortcut = K.layers.Conv2D(
        F12, (1, 1), strides=(s, s),
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)
    A = K.layers.Add()([A, shortcut])
    return K.layers.Activation('relu')(A)
