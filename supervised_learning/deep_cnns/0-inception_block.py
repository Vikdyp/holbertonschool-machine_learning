#!/usr/bin/env python3
"""Build the four parallel branches of an Inception module."""

from tensorflow import keras as K


def inception_block(A_prev, filters):
    """Concatenate 1x1, 3x3, 5x5 and pooled convolutional features.

    Args:
        A_prev: Input tensor in channels-last format.
        filters: Counts F1, F3R, F3, F5R, F5 and FPP for the six convolutions.
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    branch1 = K.layers.Conv2D(F1, (1, 1), activation='relu')(A_prev)

    branch3 = K.layers.Conv2D(F3R, (1, 1), activation='relu')(A_prev)
    branch3 = K.layers.Conv2D(
        F3, (3, 3), padding='same', activation='relu'
    )(branch3)

    branch5 = K.layers.Conv2D(F5R, (1, 1), activation='relu')(A_prev)
    branch5 = K.layers.Conv2D(
        F5, (5, 5), padding='same', activation='relu'
    )(branch5)

    branch_pool = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(1, 1), padding='same'
    )(A_prev)
    branch_pool = K.layers.Conv2D(
        FPP, (1, 1), activation='relu'
    )(branch_pool)
    return K.layers.Concatenate(axis=3)(
        [branch1, branch3, branch5, branch_pool]
    )
