#!/usr/bin/env python3
"""Compress and downsample feature maps between DenseNet blocks."""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """Return pooled features and the compressed channel count.

    Args:
        X: Input tensor in channels-last format.
        nb_filters: Number of input channels.
        compression: Fraction of channels retained by the projection.
    """
    nb_filters = int(nb_filters * compression)
    A = K.layers.BatchNormalization(axis=3)(X)
    A = K.layers.Activation('relu')(A)
    A = K.layers.Conv2D(
        nb_filters, (1, 1),
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(A)
    A = K.layers.AveragePooling2D(
        pool_size=(2, 2), strides=(2, 2)
    )(A)
    return A, nb_filters
