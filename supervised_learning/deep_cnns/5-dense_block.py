#!/usr/bin/env python3
"""Build a DenseNet block with bottleneck convolutions."""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Return concatenated features and their updated channel count.

    Args:
        X: Input tensor in channels-last format.
        nb_filters: Number of input channels.
        growth_rate: Number of new feature maps produced by each layer.
        layers: Number of bottleneck layers in the block.
    """
    for _ in range(layers):
        A = K.layers.BatchNormalization(axis=3)(X)
        A = K.layers.Activation('relu')(A)
        A = K.layers.Conv2D(
            4 * growth_rate, (1, 1),
            kernel_initializer=K.initializers.HeNormal(seed=0)
        )(A)
        A = K.layers.BatchNormalization(axis=3)(A)
        A = K.layers.Activation('relu')(A)
        A = K.layers.Conv2D(
            growth_rate, (3, 3), padding='same',
            kernel_initializer=K.initializers.HeNormal(seed=0)
        )(A)
        X = K.layers.Concatenate(axis=3)([X, A])
        nb_filters += growth_rate
    return X, nb_filters
