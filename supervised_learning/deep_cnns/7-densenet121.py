#!/usr/bin/env python3
"""Construct DenseNet-121 with bottlenecks and channel compression."""

from tensorflow import keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Return DenseNet-121 for 224x224 RGB images and 1000 classes.

    Args:
        growth_rate: Feature maps added by each dense layer.
        compression: Channel retention factor between dense blocks.
    """
    inputs = K.Input(shape=(224, 224, 3))
    nb_filters = 2 * growth_rate
    A = K.layers.BatchNormalization(axis=3)(inputs)
    A = K.layers.Activation('relu')(A)
    A = K.layers.Conv2D(
        nb_filters, (7, 7), strides=(2, 2), padding='same',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(A)
    A = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same'
    )(A)

    for layers in (6, 12, 24):
        A, nb_filters = dense_block(A, nb_filters, growth_rate, layers)
        A, nb_filters = transition_layer(A, nb_filters, compression)
    A, nb_filters = dense_block(A, nb_filters, growth_rate, 16)
    A = K.layers.AveragePooling2D(
        pool_size=(7, 7), strides=(1, 1)
    )(A)
    outputs = K.layers.Dense(
        1000, activation='softmax',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(A)
    return K.Model(inputs=inputs, outputs=outputs)
