#!/usr/bin/env python3
"""Builds neural networks with Keras functional API."""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Build a neural network using the Keras functional API."""
    inputs = K.Input(shape=(nx,))
    outputs = inputs

    for i, nodes in enumerate(layers):
        outputs = K.layers.Dense(
            nodes,
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(outputs)

        if i < len(layers) - 1:
            outputs = K.layers.Dropout(1 - keep_prob)(outputs)

    return K.models.Model(inputs=inputs, outputs=outputs)
