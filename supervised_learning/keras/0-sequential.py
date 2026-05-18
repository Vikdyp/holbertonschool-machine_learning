#!/usr/bin/env python3
"""Builds neural networks with Keras Sequential API."""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Build a neural network using the Keras Sequential API."""
    model = K.models.Sequential()

    for i, nodes in enumerate(layers):
        kwargs = {
            'activation': activations[i],
            'kernel_regularizer': K.regularizers.l2(lambtha)
        }

        if i == 0:
            kwargs['input_shape'] = (nx,)

        model.add(K.layers.Dense(nodes, **kwargs))

        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
