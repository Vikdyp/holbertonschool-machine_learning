#!/usr/bin/env python3
"""Construct the ResNet-50 image classification architecture."""

from tensorflow import keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Return a ResNet-50 model for 224x224 RGB images and 1000 classes."""
    inputs = K.Input(shape=(224, 224, 3))
    A = K.layers.Conv2D(
        64, (7, 7), strides=(2, 2), padding='same',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(inputs)
    A = K.layers.BatchNormalization(axis=3)(A)
    A = K.layers.Activation('relu')(A)
    A = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same'
    )(A)

    for filters, blocks, stride in (
        ([64, 64, 256], 3, 1),
        ([128, 128, 512], 4, 2),
        ([256, 256, 1024], 6, 2),
        ([512, 512, 2048], 3, 2),
    ):
        A = projection_block(A, filters, s=stride)
        for _ in range(blocks - 1):
            A = identity_block(A, filters)

    A = K.layers.AveragePooling2D(pool_size=(7, 7))(A)
    outputs = K.layers.Dense(
        1000, activation='softmax',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(A)
    return K.Model(inputs=inputs, outputs=outputs)
