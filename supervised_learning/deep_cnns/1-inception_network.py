#!/usr/bin/env python3
"""Construct the Inception network for image classification."""

from tensorflow import keras as K

inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Return an Inception model for 224x224 RGB images and 1000 classes."""
    inputs = K.Input(shape=(224, 224, 3))
    A = K.layers.Conv2D(
        64, (7, 7), strides=(2, 2), padding='same', activation='relu'
    )(inputs)
    A = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same'
    )(A)
    A = K.layers.Conv2D(64, (1, 1), activation='relu')(A)
    A = K.layers.Conv2D(
        192, (3, 3), padding='same', activation='relu'
    )(A)
    A = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same'
    )(A)

    A = inception_block(A, [64, 96, 128, 16, 32, 32])
    A = inception_block(A, [128, 128, 192, 32, 96, 64])
    A = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same'
    )(A)
    for filters in (
        [192, 96, 208, 16, 48, 64],
        [160, 112, 224, 24, 64, 64],
        [128, 128, 256, 24, 64, 64],
        [112, 144, 288, 32, 64, 64],
        [256, 160, 320, 32, 128, 128],
    ):
        A = inception_block(A, filters)
    A = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same'
    )(A)
    A = inception_block(A, [256, 160, 320, 32, 128, 128])
    A = inception_block(A, [384, 192, 384, 48, 128, 128])
    A = K.layers.AveragePooling2D(
        pool_size=(7, 7), strides=(1, 1)
    )(A)
    A = K.layers.Dropout(0.4)(A)
    outputs = K.layers.Dense(1000, activation='softmax')(A)
    return K.Model(inputs=inputs, outputs=outputs)
