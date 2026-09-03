#!/usr/bin/env python3
"""Build the modified LeNet-5 classifier with Keras."""

from tensorflow import keras as K


def lenet5(X):
    """Return a reproducibly initialized and compiled digit classifier."""
    layer = K.layers.Conv2D(
        6, (5, 5), padding='same', activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0))(X)
    layer = K.layers.MaxPooling2D((2, 2), strides=(2, 2))(layer)
    layer = K.layers.Conv2D(
        16, (5, 5), padding='valid', activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0))(layer)
    layer = K.layers.MaxPooling2D((2, 2), strides=(2, 2))(layer)
    layer = K.layers.Flatten()(layer)
    layer = K.layers.Dense(
        120, activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0))(layer)
    layer = K.layers.Dense(
        84, activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0))(layer)
    output = K.layers.Dense(
        10, activation='softmax',
        kernel_initializer=K.initializers.HeNormal(seed=0))(layer)
    model = K.Model(inputs=X, outputs=output)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model
