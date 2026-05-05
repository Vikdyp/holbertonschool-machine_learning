#!/usr/bin/env python3
"""Private single neuron for binary classification."""

import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification."""

    def __init__(self, nx):
        """Initialize a neuron.

        Args:
            nx: Number of input features.
        """
        if type(nx) is not int:
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Get the neuron's weights."""
        return self.__W

    @property
    def b(self):
        """Get the neuron's bias."""
        return self.__b

    @property
    def A(self):
        """Get the neuron's activated output."""
        return self.__A
