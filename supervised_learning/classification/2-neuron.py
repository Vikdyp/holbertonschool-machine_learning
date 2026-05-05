#!/usr/bin/env python3
"""Neuron forward propagation for binary classification."""

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

    def forward_prop(self, X):
        """Calculate forward propagation for the neuron.

        Args:
            X: Input data with shape (nx, m).

        Returns:
            The activated output.
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A
