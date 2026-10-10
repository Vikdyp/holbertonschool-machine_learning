#!/usr/bin/env python3
"""Represent a multivariate normal distribution estimated from observations."""
import numpy as np


class MultiNormal:
    """Estimate a Gaussian distribution and evaluate its density."""

    def __init__(self, data):
        """Estimate a column mean and sample covariance from column samples."""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1, keepdims=True)
        centered = data - self.mean
        self.cov = centered @ centered.T / (data.shape[1] - 1)

    def pdf(self, x):
        """Evaluate the probability density at a single column data point."""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))
        centered = x - self.mean
        exponent = (centered.T @ np.linalg.inv(self.cov) @ centered).item()
        denominator = np.sqrt((2 * np.pi) ** d * np.linalg.det(self.cov))
        return np.exp(-0.5 * exponent) / denominator
