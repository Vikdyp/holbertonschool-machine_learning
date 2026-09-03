#!/usr/bin/env python3
"""Normal distribution using the exercise's mathematical approximations."""


class Normal:
    """Represent a normal distribution by its mean and standard deviation."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize parameters or estimate population moments from data."""
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = float(sum(data) / len(data))
            variance = sum((value - self.mean) ** 2 for value in data)
            variance /= len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """Convert a value to its standardized score."""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Convert a standardized score to the original measurement scale."""
        return self.mean + z * self.stddev

    def pdf(self, x):
        """Return the normal probability density at x."""
        factor = self.stddev * (2 * 3.1415926536) ** 0.5
        return 2.7182818285 ** (-self.z_score(x) ** 2 / 2) / factor

    def cdf(self, x):
        """Approximate cumulative probability with the required erf series."""
        value = self.z_score(x) / 2 ** 0.5
        erf = 2 / 3.1415926536 ** 0.5 * (
            value - value ** 3 / 3 + value ** 5 / 10
            - value ** 7 / 42 + value ** 9 / 216)
        return (1 + erf) / 2
