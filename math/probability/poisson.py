#!/usr/bin/env python3
"""Poisson distribution estimated from data or an explicit rate."""


class Poisson:
    """Represent counts of independent events in a fixed interval."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize the event rate, optionally estimating it from data."""
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Return the probability of exactly k events."""
        k = int(k)
        if k < 0:
            return 0
        factorial = 1
        for number in range(2, k + 1):
            factorial *= number
        return 2.7182818285 ** (-self.lambtha) * self.lambtha ** k / factorial

    def cdf(self, k):
        """Return the probability of at most k events."""
        k = int(k)
        if k < 0:
            return 0
        return sum(self.pmf(number) for number in range(k + 1))
