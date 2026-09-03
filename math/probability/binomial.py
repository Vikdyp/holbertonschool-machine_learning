#!/usr/bin/env python3
"""Binomial distribution with parameters estimated by moments."""


class Binomial:
    """Represent successes in a fixed number of independent trials."""

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize parameters or estimate them from observed counts."""
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if not 0 < p < 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data) / len(data)
            variance = sum((value - mean) ** 2 for value in data) / len(data)
            probability = 1 - variance / mean
            self.n = round(mean / probability)
            self.p = float(mean / self.n)

    def pmf(self, k):
        """Return the mass at k successes, or zero outside the support."""
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        combinations = 1
        for number in range(1, min(k, self.n - k) + 1):
            combinations = combinations * (self.n - number + 1) // number
        return combinations * self.p ** k * (1 - self.p) ** (self.n - k)

    def cdf(self, k):
        """Sum mass through k, returning zero outside the exercise's range."""
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        return sum(self.pmf(number) for number in range(k + 1))
