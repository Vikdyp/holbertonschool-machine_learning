#!/usr/bin/env python3
"""Combine binomial likelihoods and discrete priors."""
import numpy as np


def intersection(x, n, P, Pr):
    """Combine binomial likelihoods and discrete priors."""
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that is greater than or "
                         "equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not np.all((Pr >= 0) & (Pr <= 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    coefficient = 1
    for i in range(1, min(x, n - x) + 1):
        coefficient = coefficient * (n - i + 1) // i
    probabilities = float(coefficient) * P ** x * (1 - P) ** (n - x)
    return probabilities * Pr
