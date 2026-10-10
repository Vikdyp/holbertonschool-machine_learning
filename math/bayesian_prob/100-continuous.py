#!/usr/bin/env python3
"""Calculate interval probabilities under a continuous binomial posterior."""
from scipy import special


def posterior(x, n, p1, p2):
    """Integrate the beta posterior resulting from a uniform prior."""
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that is greater than or "
                         "equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(p1) is not float or not 0 <= p1 <= 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if type(p2) is not float or not 0 <= p2 <= 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")
    return (special.betainc(x + 1, n - x + 1, p2)
            - special.betainc(x + 1, n - x + 1, p1))
