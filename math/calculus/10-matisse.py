#!/usr/bin/env python3
"""
Module that provides a function to compute the derivative
of a polynomial represented by a list of coefficients.
"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial.

    Args:
        poly (list): list of coefficients where index represents power of x

    Returns:
        list: coefficients of the derivative polynomial
        None: if poly is not valid
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if not all(isinstance(c, (int, float)) for c in poly):
        return None

    if len(poly) == 1:
        return [0]

    derivative = []
    for power in range(1, len(poly)):
        derivative.append(power * poly[power])

    while len(derivative) > 1 and derivative[-1] == 0:
        derivative.pop()

    return derivative
