#!/usr/bin/env python3
"""
Module that provides a function to compute the integral
of a polynomial represented by a list of coefficients.
"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.

    Args:
        poly (list): list of coefficients where index represents power of x
        C (int): integration constant

    Returns:
        list: coefficients of the integral polynomial
        None: if inputs are not valid
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, int):
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None

    integral = [C]

    for power, coef in enumerate(poly):
        value = coef / (power + 1)
        if value.is_integer():
            value = int(value)
        integral.append(value)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
