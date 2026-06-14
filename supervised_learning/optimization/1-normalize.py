#!/usr/bin/env python3
"""Normalize a matrix with provided standardization constants."""


def normalize(X, m, s):
    """Return X standardized with mean m and standard deviation s."""
    return (X - m) / s
