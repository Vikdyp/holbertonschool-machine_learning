#!/usr/bin/env python3
"""
This module provides a function that concatenates
two numpy.ndarrays along a given axis.
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two numpy.ndarrays along the given axis
    """
    return np.concatenate((mat1, mat2), axis=axis)
