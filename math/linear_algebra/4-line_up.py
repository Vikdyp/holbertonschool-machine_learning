#!/usr/bin/env python3
"""module that define add_arrays function"""


def add_arrays(arr1, arr2):
    """function that add arrays"""
    if len(arr1) != len(arr2):
        return None

    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])

    return result
