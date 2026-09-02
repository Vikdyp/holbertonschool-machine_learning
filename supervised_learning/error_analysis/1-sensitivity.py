#!/usr/bin/env python3
"""Calculate class sensitivity values from a confusion matrix."""

import numpy as np


def sensitivity(confusion):
    """Return the sensitivity of every class in a confusion matrix."""
    return np.diag(confusion) / np.sum(confusion, axis=1)
