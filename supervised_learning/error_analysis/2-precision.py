#!/usr/bin/env python3
"""Calculate class precision values from a confusion matrix."""

import numpy as np


def precision(confusion):
    """Return the precision of every class in a confusion matrix."""
    return np.diag(confusion) / np.sum(confusion, axis=0)
