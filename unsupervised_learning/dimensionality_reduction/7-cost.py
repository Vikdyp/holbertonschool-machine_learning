#!/usr/bin/env python3
"""Evaluate the KL divergence between high- and low-dimensional affinities."""
import numpy as np


def cost(P, Q):
    """Return KL divergence with protected logarithms and unchanged inputs."""
    p = np.maximum(P, 1e-12)
    q = np.maximum(Q, 1e-12)
    return np.sum(p * np.log(p / q))
