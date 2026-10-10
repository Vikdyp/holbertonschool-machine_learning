#!/usr/bin/env python3
"""Compute Gaussian neighbor affinities and their Shannon entropy."""
import numpy as np


def HP(Di, beta):
    """Return bit entropy and normalized affinities for one observation."""
    probabilities = np.exp(-(Di - np.min(Di)) * beta)
    total = np.sum(probabilities)
    Pi = probabilities / total
    positive = Pi > 0
    Hi = -np.sum(Pi[positive] * np.log2(Pi[positive]))
    return Hi, Pi
