#!/usr/bin/env python3
"""Perform same convolution on batches of grayscale images."""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Return a same convolution of grayscale images with a kernel."""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph = int(np.ceil((kh - 1) / 2))
    pw = int(np.ceil((kw - 1) / 2))
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)))
    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            region = padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
