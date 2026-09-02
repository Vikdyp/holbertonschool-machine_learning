#!/usr/bin/env python3
"""Convolve grayscale images with custom symmetric padding."""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Return a convolution using the requested height and width padding."""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    oh = h + (2 * ph) - kh + 1
    ow = w + (2 * pw) - kw + 1
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)))
    output = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            region = padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
