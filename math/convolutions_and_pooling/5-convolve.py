#!/usr/bin/env python3
"""Convolve batches of images with multiple kernels."""

import numpy as np


def convolve(images, kernels, padding="same", stride=(1, 1)):
    """Return a convolution with one output channel per kernel."""
    m, h, w, _ = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == "valid":
        ph = pw = 0
    else:
        ph, pw = padding

    oh = int((h + (2 * ph) - kh) / sh) + 1
    ow = int((w + (2 * pw) - kw) / sw) + 1
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    output = np.zeros((m, oh, ow, nc))

    for i in range(oh):
        for j in range(ow):
            row = i * sh
            col = j * sw
            region = padded[:, row:row + kh, col:col + kw, :]
            products = region[..., np.newaxis] * kernels
            output[:, i, j, :] = np.sum(products, axis=(1, 2, 3))

    return output
