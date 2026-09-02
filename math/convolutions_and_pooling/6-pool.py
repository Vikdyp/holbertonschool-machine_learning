#!/usr/bin/env python3
"""Perform spatial pooling on batches of images."""

import numpy as np


def pool(images, kernel_shape, stride, mode="max"):
    """Return max or average pooling over image spatial dimensions."""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    oh = int((h - kh) / sh) + 1
    ow = int((w - kw) / sw) + 1
    output = np.zeros((m, oh, ow, c))

    for i in range(oh):
        for j in range(ow):
            row = i * sh
            col = j * sw
            region = images[:, row:row + kh, col:col + kw, :]
            if mode == "max":
                output[:, i, j, :] = np.max(region, axis=(1, 2))
            else:
                output[:, i, j, :] = np.mean(region, axis=(1, 2))

    return output
