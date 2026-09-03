#!/usr/bin/env python3
"""Forward propagation through a pooling layer."""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Reduce spatial windows independently for each image and channel."""
    m, h_prev, w_prev, channels = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    h_new = (h_prev - kh) // sh + 1
    w_new = (w_prev - kw) // sw + 1
    output = np.zeros((m, h_new, w_new, channels))
    reduce = np.max if mode == 'max' else np.mean
    for row in range(h_new):
        for col in range(w_new):
            region = A_prev[:, row * sh:row * sh + kh,
                            col * sw:col * sw + kw, :]
            output[:, row, col, :] = reduce(region, axis=(1, 2))
    return output
