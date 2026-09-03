#!/usr/bin/env python3
"""Backward propagation through a pooling layer."""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Accumulate gradients from maximum or average pooling windows."""
    kh, kw = kernel_shape
    sh, sw = stride
    _, h_new, w_new, _ = dA.shape
    dA_prev = np.zeros(A_prev.shape)
    for row in range(h_new):
        for col in range(w_new):
            top, left = row * sh, col * sw
            gradient = dA[:, row, col, :][:, None, None, :]
            if mode == 'max':
                region = A_prev[:, top:top + kh, left:left + kw, :]
                mask = region == np.max(region, axis=(1, 2), keepdims=True)
                contribution = mask * gradient
            else:
                contribution = gradient / (kh * kw)
            dA_prev[:, top:top + kh, left:left + kw, :] += contribution
    return dA_prev
