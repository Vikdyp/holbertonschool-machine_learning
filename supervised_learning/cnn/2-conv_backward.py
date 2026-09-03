#!/usr/bin/env python3
"""Backward propagation through a convolutional layer."""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Return gradients for inputs, convolution kernels, and biases."""
    _, h_prev, w_prev, _ = A_prev.shape
    kh, kw, _, c_new = W.shape
    _, h_new, w_new, _ = dZ.shape
    sh, sw = stride
    ph = pw = 0
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    dA = np.zeros(padded.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for row in range(h_new):
        for col in range(w_new):
            top, left = row * sh, col * sw
            region = padded[:, top:top + kh, left:left + kw, :]
            for channel in range(c_new):
                gradient = dZ[:, row, col, channel][:, None, None, None]
                dA[:, top:top + kh, left:left + kw, :] += (
                    gradient * W[:, :, :, channel])
                dW[:, :, :, channel] += np.sum(region * gradient, axis=0)
    return dA[:, ph:ph + h_prev, pw:pw + w_prev, :], dW, db
