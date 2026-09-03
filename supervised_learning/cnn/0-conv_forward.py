#!/usr/bin/env python3
"""Forward propagation through a convolutional layer."""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Convolve NHWC inputs, add biases, and apply the activation."""
    m, h_prev, w_prev, _ = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    ph = pw = 0
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    h_new = (h_prev + 2 * ph - kh) // sh + 1
    w_new = (w_prev + 2 * pw - kw) // sw + 1
    padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    Z = np.zeros((m, h_new, w_new, c_new))
    for row in range(h_new):
        for col in range(w_new):
            region = padded[:, row * sh:row * sh + kh,
                            col * sw:col * sw + kw, :]
            for channel in range(c_new):
                Z[:, row, col, channel] = np.sum(
                    region * W[:, :, :, channel], axis=(1, 2, 3))
    return activation(Z + b)
