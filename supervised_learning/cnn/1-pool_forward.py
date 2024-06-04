#!/usr/bin/env python3
"""Convolutional Forward Prop"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over
    a pooling layer of a neural network
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape

    sh, sw = stride

    if mode == "max":
        pool = np.max
    elif mode == "avg":
        pool = np.mean

    pad = (0, 0)

    images = np.pad(A_prev, pad_width=((0, 0),
                                       (pad[0], pad[0]),
                                       (pad[1], pad[1]),
                                       (0, 0)))
    oh = ((h_prev - kh +
          (2 * pad[0])) // sh) + 1
    ow = ((w_prev - kw +
          (2 * pad[1])) // sw) + 1

    Z = np.zeros(shape=(m, oh, ow, c_prev))

    for h in range(oh):
        for w in range(ow):
            X_slice = images[:, (h*sh):(h*sh)+kh,
                                (w*sw):(w*sw)+kw, :]
            Z[:, h, w, :] = pool(X_slice, axis=(1, 2))
    return Z
