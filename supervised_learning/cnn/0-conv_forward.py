#!/usr/bin/env python3
"""Convolutional Forward Prop"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a
    convolutional layer of a neural network
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape

    sh, sw = stride

    if (padding == "valid"):
        pad = (0, 0)
    elif (padding == "same"):
        pad = ((sh * (h_prev - 1) -
                h_prev + kh) // 2,
               (sw * (w_prev - 1) -
                w_prev + kw) // 2)
    images = np.pad(A_prev, pad_width=((0, 0),
                                       (pad[0], pad[0]),
                                       (pad[1], pad[1]),
                                       (0, 0)))
    oh = ((h_prev - kh +
          (2 * pad[0])) // sh) + 1
    ow = ((w_prev - kw +
          (2 * pad[1])) // sw) + 1

    Z = np.zeros(shape=(m, oh, ow, c_new))

    for f in range(c_new):
        for h in range(oh):
            for w in range(ow):
                X_slice = images[:, (h*sh):(h*sh)+kh,
                                    (w*sw):(w*sw)+kw, :]
                Z[:, h, w, f] = np.sum(X_slice * W[:, :, :, f], axis=(1, 2, 3))
    return activation(Z + b)
