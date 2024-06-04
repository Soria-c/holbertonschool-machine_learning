#!/usr/bin/env python3
"""Convolutional Back Prop"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a
    convolutional layer of a neural network
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    m, h_new, w_new, c_new = dZ.shape
    sh, sw = stride

    if (padding == "valid"):
        pad = (0, 0)
    elif (padding == "same"):
        pad = (1 + (sh * (h_prev - 1) -
               h_prev + kh) // 2,
               1 + (sw * (w_prev - 1) -
               w_prev + kw) // 2)
    images = np.pad(A_prev, pad_width=((0, 0),
                                       (pad[0], pad[0]),
                                       (pad[1], pad[1]),
                                       (0, 0)))
    dW = np.zeros(shape=W.shape)
    da = np.zeros(shape=A_prev.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    dA = np.pad(da, pad_width=((0, 0),
                               (pad[0], pad[0]),
                               (pad[1], pad[1]),
                               (0, 0)))
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for k in range(c_new):
                    filter = W[:, :, :, k]
                    dz = dZ[i, h, w, k]
                    slc = images[i, (h*sh):(h*sh)+kh, (w*sw):(w*sw)+kw, :]
                    dW[:, :, :, k] += slc * dz
                    dA[i, (h*sh):(h*sh)+kh,
                       (w*sw):(w*sw)+kw, :] += (dz * filter)
    if padding == 'same':
        dA = dA[:, pad[0]: -pad[0], pad[1]: -pad[1], :]
    return dA, dW, db
