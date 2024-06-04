#!/usr/bin/env python3
"""Pooling Back Prop"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a
    pooling layer of a neural network
    """
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    partials = np.zeros_like(A_prev)
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c):
                    if mode == 'avg':
                        avg_dA = dA[i, h, w, ch] / kh / kw
                        partials[i, (h*sh):(h*sh)+kh,
                                    (w*sw):(w*sw)+kw, ch] +=\
                            (np.ones((kh, kw)) * avg_dA)
                    else:
                        slc = A_prev[i, (h*sh):(h*sh)+kh, (w*sw):(w*sw)+kw, ch]
                        mask = (slc == np.max(slc))
                        partials[i, (h*sh):(h*sh)+kh,
                                    (w*sw):(w*sw)+kw, ch] +=\
                            (mask * dA[i, h, w, ch])
    return partials
