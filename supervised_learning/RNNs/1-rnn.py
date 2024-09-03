#!/usr/bin/env python3
"""Forward propragation"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN.
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))

    H[0] = h_0

    for time_step in range(t):
        h_prev = H[time_step]
        x_t = X[time_step]

        h_next, y = rnn_cell.forward(h_prev, x_t)

        H[time_step + 1] = h_next
        Y[time_step] = y

    return H, Y
