#!/usr/bin/env python3
"""BIRNN forward propagation"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN.
    """
    t, m, i = X.shape
    _, h = h_0.shape

    H_forward = []
    H_backward = []

    h_prev = h_0
    for time_step in range(t):
        h_prev = bi_cell.forward(h_prev, X[time_step])
        H_forward.append(h_prev)

    H_forward = np.array(H_forward)

    h_next = h_t
    for time_step in reversed(range(t)):
        h_next = bi_cell.backward(h_next, X[time_step])
        H_backward.insert(0, h_next)

    H_backward = np.array(H_backward)

    H = np.concatenate((H_forward, H_backward), axis=-1)

    Y = bi_cell.output(H)

    return H, Y
