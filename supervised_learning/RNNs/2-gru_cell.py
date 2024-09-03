#!/usr/bin/env python3
"""GRU Cell"""
import numpy as np


class GRUCell:
    """GRU Cell class"""
    def __init__(self, i, h, o):
        """
        Constructor for the GRUCell class.
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.
        """
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        z_t = self.sigmoid(np.dot(concat_input, self.Wz) + self.bz)

        r_t = self.sigmoid(np.dot(concat_input, self.Wr) + self.br)

        concat_reset = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.dot(concat_reset, self.Wh) + self.bh)

        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y
