#!/usr/bin/env python3
"""LSTM Cell"""
import numpy as np


class LSTMCell:
    """LSTM Cell class"""
    def __init__(self, i, h, o):
        """
        Constructor for the LSTMCell class.
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """
        Perform forward propagation for one time step of the LSTM.
        """
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        f_t = self.sigmoid(np.dot(concat_input, self.Wf) + self.bf)

        u_t = self.sigmoid(np.dot(concat_input, self.Wu) + self.bu)

        c_tilde = np.tanh(np.dot(concat_input, self.Wc) + self.bc)

        c_next = f_t * c_prev + u_t * c_tilde

        o_t = self.sigmoid(np.dot(concat_input, self.Wo) + self.bo)

        h_next = o_t * np.tanh(c_next)

        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, c_next, y
