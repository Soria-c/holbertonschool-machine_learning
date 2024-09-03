#!/usr/bin/env python3
"""Bidirectional Cell"""
import numpy as np


class BidirectionalCell:
    def __init__(self, i, h, o):
        """
        Initialize a bidirectional RNN cell.
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.bhf = np.zeros((1, h))

        self.Whb = np.random.normal(size=(i + h, h))
        self.bhb = np.zeros((1, h))

        self.Wy = np.random.normal(size=(2 * h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculate the hidden state in the forward direction for one time step.
        """
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(np.dot(concat_input, self.Whf) + self.bhf)

        return h_next
