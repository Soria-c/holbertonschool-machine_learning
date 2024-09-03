import numpy as np
"""RNN Cell"""


class RNNCell:
    """RNN Cell class"""
    def __init__(self, i, h, o):
        """
        Constructor for the RNNCell class.
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.
        """
        h_x_concat = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(np.dot(h_x_concat, self.Wh) + self.bh)

        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y

    def softmax(self, z):
        """
        Softmax activation function.
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
