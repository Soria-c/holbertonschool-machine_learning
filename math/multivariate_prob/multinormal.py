#!/usr/bin/env python3
"""Initialize Multinormal"""

import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution
    """
    def __init__(self, data):
        """
        Constructor
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        _, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)

        data_c = data - self.mean
        self.cov = np.dot(data_c, data_c.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        n = self.mean.shape[0]

        if x.shape != (n, 1):
            raise ValueError(f"x must have the shape ({n}, 1)")

        x_c = x - self.mean
        i_cov = np.linalg.inv(self.cov)
        d_cov = np.linalg.det(self.cov)

        a = 1 / (((2 * np.pi) ** (n / 2)) * d_cov ** 0.5)
        b = np.exp(
                -0.5 * np.dot(np.dot(x_c.T, i_cov), x_c))
        return float(a * b)
