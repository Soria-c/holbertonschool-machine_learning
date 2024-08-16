#!/usr/bin/env python3
"""Initialize Gaussian Process"""

import numpy as np


class GaussianProcess:
    """
    Represents a noiseless 1D Gaussian process
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Constructor
        """
        self.X = X_init
        self.Y = Y_init
        self.t = X_init.shape[0]
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        """
        return (self.sigma_f ** 2) *\
            np.exp(-((X1-X2.T) ** 2) / (2*(self.l ** 2)))
