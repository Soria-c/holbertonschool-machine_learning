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
            np.exp(-((X1 - X2.T) ** 2) / (2 * (self.l ** 2)))

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation of points in a Gaussian
        process
        """
        # Kernel of the training data and the sample points
        K_s = self.kernel(self.X, X_s)

        # Kernel of the sample points with themselves
        K_ss = self.kernel(X_s, X_s)

        # Calculate the inverse of the kernel matrix of the training data
        K_inv = np.linalg.inv(self.K)

        # Predict the mean for the sample points
        mu = K_s.T.dot(K_inv).dot(self.Y).flatten()

        # Predict the variance for the sample points
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma = np.diag(cov_s)

        return mu, sigma

    def update(self, X_new, Y_new):
        """
        Updates the Gaussian Process with a new data point

        Parameters:
        - X_new: numpy.ndarray of shape (1,), new sample point
        - Y_new: numpy.ndarray of shape (1,), new sample function value

        Updates:
        - X: Updated training input data with X_new
        - Y: Updated training output data with Y_new
        - K: Updated kernel matrix with the new data
        """
        # Update the training data with the new sample
        self.X = np.vstack((self.X, X_new.reshape(1, -1)))
        self.Y = np.vstack((self.Y, Y_new.reshape(1, -1)))

        # Recompute the covariance matrix with the new data
        # self.K = self.kernel(self.X, self.X) + 1e-8 * np.eye(len(self.X))
        self.K = self.kernel(self.X, self.X)
