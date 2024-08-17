#!/usr/bin/env python3
"""Mean and Covariance"""

import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)

    X_C = X - mean
    cov = np.dot(X_C.T, X_C) / (X.shape[0] - 1)
    return mean, np.cov(X.T, bias=False)
