#!/usr/bin/env python3
"""Learning Rate Decay"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Updates the learning rate using inverse time decay
    ```
    Parameters:
    -----------
    Z: numpy.ndarray(m, n)
        m is the number of data points
        n is the number of features in Z
    gamma: numpy.ndarray(1, n)
        containsthe scales used for batch normalization
    beta: numpy.ndarray(1, n)
        contains the offsets used for batch normalization
    epsilon: float
        small number used to avoid division by zero
    Return:
    -------
    normalized Z matrix
    """
    mean = np.mean(Z, axis=0)
    variance = np.std(Z, axis=0)**2
    Z_norm = (Z - mean) / ((variance + epsilon) ** 0.5)
    return gamma * Z_norm + beta
